from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
import random
import copy
import numpy as np
import torchvision
from collections import OrderedDict
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation


class RLT_dimp(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'


        # Initialize network
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net
        self.neg_net = copy.deepcopy(self.net) if self.params.get('neg_net', False) else None

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox'] # state: gt
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2]) # center position (row, col)
        self.target_sz = torch.Tensor([state[3], state[2]]) # target size (height, width)

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]]) # image size (height, width)
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)

        # [---------------- More discriminative feature learning  (new) ----------------]
        # generate_init_more_samples -> init_backbone_feat_more (more samples)

        # Extract and transform sample
        if self.params.get('init_net_more_learn', False) or self.params.get('init_iou_more_learn', False):
            init_backbone_feat_more = self.generate_init_more_samples(im) # augmented samples
            if (self.params.get('init_net_more_learn', False) == False) or (self.params.get('init_iou_more_learn', False) == False):
                init_backbone_feat = OrderedDict()
                for key, value in init_backbone_feat_more.items():
                    init_backbone_feat[key] = value[self.idx_original]
        else:
            init_backbone_feat = self.generate_init_samples(im) # augmented samples

        # Initialize classifier
        if self.params.get('init_net_more_learn', False):
            self.init_classifier(init_backbone_feat_more, self.transforms_more) # training using samples (maybe positive sets)
        else:
            self.init_classifier(init_backbone_feat, self.transforms) # training using samples (maybe positive sets)

        # Initialize IoUNet
        if self.params.get('use_iou_net', True):
            if self.params.get('init_iou_more_learn', False):
                self.init_iou_net(init_backbone_feat_more, self.transforms_more) # training using samples (maybe positive sets)
            else:
                self.init_iou_net(init_backbone_feat, self.transforms)

        self.old_flag = 'normal'

        if self.params.get('erasing_mode', False):
        # random erasing for measuring uncertainty
            lower_scale = self.params.get('lower_scale',0.02)
            upper_scale = self.params.get('upper_scale',0.1)
            lower_ratio = self.params.get('lower_ratio',0.3)
            upper_ratio = self.params.get('upper_ratio',3.3)
            self.random_erasing = torchvision.transforms.RandomErasing(p=1.0, scale=(lower_scale, upper_scale), ratio=(lower_ratio, upper_ratio), value=0, inplace=False)

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num
        if self.frame_num == 2:
            self.search_global = False
            self.search_random = False
            self.redetection = False
            self.cnt_track = 0
            self.cnt_empty = 0

        # Convert image
        im = numpy_to_torch(image)
        # print('.')

        # [---------------- enhanced short-term tracking using random erasing (new) ----------------]
        if (self.search_global == False) and (self.search_random == False): # previous frame -> found

            if self.params.get('erasing_mode', False) and self.cnt_track % self.params.get('erasing_cnt', 1) == 0:
                backbone_feat, sample_coords, im_patches = self.extract_backbone_features_with_erasing(im,
                                                                                                       self.get_centered_sample_pos(),
                                                                                                       self.target_scale * self.params.scale_factors,
                                                                                                       self.img_sample_sz)
                test_x = self.get_classification_features(backbone_feat)  # Batch x 512 x 18 x 18
                sample_pos, sample_scales = self.get_sample_location(sample_coords)
                scores_raw = self.classify_target(test_x)  # Batch x 19 x 19
                average_scores_raw = scores_raw.mean(dim=0).unsqueeze(0)
                translation_vec, scale_ind, s, flag = self.localize_target(average_scores_raw, sample_pos, sample_scales)
                if self.params.get('use_original_pos', False):
                    original_scores_raw = scores_raw[-2:-1, :, :, :]
                    translation_vec, scale_ind, s, _ = self.localize_target(original_scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec

            else:
                # original
                # Extract backbone features

                backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                                          self.target_scale * self.params.scale_factors,
                                                                                          self.img_sample_sz)
                # Extract classification features
                test_x = self.get_classification_features(backbone_feat)  # 512 x 18 x 18

                # Location of sample
                sample_pos, sample_scales = self.get_sample_location(sample_coords)

                # Compute classification scores
                scores_raw = self.classify_target(test_x)  # 19 x 19

                # Localize the target
                translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec


            if self.params.get('redetection_now', False) and flag == 'not_found':
                self.search_global = False
                self.search_random = True
                self.redetection = True
                # self.cnt_empty = 1




        # [---------------- Global re-detection with random property (new) ----------------]
        if self.search_global or self.search_random:
            # find candidate (global: sliding window, random: random window

            # print('random search')
            search_pos = self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * self.img_support_sz / (2 * self.feature_sz)
            search_pos_sample = search_pos.clone()
            list_search_pos = [search_pos]

            all_pos = self.find_all_index(search_pos.clone(), self.target_scale * self.img_sample_sz, self.image_sz,
                                          self.params.get('redetection_global_search_flag', 1), 50, False)
            if self.search_random:
                if self.params.get('additional_candidate_adaptive', False):
                    num_add = int(self.params.get('additional_candidate_adaptive_ratio', 0.33) * len(all_pos))
                    if num_add < self.params.get('additional_candidate_adaptive_min', 1):
                        num_add = self.params.get('additional_candidate_adaptive_min', 1)
                    if num_add > self.params.get('additional_candidate_adaptive_max', 10):
                        num_add = self.params.get('additional_candidate_adaptive_max', 10)

                else:
                    num_add = self.params.get('additional_candidate_random', 0)
                idx_remain = [x for i, x in enumerate(np.random.permutation(len(all_pos))) if i < num_add] # random n value
                idx_remain = sorted(idx_remain)
            else:
                if self.params.get('global_search_memory_limit', 10000) < len(all_pos):
                    num_add = self.params.get('global_search_memory_limit', 10000)
                    idx_remain = [x for i, x in enumerate(np.random.permutation(len(all_pos))) if i < num_add] # random n value
                    idx_remain = sorted(idx_remain)
                else:
                    idx_remain = [x for x in range(len(all_pos))]

            for i in idx_remain:
                list_search_pos.append(all_pos[i])

            flag_batch = True

            if flag_batch: # batch version
                backbone_feat, sample_coords, im_patches = self.extract_backbone_features_with_multiple_search(im, list_search_pos, self.target_scale * self.params.scale_factors, self.img_sample_sz)
                test_x = self.get_classification_features(backbone_feat)  # 512 x 18 x 18
                sample_pos, sample_scales = self.get_sample_location(sample_coords)
                scores_raw = self.classify_target(test_x)  # 19 x 19

                if self.params.get('redetection_score_penalty', False):
                    dist1 = np.zeros(len(list_search_pos))
                    dist_max = math.sqrt(sum(self.image_sz ** 2))
                    for i in range(len(list_search_pos)):
                        dist1[i] = math.sqrt(sum((list_search_pos[i] - list_search_pos[0]) ** 2))
                    weight_penalty = 1.0 - self.params.get('redetection_score_penalty_alpha', 0.5) * (dist1 / dist_max) * math.exp(
                        - self.params.get('redetection_score_penalty_beta', 0.5) * (self.cnt_empty - 1))
                    for i in range(len(scores_raw)):
                        scores_raw[i] *= weight_penalty[i]

                if self.redetection:
                    for i in range(len(scores_raw)):
                        if i != 0: # not original
                            scores_raw[i] *= self.params.get('redetection_basic_penalty', 1.0)

                translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
                new_pos = sample_pos[scale_ind, :] + translation_vec


                s = torch.unsqueeze(s[scale_ind],0)
                sample_pos = torch.unsqueeze(sample_pos[scale_ind],0)
                scores_raw = torch.unsqueeze(scores_raw[scale_ind],0)
                sample_scales = torch.unsqueeze(sample_scales[scale_ind],0)
                sample_coords = torch.unsqueeze(sample_coords[scale_ind],0)
                test_x = torch.unsqueeze(test_x[scale_ind],0)

                backbone_feat_new = OrderedDict()
                for key, value in backbone_feat.items():
                    backbone_feat_new[key] = torch.unsqueeze(value[scale_ind],0)
                backbone_feat = backbone_feat_new

                if scale_ind.is_cuda:
                    scale_ind = torch.tensor(0).cuda()
                else:
                    scale_ind = torch.tensor(0)

            else:  # not batch version
                for i in range(len(list_search_pos)):
                    if i == 0: # original search
                        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, list_search_pos[i], self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        test_x = self.get_classification_features(backbone_feat)  # 512 x 18 x 18
                        sample_pos, sample_scales = self.get_sample_location(sample_coords)
                        scores_raw = self.classify_target(test_x)  # 19 x 19
                        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
                        new_pos = sample_pos[scale_ind, :] + translation_vec

                        list_backbone_feat = [backbone_feat]
                        list_sample_coords = [sample_coords]
                        list_test_x = [test_x]
                        list_sample_pos = [sample_pos]
                        list_sample_scales = [sample_scales]
                        list_scale_ind = [scale_ind]
                        list_s = [s]
                        list_flag = [flag]
                        list_new_pos = [new_pos]

                    else: # random search
                        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, list_search_pos[i], self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        test_x = self.get_classification_features(backbone_feat)  # 512 x 18 x 18
                        sample_pos, sample_scales = self.get_sample_location(sample_coords)
                        scores_raw = self.classify_target(test_x)  # 19 x 19

                        if self.params.get('redetection_score_penalty', False):
                            dist1 = math.sqrt(sum((list_search_pos[i] - list_search_pos[0])**2))
                            dist_max = math.sqrt(sum(self.image_sz**2))
                            weight_penalty = 1.0 - self.params.get('redetection_score_penalty_alpha', 0.5) * (dist1 / dist_max) * math.exp(- self.params.get('redetection_score_penalty_beta', 0.5) * (self.cnt_empty - 1))
                            scores_raw *= weight_penalty
                            # print(weight_penalty)

                        if self.redetection:
                            scores_raw *= self.params.get('redetection_basic_penalty', 1.0)

                        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
                        new_pos = sample_pos[scale_ind, :] + translation_vec

                        list_backbone_feat.append(backbone_feat)
                        list_sample_coords.append(sample_coords)
                        list_test_x.append(test_x)
                        list_sample_pos.append(sample_pos)
                        list_sample_scales.append(sample_scales)
                        list_scale_ind.append(scale_ind)
                        list_s.append(s)
                        list_flag.append(flag)
                        list_new_pos.append(new_pos)

                # find max s
                for i in range(len(list_search_pos)):
                    local_s = list_s[i]
                    local_scale_ind = list_scale_ind[i]
                    score_map = local_s[local_scale_ind, ...]
                    max_score = torch.max(score_map).item()  # confidence
                    if i == 0:
                        list_max_score = [max_score]
                    else:
                        list_max_score.append(max_score)
                find_i = list_max_score.index(max(list_max_score))

                backbone_feat = list_backbone_feat[find_i]
                sample_coords = list_sample_coords[find_i]
                test_x = list_test_x[find_i]
                sample_pos = list_sample_pos[find_i]
                sample_scales = list_sample_scales[find_i]
                scale_ind = list_scale_ind[find_i]
                s = list_s[find_i]
                flag = list_flag[find_i]
                new_pos = list_new_pos[find_i]


        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'
                if self.params.get('use_classifier', True):
                    self.update_state(new_pos)
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        if update_flag and self.params.get('update_classifier', False):
            if self.cnt_track >= self.params.get('no_update_early_redetection', 0): # early sample (redetected) is not updated
                # <Original>
                # Get train sample
                train_x = test_x[scale_ind:scale_ind+1, ...]

                # Create target_box and label for spatial sample
                target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

                # <Additional>
                # [------------------------ more discriminative feature learning (new)-------------------------]
                if self.params.get('track_net_more_learn', False) and s[scale_ind,...].max().item() > self.params.get('track_net_more_learn_score', True) and \
                        (self.params.get('track_net_more_learn_not_save', True) or (self.cnt_track % self.params.get('track_net_more_learn_cnt', 1) == 0) and self.cnt_track > 0): # first track (not used)
                    # conditions
                    # 1) score should be bigger than track_net_more_learn_score
                    # 2) track_net_more_learn is True
                    # 3) track_net_more_learn_not_save is True or track_net_more_learn_cnt condition is satisfied

                    more_search_pos = self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * self.img_support_sz / (2 * self.feature_sz)
                    list_more_search_pos = []
                    all_pos = self.find_all_index(more_search_pos.clone(), self.target_scale * self.img_sample_sz, self.image_sz,
                                                  self.params.get('track_net_more_learn_search_flag', 1), self.params.get('train_more_sample_limit', 5), False)
                    for i in reversed(range(len(all_pos))):

                        row_min = int(all_pos[i][0]) - int((self.target_sz[0] - 1) / 2)
                        row_max = int(all_pos[i][0]) + int((self.target_sz[0] - 1) / 2)
                        col_min = int(all_pos[i][1]) - int((self.target_sz[1] - 1) / 2)
                        col_max = int(all_pos[i][1]) + int((self.target_sz[1] - 1) / 2)

                        if (row_min < 0) or (col_min < 0) or (row_max >= int(self.image_sz[0])) or (col_max >= int(self.image_sz[1])):
                            del(all_pos[i])

                    num_add = self.params.get('additional_train_candidate', 0)
                    idx_remain = [x for i, x in enumerate(np.random.permutation(len(all_pos))) if i < num_add]  # random n value
                    idx_remain = sorted(idx_remain)

                    for i in idx_remain:
                        list_more_search_pos.append(all_pos[i])

                    if len(list_more_search_pos) > 0:
                        # image masking
                        row_min = int(self.pos[0]) - int((self.target_sz[0] - 1) / 2)
                        row_max = int(self.pos[0]) + int((self.target_sz[0] - 1) / 2)
                        col_min = int(self.pos[1]) - int((self.target_sz[1] - 1) / 2)
                        col_max = int(self.pos[1]) + int((self.target_sz[1] - 1) / 2)
                        im_mask = im.clone()
                        im_target = im_mask[:, :, row_min:row_max + 1, col_min:col_max + 1].clone()
                        im_mask[:, :, row_min:row_max + 1, col_min:col_max + 1] = 0
                        # self.tensor_image_save(im_mask[0], "im")

                        flag_contour_all = []
                        for i in range(len(list_more_search_pos)):
                            im_mask, flag_contour = self.im_masking(im_mask, im_target, list_more_search_pos[i], self.image_sz, [im_target.shape[2], im_target.shape[3]])
                            flag_contour_all.append(flag_contour)
                        # self.tensor_image_save(im_mask[0], "im_after")

                        backbone_feat_more, sample_coords_more, _ = self.extract_backbone_features_with_multiple_search(im, list_more_search_pos, self.target_scale * self.params.scale_factors, self.img_sample_sz)
                        train_x_more = self.get_classification_features(backbone_feat_more)
                        sample_pos_more, sample_scales_more = self.get_sample_location(sample_coords_more)
                        target_box_more = []
                        for i in range(len(sample_pos_more)):
                            target_box_more_local = self.get_iounet_box(list_more_search_pos[i], self.target_sz, sample_pos_more[i], sample_scales_more[i])
                            target_box_more_local = torch.unsqueeze(target_box_more_local, dim = 0)
                            if i == 0:
                                target_box_more = target_box_more_local
                            else:
                                target_box_more = torch.cat((target_box_more, target_box_more_local), dim = 0)

                        # Update the classifier model
                        self.update_classifier_more(train_x, target_box, learning_rate, s[scale_ind,...], train_x_more, target_box_more)
                    else:
                        self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])
                else:
                    self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])


        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]
        max_score = torch.max(score_map).item() # confidence

        # Visualize and set debug info
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        # ---------------- [re-detection flag] ----------------#
        if flag == 'not_found' and self.params.get('re_detection', False):
            self.cnt_track = 0
            if self.old_flag == 'not_found': # continuous
                self.cnt_empty += 1
                self.redetection = True
            else: # first not_found
                self.cnt_empty = 1
                self.redetection = True
                # self.terminate_pos =
            # print('self.cnt_empty: {}, self.pos: {}, self.feature_sz: {}, self.kernel_size: {}, self.target_scale: {}, self.img_support_sz: {}'.format(self.cnt_empty, self.pos, self.feature_sz, self.kernel_size, self.target_scale, self.img_support_sz))
            # print('.')
            if self.cnt_empty == 1:
                self.search_global = False
                self.search_random = True
            elif self.cnt_empty % self.params.get('cnt_global', 10000000) == 0:
                self.search_global = True
                self.search_random = False
            elif self.cnt_empty % self.params.get('cnt_random', 10000000) == 0:
                self.search_global = False
                self.search_random = True
            else:
                self.search_global = False
                self.search_random = False
        else:
            self.cnt_empty = 0
            self.search_global = False
            self.search_random = False
            self.redetection = False
            if self.old_flag == 'not_found':
                self.cnt_track = 1
            else:
                self.cnt_track += 1
        # print('[Frame: {}] previous state: {} / present state: {} / empty count: {} / track count: {}'.format(self.frame_num, self.old_flag, flag, self.cnt_empty, self.cnt_track))
        self.old_flag = flag


        # ---------------- [confidence] ----------------#
        flag_confidence = self.params.get('flag_confidence', 'none')
        if flag_confidence == 1: # basic
            confidence = max_score
        elif flag_confidence == 2: # upper/lower limit
            if max_score > 1:
                confidence = 1.0
            elif max_score < 0:
                confidence = 0.0
            else:
                confidence = max_score
        elif flag_confidence == 3: # consider not_found
            if max_score > 1:
                confidence = 1.0
            elif flag == 'not_found' or max_score < 0:
                confidence = 0.0
            else:
                confidence = max_score
        elif flag_confidence == 4: # 1.0 fix
            confidence = 1.0
        elif flag_confidence == 5: # 1 or 0
            if flag == 'not_found':
                confidence = 0.0
            else:
                confidence = 1.0
        elif flag_confidence == 6: # 1 or 0
            if flag == 'not_found':
                confidence = 0.0
            elif flag == 'uncertain':
                confidence = 0.5
            else:
                confidence = 1.0
        elif flag_confidence == 7: # 1 or 0
            if flag == 'not_found':
                confidence = 0.0
            elif flag == 'uncertain':
                confidence = 0.66
            elif flag == 'hard_negative':
                confidence = 0.33
            else:
                confidence = 1.0
        elif flag_confidence == 8: # 1 or 0
            if flag == 'not_found':
                confidence = 0.0
            elif flag == 'hard_negative':
                confidence = 0.66
            elif flag == 'uncertain':
                confidence = 0.33
            else:
                confidence = 1.0
        elif flag_confidence == 9:
            if flag == 'not_found':
                confidence = 0.0
            elif flag == 'hard_negative':
                confidence = 0.5
            else:
                confidence = 1.0
        else:
            confidence = max_score

        # print(flag)
        out = {'target_bbox': output_state,
               'confidence': confidence}



        return out


    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt()
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered."""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""

        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold: # 0.25
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')): # x
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')): # x
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1: # similar sample (0.8 * max1)
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold: # similar sample (0.5 * max1) and bigger than not_found_th(0.25)
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def extract_backbone_features_with_multiple_search(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        for i in range(len(pos)):
            im_patches_local, patch_coords_local = sample_patch_multiscale(im, pos[i], scales, sz,
                                                               mode=self.params.get('border_mode', 'replicate'),
                                                               max_scale_change=self.params.get('patch_max_scale_change', None))
            if i == 0:
                im_patches = im_patches_local
                patch_coords = patch_coords_local
            else:
                im_patches = torch.cat((im_patches, im_patches_local), dim = 0)
                patch_coords = torch.cat((patch_coords, patch_coords_local), dim = 0)

        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def extract_backbone_features_with_erasing(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        ims = [self.random_erasing(im_patches.squeeze(0)).unsqueeze(0) for i in range(self.params.get('num_erasing'))]
        ims.append(im_patches)
        im_patches = torch.cat(ims, dim=0)
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat



    def tensor_image_save(self, im, im_name):
        import matplotlib
        matplotlib.use('Agg')
        import torchvision

        pilTrans = torchvision.transforms.ToPILImage()
        pilImg = pilTrans(im/255.0)
        matplotlib.pyplot.imshow(pilImg)
        matplotlib.pyplot.savefig(im_name)

    def find_all_index(self, sample_pos, search_size, image_size, flag, num_limit = 50, delete_ori = False):
        all_pos = []
        init_pos = sample_pos.clone()
        if flag == 0:  # 0: 1/4, 1: half overlap, 2: non overlap
            row_gap = int(search_size[0] / 4)
            col_gap = int(search_size[1] / 4)
        elif flag == 1:
            row_gap = int(search_size[0] / 2)
            col_gap = int(search_size[1] / 2)
        else:
            row_gap = int(search_size[0])
            col_gap = int(search_size[1])

        row_all_idx = []
        col_all_idx = []
        for i in range(-num_limit, num_limit):
            row_val = init_pos[0] + i * row_gap
            col_val = init_pos[1] + i * col_gap
            if row_val >= 0 and row_val < image_size[0]:
                row_all_idx.append(int(row_val))
            if col_val >= 0 and col_val < image_size[1]:
                col_all_idx.append(int(col_val))

        for i in range(len(row_all_idx)):
            for j in range(len(col_all_idx)):
                if delete_ori:
                    sample_pos[0] = row_all_idx[i]
                    sample_pos[1] = col_all_idx[j]
                    all_pos.append(sample_pos.clone())
                else:
                    if not (row_all_idx[i] == int(init_pos[0]) and col_all_idx[j] == int(init_pos[1])):
                        sample_pos[0] = row_all_idx[i]
                        sample_pos[1] = col_all_idx[j]
                        all_pos.append(sample_pos.clone())

        return all_pos

    def im_masking(self, im, im_target, now_pos, image_sz, target_sz):

        flag_contour = False
        im_mask = im.clone()
        row_min = int(now_pos[0]) - int((target_sz[0] - 1) / 2)
        row_max = int(now_pos[0]) + int((target_sz[0] - 1) / 2)
        col_min = int(now_pos[1]) - int((target_sz[1] - 1) / 2)
        col_max = int(now_pos[1]) + int((target_sz[1] - 1) / 2)

        if row_min < 0:
            row_min_gap = -row_min
            row_min = 0
            flag_contour = True
        else:
            row_min_gap = 0

        if col_min < 0:
            col_min_gap = -col_min
            col_min = 0
            flag_contour = True
        else:
            col_min_gap = 0

        if row_max >= int(image_sz[0]):
            row_max_gap = target_sz[0] - row_max + int(image_sz[0]) - 1
            row_max = int(image_sz[0]) - 1
            flag_contour = True
        else:
            row_max_gap = target_sz[0]

        if col_max >= int(image_sz[1]):
            col_max_gap = target_sz[1] - col_max + int(image_sz[1]) - 1
            col_max = int(image_sz[1]) - 1
            flag_contour = True
        else:
            col_max_gap = target_sz[1]
        if ((row_max + 1 - row_min) == (row_max_gap - row_min_gap)) and ((col_max + 1 - col_min) == (col_max_gap - col_min_gap)):
            if self.params.get('init_blending', 0) != 0:
                blend_ratio = self.params.get('init_blending', 0)
                im_mask[:, :, row_min:row_max + 1, col_min:col_max + 1] = blend_ratio * im_mask[:, :, row_min:row_max + 1, col_min:col_max + 1] + \
                                                                          (1.0-blend_ratio) * im_target[:, :, row_min_gap:row_max_gap, col_min_gap:col_max_gap]  # 16, 40
            else:
                im_mask[:, :, row_min:row_max + 1, col_min:col_max + 1] = im_target[:, :, row_min_gap:row_max_gap, col_min_gap:col_max_gap]  # 16, 40

        # self.tensor_image_save(im_mask[0], "im_mask")

        return im_mask, flag_contour

    def generate_init_more_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)


        # [------------------- more samples (new) ---------------------]

        idx_original = [x for x in range(len(im_patches))]

        # aug_expansion
        if self.params.get('init_more_learn_expand_searching_size', False) == False:
            aug_expansion_sz = self.img_sample_sz.clone()
            aug_output_sz = None

        # find all position (present pos, searching size, flag)
        all_pos = self.find_all_index(self.init_sample_pos.clone(), aug_expansion_sz, self.image_sz, self.params.get('init_more_learn_flag', 1), self.params.get('init_more_sample_limit', 5), False)

        # image masking
        # for i in range(len(im_patches)):
        #     tensor_image_save(im_patches[i], "out" + str(i) + ".png")
        # tensor_image_save(im[0], "im")
        row_min = int(self.pos[0]) - int((self.target_sz[0] - 1)/2)
        row_max = int(self.pos[0]) + int((self.target_sz[0] - 1)/2)
        col_min = int(self.pos[1]) - int((self.target_sz[1] - 1)/2)
        col_max = int(self.pos[1]) + int((self.target_sz[1] - 1)/2)
        im_target = im[:, :, row_min:row_max + 1, col_min:col_max + 1].clone()
        im[:, :, row_min:row_max + 1, col_min:col_max + 1] = 0
        # tensor_image_save(im[0], "im")

        # transformation
        self.transforms_ori = self.transforms.copy()
        all_transforms = []
        all_transforms.extend(self.transforms)
        if self.params.get('init_more_learn_no_transform', False):
            new_transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
            # new_transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])
        else:
            new_transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
            if 'fliplr' in augs and augs['fliplr']:
                new_transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
            if 'blur' in augs:
                new_transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
            if 'scale' in augs:
                new_transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
            if 'rotate' in augs:
                new_transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])
            # new_transforms = self.transforms.copy()
        cnt = 0
        for i in range(len(all_pos)):
            im_mask, flag_contour = self.im_masking(im, im_target, all_pos[i], self.image_sz, [im_target.shape[2], im_target.shape[3]])
            if not flag_contour:
                cnt += 1
                im_patches_new = sample_patch_transformed(im_mask, all_pos[i], self.init_sample_scale, aug_expansion_sz, new_transforms)
                im_patches = torch.cat((im_patches, im_patches_new), dim=0)

        for i in range(cnt):
            all_transforms.extend(new_transforms)
        self.transforms = all_transforms

        if len(im_patches) > self.params.get('init_more_learn_memory_limit', len(idx_original)):
            num_remain = self.params.get('init_more_learn_memory_limit', len(idx_original)) - len(idx_original)
            delete_idx = [x for x in range(len(im_patches)) if x >= len(idx_original)]
            num_delete = len(delete_idx) - num_remain
            delete_idx_rand = sorted(np.random.permutation(len(delete_idx))[0:num_delete])
            new_delete_idx = [delete_idx[x] for x in delete_idx_rand]
            remain_idx = [x for x in range(len(im_patches)) if x not in new_delete_idx]
            im_patches = im_patches[remain_idx]
            new_transforms = []
            for x in remain_idx:
                new_transforms.append(self.transforms[x])
            self.transforms = new_transforms
            # for i in reversed(delete_idx):

        self.transforms_more = self.transforms.copy()
        self.transforms = self.transforms_ori.copy()
        self.idx_original = idx_original

        # Extract initial backbone features
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)




        return init_backbone_feat


    def init_target_boxes(self, transforms):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()
        for T in transforms:
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])

        # params.memory_weight_ratio
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        if self.params.get('memory_weight_ratio', 1.0) != 1.0:
            sw[self.idx_original] *= self.params.get('memory_weight_ratio', 1.0)
            sw /= sum(sw)

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_memory_more(self, sample_x: TensorList, target_box, learning_rate=None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights_more(self.sample_weights, self.previous_replace_ind,
                                                 self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind + 1, ...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0], :] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind


    def update_sample_weights_more(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None) * self.params.get('track_net_more_learn_save_weight', 1)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""

        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])



    def init_iou_net(self, backbone_feat, transforms):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()


        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            transforms.extend(transforms[:1]*num)

        if self.params.iounet_augmentation:
            for T in transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([transforms[0].shift[1], transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)

        # Remove other augmentations such as rotation
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation])


    def init_classifier(self, init_backbone_feat, transforms):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']
            transforms.extend(transforms[:1]*num)
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))
        ksz = self.net.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes(transforms)

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory

        if self.cnt_track >= self.params.get('no_save_early_redetection', 0):  # early sample (redetected) is not saved
            if hard_negative_flag or self.frame_num % self.params.get('save_sample_interval', 1) == 0:
                self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # print(self.num_stored_samples[0])
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)


    def update_classifier_more(self, train_x, target_box, learning_rate=None, scores=None, train_x_more=None, target_box_more=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory

        if self.cnt_track >= self.params.get('no_save_early_redetection', 0):  # early sample (redetected) is not saved
            if hard_negative_flag or self.frame_num % self.params.get('save_sample_interval', 1) == 0:
                self.update_memory(TensorList([train_x]), target_box, learning_rate)
                if not self.params.get('track_net_more_learn_not_save', True):
                    for i in range(len(target_box_more)):
                        train_x_more_local = torch.unsqueeze(train_x_more[i], dim=0)
                        self.update_memory_more(TensorList([train_x_more_local]), target_box_more[i], learning_rate)


        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # print(self.num_stored_samples[0])
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # if self.params.get('track_net_more_learn_not_save', True):
            if sample_weights.is_cuda:
                np_weight = sample_weights.cpu().numpy()
            else:
                np_weight = sample_weights.numpy()
            np_weight = list(np_weight)
            for i in range(len(train_x_more)):
                np_weight.append(min(np_weight))
            tensor_weight = torch.Tensor(np_weight)
            if sample_weights.is_cuda:
                sample_weights = tensor_weight.cuda()

            samples = torch.cat((samples, train_x_more), dim = 0)
            if target_boxes.is_cuda:
                target_boxes = torch.cat((target_boxes, target_box_more.cuda()), dim = 0)
            else:
                target_boxes = torch.cat((target_boxes, target_box_more), dim = 0)

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])

            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression."""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')