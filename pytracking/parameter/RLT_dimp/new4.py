from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    params.image_sample_size = 22*16
    params.search_area_scale = 6
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.5

    # Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20

    # Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1

    # Detection parameters
    params.window_output = False

    # Init augmentation parameters
    params.use_augmentation = True
    params.augmentation = {'fliplr': True,
                           'rotate': [10, -10, 45, -45],
                           'blur': [(3,1), (1, 3), (2, 2)],
                           'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                           'dropout': (2, 0.2)}

    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3

    # Advanced localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.45
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True

    # IoUnet parameters
    params.box_refinement_space = 'relative'
    params.iounet_augmentation = False      # Use the augmented samples to compute the modulation vector
    params.iounet_k = 3                     # Top-k average to estimate final box
    params.num_init_random_boxes = 9        # Num extra random boxes in addition to the classifier prediction
    params.box_jitter_pos = 0.1             # How much to jitter the translation for random boxes
    params.box_jitter_sz = 0.5              # How much to jitter the scale for random boxes
    params.maximal_aspect_ratio = 6         # Limit on the aspect ratio
    params.box_refinement_iter = 10          # Number of iterations for refining the boxes
    params.box_refinement_step_length = 2.5e-3 # 1   # Gradient step length in the bounding box refinement
    params.box_refinement_step_decay = 1    # Multiplicative step length decay (1 means no decay)

    params.net = NetWithBackbone(net_path='super_dimp.pth.tar',
                                 use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'
    params.perform_hn_without_windowing = False
    params.save_sample_interval = 2 # save memory interval


    # [------ new parameters -------]
    # parameters for re-detection
    params.re_detection = True # [default:True] Re-detection
    params.flag_confidence = 6 # [default:6] different methods about confidence score
    params.cnt_global = 10000 # [default:10000->not use] global search for each count
    params.global_search_memory_limit = 200 # global search memory limit
    params.cnt_random = 5 # [default:5] random search for each count
    params.additional_candidate_random = 3 # [default:3] the number of candidates to search (when additional_candidate_adaptive is False)
    params.additional_candidate_adaptive = True # [default:True] adaptive number of candidates to search
    params.additional_candidate_adaptive_ratio = 0.1 # [default:0.1] ratio for adaptive number
    params.additional_candidate_adaptive_min = 1 # [default:1] minimum number for searching
    params.additional_candidate_adaptive_max = 10 # [default:10] maximum number for searching
    params.redetection_score_penalty = True # [default:True] score penalty for re-detection
    params.redetection_score_penalty_alpha = 0.75 # [default:0.75] score penalty parameter, about distance (big value -> more penalty)
    params.redetection_score_penalty_beta = 0.25 # [default:0.25] score penalty parameter, about time (small value -> slow detect)
    params.redetection_basic_penalty = 0.75 # [default:0.75] score penalty parameter, total score reduction, (0.75 -> 25% reduce score)
    params.redetection_now = True # [default:True] re-detection immediately after tracking failure
    params.no_update_early_redetection = 1 # no update period after re-detection success
    params.no_save_early_redetection = 0 # no save period after re-detection success
    params.redetection_global_search_flag = 1 # search position flag (0: 1/4 overlap, 1: half overlap, 2: none overlap)

    # parameters for more discriminative learning
    params.init_net_more_learn = True # more discriminative learning (init)
    params.init_more_learn_expand_searching_size = True # expanding searching size for more init samples
    params.init_more_learn_flag = 1 # search position flag (0: 1/4 overlap, 1: half overlap, 2: none overlap)
    params.init_more_learn_memory_limit = 40 # 100 -> 4Gb (smaller than sample_memory_size - 2), default:40
    params.init_more_learn_no_transform = True # whether transformation
    params.init_more_sample_limit = 10 # searching number limit
    params.init_iou_more_learn = True # same as init_net_more_learn, iounet_augmentation = True -> can change

    params.track_net_more_learn = True # more discriminative learning (track)
    params.track_net_more_learn_search_flag = 2 # search position flag (0: 1/4 overlap, 1: half overlap, 2: none overlap)
    params.track_net_more_learn_cnt = 5 # more learning period
    params.track_net_more_learn_score = 0.80 # score condition for more learning
    params.track_net_more_learn_not_save = True # if True: not save and learn each time (not depend on track_net_more_learn_cnt)
    params.track_net_more_learn_save_weight = 0.5 # reduce memory weight (due to not real data)
    params.train_more_sample_limit = 10 # searching number limit
    params.additional_train_candidate = 2 # number of additional samples
    params.memory_weight_ratio = 2 # weight for initial feature (higher -> more important)
    params.init_blending = 0.0 # image blending with target and background (0: not blending, 0.1: 10% background, 90% target)

    # parameters for random erasing
    params.erasing_mode = True # Random erasing (RE) mode when tracking
    params.use_original_pos = False # Random erasing flag
    params.erasing_cnt = 5  # Random erasing period
    params.lower_scale = 0.02 # RE parameters
    params.upper_scale = 0.05 # RE parameters
    params.lower_ratio = 0.7 # RE parameters
    params.upper_ratio = 1.3 # RE parameters
    params.num_erasing = 10 # the number of random erasing images


    return params
