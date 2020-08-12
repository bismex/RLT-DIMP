import os
import sys
import argparse

from collections import namedtuple
import importlib
import os

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Tracker


def run_vot2020(tracker_name, tracker_param, run_id=None, debug=0, visdom_info=None):
    tracker = Tracker(tracker_name, tracker_param, run_id)
    tracker.run_vot2020(debug, visdom_info)

def run_vot2020_LT(tracker_name, tracker_param, run_id=None, gpu_id = 0, debug=0, visdom_info=None):

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tracker = Tracker(tracker_name, tracker_param, run_id)


    name = 'vot'
    DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])
    pt = "pytracking.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter
    dataset_dict = dict(vot=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()))
    dset_info = dataset_dict.get(name)
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)
    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)
    dataset_path = dataset.base_path
    print(dataset_path)
    if 'sequences' in dataset_path:
        dataset_path_all = dataset_path.split('/')
        final_path = ''
        for x in dataset_path_all:
            if x != 'sequences':
                final_path = os.path.join(final_path, x)
        final_path = '/' + final_path
    else:
        final_path = dataset_path
    tracker.run_vot2020_LT(debug, visdom_info, final_path)

def run_vot(tracker_name, tracker_param, run_id=None):
    tracker = Tracker(tracker_name, tracker_param, run_id)
    tracker.run_vot()


def main():
    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('--tracker_name', type=str, default='dimp', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default='super_dimp', help='Name of parameter file.')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu_id')
    parser.add_argument('--run_id', type=int, default=None)

    args = parser.parse_args()

    run_vot(args.tracker_name, args.tracker_param, args.run_id, args.gpu_id)


if __name__ == '__main__':
    main()
