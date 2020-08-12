from pytracking.evaluation import Tracker, get_dataset, trackerlist


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('dimp', 'dimp18_vot', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset


def vot_lt_2020_all_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('dimp', 'dimp18_vot', range(1))+ \
               trackerlist('dimp', 'dimp50', range(1))+ \
               trackerlist('dimp', 'dimp50_vot', range(1))+ \
               trackerlist('dimp', 'dimp50_vot19', range(1))+ \
               trackerlist('dimp', 'prdimp18', range(1))+ \
               trackerlist('dimp', 'prdimp50', range(1))+ \
               trackerlist('dimp', 'super_dimp', range(1))

    dataset = get_dataset('vot')
    return trackers, dataset

# def vot_lt_2020_local1():
#     trackers = trackerlist('dimp', 'dimp18', range(1)) + \
#                trackerlist('dimp', 'dimp18_vot', range(1))
#
#     dataset = get_dataset('vot')
#     return trackers, dataset
def vot_lt_2020_local2():
    trackers = trackerlist('dimp', 'dimp18', None) + \
               trackerlist('dimp', 'dimp18_vot', None)

    dataset = get_dataset('vot')
    return trackers, dataset
def vot_lt_2020_local3():
    trackers = trackerlist('dimp', 'dimp50', None) + \
               trackerlist('dimp', 'dimp50_vot', None) + \
               trackerlist('dimp', 'dimp50_vot19', None)

    dataset = get_dataset('vot')
    return trackers, dataset
def vot_lt_2020_local4():
    trackers = trackerlist('dimp', 'prdimp18', None)+ \
               trackerlist('dimp', 'prdimp50', None)+ \
               trackerlist('dimp', 'super_dimp', None)

    dataset = get_dataset('vot')
    return trackers, dataset
