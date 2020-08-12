
from analysis.plot_results import plot_results, print_results, print_per_sequence_results
from evaluation import Tracker, get_dataset, trackerlist

trackers = []
# trackers.extend(trackerlist('dimp', 'dimp18', None, 'dimp18'))
# trackers.extend(trackerlist('dimp', 'dimp18_vot', None, 'dimp18_vot'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, 'dimp50'))
# trackers.extend(trackerlist('dimp', 'dimp50_vot', None, 'dimp50_vot'))
# trackers.extend(trackerlist('dimp', 'dimp50_vot19', None, 'dimp50_vot19'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, 'prdimp18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, 'prdimp50'))
trackers.extend(trackerlist('dimp', 'super_dimp', None, 'super_dimp'))


# dataset = get_dataset('vot')
# print_results(trackers, dataset, 'VOT', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))



# filter_criteria = {'mode': 'ao_max', 'threshold': 100}
# filter_criteria = {'mode': 'ao_min', 'threshold': 10.0}
# filter_criteria = {'mode': 'delta_ao', 'threshold': 40.0}
filter_criteria = None
dataset = get_dataset('vot')
print_per_sequence_results(trackers, dataset, 'VOT', merge_results=True, filter_criteria=filter_criteria, force_evaluation=False)