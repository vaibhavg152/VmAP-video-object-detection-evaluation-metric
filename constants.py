import os

import numpy as np

root_path = os.path.dirname(os.path.realpath(__file__)) + '/'
dataset_names = ['VID', 'VIDT', 'COCO']
coco_classes = ['airplane', 'bear', 'bicycle', 'bird', 'bus', 'car', 'dog', 'domestic_cat', 'elephant', 'horse',
                'motorcycle', 'sheep', 'train', 'zebra']
set_formation_criteria = ['none', 'loc', 'app', 'app_loc', 'time']
set_formation_map = set_formation_criteria[0]
set_formation_loc = set_formation_criteria[1]
set_formation_app = set_formation_criteria[2]
set_formation_app_loc = set_formation_criteria[3]
set_formation_time = set_formation_criteria[4]
matching_default = 'ANY'
matching_ln = 'LN'
matching_kf = 'KF'
detector_names = {
    'VIDT': ["MEGA-BASE", 'CATDET', "DFF", 'FGFA', 'FRCNN', 'MEGA', 'RDN', 'RFCN', 'RFCN-DFF-10', 'RFCN-FGFA-7'],
    'VID': ['DFF', 'MEGA-BASE', 'FGFA', 'RDN', 'MEGA'],
    'COCO': ['CENTRIPETAL', 'CORNERNET', 'DETR', 'FCOS', 'FRCNN_R50', 'HTC', 'RETINANET_R50', 'YOLOV3', 'DFF',
             'FGFA', 'MEGA', 'MEGA-BASE', 'RDN']}

# minimum iou threshold for declaring a match
MIN_OVERLAP_COCO = np.arange(0.5, 0.951, 0.05)
MIN_OVERLAP = 0.3
NMS_THRESHOLD = 0.8

VID_path = '/home/chrystle/VID/'
vid_image_path = VID_path + 'Data/VID/val/'
vid_xml_annotations_path = VID_path + 'Annotations/VID/val/'

detections_path = root_path + 'detector/results/'
synthetic_detections_path = detections_path + 'synthetic/'

cache_path = root_path + '.cache/'
gt_cache_path = cache_path + 'ground_truth_sets/'
videos_cache_path = cache_path + 'videos/'
eval_cache_path = cache_path + 'scores/'
op_cache_path = cache_path + 'operating_point/'
op_scores_cache_path = cache_path + 'operating_point/scores/'
custom_datasets_path = cache_path + 'custom_datasets/'
datasets_cache_path = cache_path + 'dataset_utils/'
vid_annotations_path = '{}/VID/annotations/'.format(datasets_cache_path)
datasets_path = root_path + 'datasets/'

results_path = root_path + 'results/'
experiments_results_path = results_path + 'experiments/'
set_formation_ablation_path = experiments_results_path + 'set_formation_ablation/'
synthetic_detector_results_path = experiments_results_path + 'synthetic_detectors/'
ranking_results_path = experiments_results_path + 'post_tracker_ranking/'
gamma_vis_path = experiments_results_path + 'gamma_boxes/'
set_vis_path = experiments_results_path + 'set_visualizations/'

eval_results_path = results_path + 'evaluation/'
pr_plot_path = eval_results_path + 'pr_plots/'
timeline_plot_path = eval_results_path + 'timeline_plots/'
visualisations_path = results_path + 'viz/'
plotted_detections_path = visualisations_path + 'plotted_detections/'
plotted_videos_path = visualisations_path + 'plotted_videos/'
