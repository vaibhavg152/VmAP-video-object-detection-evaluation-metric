from constants import detector_names
from datasets.dataset import Dataset
from models.detector import Detector


def test_op():
    dataset = Dataset('COCO', pre_load=True, num_videos=10)
    for det in detector_names['COCO']:
        detector = Detector(dataset, det, confidence='max_f1')
        print(detector.conf_thresholds)


def test_detectors():
    for dat in ['VIDT', 'COCO', 'VID']:
        dataset = Dataset(dat, lazy_load=True, num_videos=10, num_objects=1)
        for det in detector_names[dat]:
            Detector(dataset, det, 0.9)


def test_nms():
    for dat in ['COCO', 'VIDT', 'VID']:
        dataset = Dataset(dat, lazy_load=True, num_videos=10, num_objects=1)
        for det in detector_names[dat]:
            Detector(dataset, det, 0.7, t_nms_window=3)


def test_trackers():
    for dat in ['COCO', 'VIDT', 'VID']:
        dataset = Dataset(dat, lazy_load=True, num_videos=10, num_objects=1)
        for tracker_name in ['IOU_LA', 'IDEAL', 'SORT', 'KF']:
            print(tracker_name)
            for det in detector_names[dat][:3]:
                Detector(dataset, det, 'max_f1', tracker_name=tracker_name)
