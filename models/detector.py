import csv
import os
import time

import numpy as np

from constants import detections_path, synthetic_detections_path, op_cache_path
from video_elements.bndbox import BoundingBox
from video_elements.video_elements import Frame


class Detector:
    def __init__(self, dataset, detector_name, confidence=0, fix_precision_at=None, tracker_name=None, t_nms_window=1,
                 eager_load=True, verbose=True):
        self.dataset = dataset
        self.name = detector_name
        # assert self.name in detector_names[dataset.name], "{} not present in {}".format(self.name,
        #                                                                                 detector_names[dataset.name])

        self.video_names = dataset.get_video_names()
        self.classes = dataset.get_class_names()
        self.t_nms_window = t_nms_window
        self.tracker_name = tracker_name
        self.verbose = verbose

        self.c_t = confidence
        if isinstance(confidence, float) or isinstance(confidence, int):
            self.conf_thresholds = dict([(cls, confidence) for cls in self.classes])
        elif isinstance(confidence, list):
            assert len(confidence) == len(self.classes), "Got {} confidence values for {} classes".format(
                len(confidence), len(self.classes))
            self.conf_thresholds = dict(zip(self.classes, confidence))
        elif isinstance(confidence, dict):
            self.conf_thresholds = confidence
        elif isinstance(confidence, str):
            if confidence == 'max_f1':
                if verbose:
                    print("getting OPmAP")
                self.conf_thresholds = operating_points(self.dataset.name, self.name, fix_precision_at, True,
                                                        self.t_nms_window)
            elif confidence == 'max_vf1':
                if verbose:
                    print("getting OPVmAP")
                self.conf_thresholds = operating_points(self.dataset.name, self.name, fix_precision_at, False,
                                                        self.t_nms_window)
            else:
                assert False, "confidence values can be max_f1 or max_vf1."

        self.detections = None
        if eager_load:
            self.parse_csv()

    def parse_csv(self):
        """
        Reads the results for video_name of detector_class from the text file present in detector.tracked_detections .

        returns:
            dict of lists: dictionary with keys as frame id and values as list of [class_id, confidence, xmin, xmax,
             ymin, ymax]
        """
        detector = self.name if self.tracker_name is None else '{}_{}'.format(self.tracker_name, self.name)
        self.detections = {}
        video_wise_dets = {}
        if self.verbose:
            print("Reading detections....")
        for video_idx, video_name in enumerate(self.video_names):
            file_path = '{}/{}/{}.csv'.format(detections_path, detector, video_name)
            if self.verbose and video_idx % 20 == 0:
                print('\r', video_idx, video_name, end='')
            if not os.path.exists(file_path):
                file_path_s = '{}/{}/{}.csv'.format(synthetic_detections_path, detector, video_name)
                if not os.path.exists(file_path_s):
                    if self.tracker_name:
                        print('Tracking results not present for {} for {}.'.format(video_name, detector),
                              'Running tracker now (with default parameters).')
                        s = time.time()
                        from models.trackers.tracker import track_videos
                        det_temp = Detector(self.dataset, self.name, self.conf_thresholds, tracker_name=None,
                                            t_nms_window=self.t_nms_window)
                        track_videos(det_temp, self.tracker_name, self.dataset, self.classes)
                        print('Finished tracking. Took {} seconds.'.format(time.time()-s))
                    else:
                        raise ValueError('Detections not found at {}'.format(file_path))

            temp_dets = {}
            with open(file_path) as f:
                lines = list(csv.reader(f))
                for line in lines:
                    if len(line) < (7 if self.tracker_name else 6):
                        line = line[0].split(' ')
                    if len(line) > (7 if self.tracker_name else 6):
                        if self.tracker_name:
                            frame_id, cls, confidence, obj_id = int(line[0]), line[2], float(line[3]), int(line[1])
                            bbox = [int(i) for i in line[4:8]]
                        else:
                            frame_id, cls, confidence, obj_id = int(line[0]), line[1], float(line[2]), None
                            bbox = [int(i) for i in line[3:7]]
                        try:
                            cls = self.dataset.class_name_from_id(cls)
                        except KeyError:
                            continue

                        # fix a detection to have min width and height of 1
                        if bbox[2] == bbox[0]:
                            bbox[2] += 1
                        if bbox[3] == bbox[1]:
                            bbox[3] += 1

                        if self.classes is None or cls in self.classes:
                            if confidence >= self.conf_thresholds[cls]:
                                temp_dets[frame_id] = temp_dets.get(frame_id, []) + [
                                    BoundingBox(video_name, frame_id, cls, bbox, confidence, obj_id)]

            self.detections[video_name] = [Frame(frame_id, video_name, temp_dets.get(frame_id, [])) for frame_id in
                                           range(len(self.dataset.videos[video_name]))]
            video_wise_dets[video_name] = sum([len(f.bboxes) for f in self.detections[video_name]])
            #            print(len(self.dataset.videos[video_name]))
            #            print(max(list(temp_dets.keys())), len(self.detections[video_name]))

            if self.t_nms_window > 1:
                if self.verbose:
                    print('Suppressing detections')
                self.suppress()

        if self.verbose:
            print('Finished reading.')
        #        import json
        #        json.dump(video_wise_dets, open('blah.json', 'w+'), indent=4)
        #        assert False
        return self.detections

    def __call__(self, video):
        """get results for a frame."""
        assert video.name in self.detections.keys(), "Detector {} not instantiated on video {}.".format(self.name,
                                                                                                        video.name)
        return self.detections[video.name]

    def __getitem__(self, item):
        return self.detections.__getitem__(item)

    def __iter__(self):
        return self.detections.items()

    def get_next(self, video_name, frame_id):
        """get results for the next detected frame after frame_id"""
        assert video_name in self.detections.keys(), "Detector {} not instantiated on video {}.".format(self.name,
                                                                                                        video_name)
        assert frame_id <= len(self.detections[video_name]), "frame id {} not present in video {}.".format(frame_id,
                                                                                                           video_name)
        while frame_id not in self.detections[video_name]:
            frame_id += 1
        return self.detections[video_name][frame_id]

    def suppress(self):
        from models.nms_utils import non_max_suppression_fast
        fid = 0
        temp_dets = {}
        for vid_name, dets in self.detections.items():
            while fid <= len(dets):
                cur_window = []
                frame_ids = []
                for _ in range(self.t_nms_window):
                    if fid not in dets:
                        fid += 1
                        continue
                    frame_dets = dets[fid]
                    cur_window += frame_dets
                    frame_ids += [fid for _ in range(len(frame_dets))]
                    temp_dets[fid] = []
                    fid += 1
                if not cur_window:
                    continue
                cls_wise_window = {}
                for det in cur_window:
                    cls_wise_window[det.class_name] = cls_wise_window.get(det.class_name, []) + [det]
                for cls in cls_wise_window.keys():
                    confs, boxes = [], []
                    for det in cls_wise_window[cls]:
                        confs.append(det.confidence)
                        boxes.append(det.box)
                    indices = non_max_suppression_fast(np.array(boxes), confs)
                    for idx in indices:
                        temp_dets[frame_ids[idx]].append(
                            BoundingBox(self.video_names, frame_ids[idx], cls, boxes[idx], confs[idx]))
            self.detections[vid_name] = temp_dets


def generate_detections(detector):
    predictions = {}
    for vid_name in detector.video_names:
        predictions[vid_name] = {}
        for idx, frame in detector.detections[vid_name].items():
            for det in frame.bboxes:
                if det.class_name in detector.classes():
                    predictions[vid_name][det.class_name] = predictions[vid_name].get(det.class_name, []).append(det)

        # sorting tracked_detections according to confidence values
        for cls_name in predictions[vid_name].keys():
            predictions[vid_name][cls_name] = sorted(predictions[vid_name][cls_name], key=lambda x: x.confidence,
                                                     reverse=True)

    return predictions


def operating_points(dataset_name, detector_name, fix_precision=None, frame_level=True, t_nms_window=1):
    from metrics.mAP import MAP
    from metrics.vmap import VmAP
    from datasets.dataset import Dataset
    num_object = None
    dataset = Dataset(dataset_name, lazy_load=frame_level, num_objects=num_object)
    detector = Detector(dataset, detector_name, 0, fix_precision, None, t_nms_window=t_nms_window, eager_load=False)
    metric = MAP(dataset, detector, evaluate=None) if frame_level else VmAP(dataset, detector, evaluate=None)
    return metric.get_operating_points(fix_precision=fix_precision)


def cache_operating_points2(detector_name, metric_name, dataset_name, class_names, tracker_name=None,
                            fix_precision=None, overwrite=True, results_path=None, t_nms_window=1):
    import os
    from datasets.dataset import Dataset
    from metrics.mAP import MAP
    from metrics.vmap import VmAP
    from models.trackers.single_object_tracker import OpenCVTracker, KFTracker
    import json
    results = {}
    num_objects = None
    if tracker_name:
        if tracker_name == KFTracker.name or tracker_name in OpenCVTracker.OPENCV_OBJECT_TRACKERS.keys():
            num_objects = 1
    for cls in class_names:
        dataset = Dataset('VID', lazy_load=metric_name == MAP.name, class_names=[cls], num_objects=num_objects)
        detector = Detector(dataset, detector_name, 0, fix_precision, tracker_name, t_nms_window=t_nms_window,
                            eager_load=False)
        metric = MAP(dataset, detector, None) if metric_name == MAP.name else VmAP(dataset, detector, None)

        rec, prec, tp, fp, fn = metric.get_precision_recall(cls)
        # print(cls, rec[-1], prec[-1], tp, fp, fn)
        if rec[-1] == 0 and prec[-1] == 0:
            results[cls] = 1
            # print('detector 195 lol')
        elif fix_precision is None:
            fsc = [rec[idx] * prec[idx] / (rec[idx] + prec[idx]) for idx in range(len(prec))]
            results[cls] = min(metric.predictions[cls][np.nanargmax(fsc)].confidence, 0.99)
        else:
            # print('detector 200 lol')
            for det_idx, precision in enumerate(reversed(prec)):
                if precision > fix_precision:
                    results[cls] = metric.predictions[cls][-1 - det_idx].confidence
                    break
            if cls not in results:
                results[cls] = 1

    if results_path is None:
        results_path = '{}/{}'.format(op_cache_path, dataset_name)
    os.makedirs(results_path, exist_ok=True)

    detector_name = detector_name if tracker_name is None else '{}_{}'.format(tracker_name, detector_name)
    file_path = '{}/{}{}.json'.format(results_path, detector_name, fix_precision)
    if overwrite or not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    return results


if __name__ == '__main__':
    from models.utils import get_class_wise_videos
    from constants import detector_names

    all_classes = list(get_class_wise_videos().keys())
    for det_name in detector_names['VID']:
        print(det_name)
        cache_operating_points2(det_name, 'VmAP', 'VID', all_classes)
        print('--')
        cache_operating_points2(det_name, 'mAP', 'VID', all_classes)
