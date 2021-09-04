import csv
import os

import numpy as np

from constants import detections_path
from models.utils import linear_assignment, iou_batch


class MultiObjectTracker:
    """
    Implementation of the tracker object which may be part of the aggregator_name used.

    Available trackers: SORT, IOU_LA

    Parameters
    -----------
    tracker_name: str
        name of the tracker used. Should be one of 'SORT', 'IOU_LA'
    dataset: Dataset
        name of the video sequence for the tracker
    """
    tracker_classes = ['SORT', 'IOU_LA', 'IDEAL']

    def __init__(self, tracker_name, dataset, iou_threshold=0.3):
        self.name = tracker_name
        self.dataset = dataset
        self.iou_threshold = iou_threshold
        self.saved_tracks = {}
        self.frame_id = -1

    def reset_video(self):
        pass

    def update(self, detections, frame_id, video):
        raise NotImplementedError()

    def associate(self, detections, trackers):
        """
        Assigns results to tracked object (both represented as bounding boxes)

        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = iou_batch(detections, trackers)

        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > self.iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def track_video_sequences(self, detector, cache=False, overwrite_cache=False, verbose=False):
        """
        Runs the tracker_name on self.results.

        Parameters
        ----------
        detector: Detector
        cache: bool
        overwrite_cache: bool
        verbose: bool
        """
        for idx, video_name in enumerate(self.dataset.get_video_names()):
            video = self.dataset.videos[video_name]
            if verbose:
                print('\r{}/{}'.format(1 + idx, len(self.dataset.get_video_names())), video.name, end='')
            lines = []
            self.saved_tracks[video.name] = {}
            self.reset_video()
            self.frame_id = -1
            for frame_number in range(len(video)):
                objects = self.update(detector[video.name][frame_number].bboxes, frame_number, video)
                for obj in objects:
                    lines.append(obj.to_list())
                self.saved_tracks[video.name][frame_number] = self.saved_tracks[video.name].get(frame_number,
                                                                                                []) + objects
            if cache:
                tr_path = '{}/{}_{}/'.format(detections_path, self.name, detector.name)
                os.makedirs(tr_path, exist_ok=True)
                if overwrite_cache or not os.path.exists(tr_path + video.name + '.csv'):
                    with open(tr_path + video.name + '.csv', 'w+') as f:
                        writer = csv.writer(f)
                        writer.writerows(lines)


def track_videos(detector, tracker_name, dataset, classes=None, tracker_age=10, verbose=True, min_hits=1,
                 iou_threshold=0.3):
    from models.trackers.SORT import Sort
    from models.trackers.ideal_tracker import IdealTracker
    from models.trackers.iou_linear_assignment import IouLinearAssignment
    from models.trackers.single_object_tracker import OpenCVTracker, KFTracker

    if tracker_name == Sort.name:
        tracker = Sort(dataset, tracker_age, min_hits)
    elif tracker_name == IouLinearAssignment.name:
        tracker = IouLinearAssignment(dataset, iou_threshold)
    elif tracker_name == IdealTracker.name:
        tracker = IdealTracker(dataset)
    elif tracker_name == KFTracker.name:
        tracker = KFTracker(dataset, classes[0])
    elif tracker_name in OpenCVTracker.OPENCV_OBJECT_TRACKERS.keys():
        tracker = OpenCVTracker(dataset, classes[0], tracker_name)
    else:
        raise NotImplementedError("Given tracker {} is not implemented.".format(tracker_name))

    if verbose:
        print(detector.name)
    tracker.track_video_sequences(detector, cache=True, verbose=verbose)
