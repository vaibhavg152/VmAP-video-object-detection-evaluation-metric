import glob
from pathlib import Path

import cv2

from video_elements.bndbox import BoundingBox
from constants import vid_image_path
from models.trackers.KalmanBox import KalmanBoxTracker
from models.trackers.tracker import MultiObjectTracker


class KFTracker(MultiObjectTracker):
    name = 'KF'

    def __init__(self, dataset, class_name):
        assert dataset.num_objects == 1, "{} tracker available only for single object videos".format(self.name)
        super().__init__(self.name, dataset)
        self.class_name = class_name
        self.tracker = None

    def reset_video(self):
        self.tracker = None
        self.frame_id = -1

    def update(self, detections, frame_id, video):
        valid_dets = [d for d in detections if d.class_name == self.class_name]
        if len(valid_dets):
            detection = max(valid_dets, key=lambda x: x.confidence)
            if self.tracker is None:
                self.tracker = KalmanBoxTracker(detection)
            self.tracker.update(detection)
            return [self.tracker.predict()]
        return []


class OpenCVTracker(MultiObjectTracker):
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        "goturn": cv2.TrackerGOTURN_create
    }

    def __init__(self, dataset, class_name, tracker_name):
        assert tracker_name in self.OPENCV_OBJECT_TRACKERS.keys()
        assert dataset.num_objects == 1, "{} tracker available only for single object videos".format(tracker_name)
        super().__init__(tracker_name, dataset)
        (major, minor) = cv2.__version__.split(".")[:2]
        if int(major) == 3 and int(minor) < 3:
            self.tracker = cv2.Tracker_create(tracker_name.upper())
        else:
            self.tracker = self.OPENCV_OBJECT_TRACKERS[tracker_name.lower()]()
        self.class_name = class_name
        self.confidence = None
        self.saved_tracks = {}
        self.frames = {}
        for f in sorted(glob.glob(vid_image_path + self.dataset.name + '/*.JPEG'), key=lambda x: int(Path(x).stem)):
            self.frames[int(Path(f).stem)] = f

    def reset_video(self):
        self.frame_id = -1
        self.confidence = None
        self.saved_tracks = {}

    def update(self, detections, frame_id, video):
        if frame_id not in self.frames:
            print("Warning: No image found for frame {} in the video {}. Skipping.".format(frame_id, self.dataset.name))
        frame = cv2.imread(self.frames[frame_id])
        valid_dets = [d for d in detections if d.class_name == self.class_name]
        detection = max(valid_dets, key=lambda det: det.confidence) if len(valid_dets) else None

        if detection:
            bbox = detection.box
            bbox = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            self.tracker = self.OPENCV_OBJECT_TRACKERS[self.name]()
            self.tracker.init(frame, bbox)
            self.confidence = detection.confidence
            return [detection]
        (success, box) = self.tracker.update(frame)
        del frame
        if success:
            (x, y, w, h) = [int(v) for v in box]
            box = [x, y, x + w, y + h]
            return [BoundingBox(self.dataset.name, frame_id, self.class_name, box, self.confidence)]
        return []
