import numpy as np
from filterpy.kalman import KalmanFilter

from video_elements.bndbox import BoundingBox


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a trackers using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # changed
        self.detection = bbox
        # self.box = bbox.box
        self.kf.x[:4] = self.convert_bbox_to_z(self.detection.box)
        # self.class_id = bbox.class_name
        # self.confidence = bbox.confidence

        self.time_since_update = 0
        self.id = self.count
        self.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def convert_bbox_to_z(bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = max(bbox[2] - bbox[0], 1)
        h = max(bbox[3] - bbox[1], 1)
        x = bbox[0] + w / 2.
        y = bbox[1] + h / 2.
        s = w * h  # scale is just area
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def convert_x_to_bbox(x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

    def __str__(self):
        return str(self.get_state().to_list().astype(int))

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        # changed
        self.detection = bbox
        self.kf.update(self.convert_bbox_to_z(self.detection.box))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # changed
        box = BoundingBox(self.detection.video_name, self.detection.frame_id, self.detection.class_name,
                          list(self.convert_x_to_bbox(self.kf.x)[0]), self.detection.confidence, self.id + 1)
        self.history.append(box)

        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        CHANGED
        """
        box = BoundingBox(self.detection.video_name, self.detection.frame_id, self.detection.class_name,
                          list(self.convert_x_to_bbox(self.kf.x)[0]), self.detection.confidence, self.id + 1)
        self.history.append(box)
        return box
