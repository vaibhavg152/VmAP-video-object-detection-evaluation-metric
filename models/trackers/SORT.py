"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex B e w ley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np

from models.trackers.KalmanBox import KalmanBoxTracker
from models.trackers.tracker import MultiObjectTracker


class Sort(MultiObjectTracker):
    name = 'SORT'

    def __init__(self, dataset, max_age, min_hits, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        super().__init__(self.name, dataset, iou_threshold)
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []
        KalmanBoxTracker.count = 0

    def reset_video(self):
        self.trackers = []
        KalmanBoxTracker.count = 0
        self.frame_id = -1

    def update(self, detections, frame_id, video):
        """
        Params:
          detections - a numpy array of results in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty results.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of results provided.
        """
        self.frame_id += 1
        if self.frame_id != frame_id:
            print("Warning: frame skipped for tracking.")
            self.frame_id = frame_id

        # get predicted locations from existing trackers.
        tracklets = np.zeros((len(self.trackers), 4))  # changed to 6 for including class IDs and confidence
        to_del = []
        ret = []
        for t, trk in enumerate(tracklets):
            trk[:] = self.trackers[t].predict().box
            if np.any(np.isnan(trk)):
                to_del.append(t)
        tracklets = np.ma.compress_rows(np.ma.masked_invalid(tracklets))
        for t in reversed(to_del):
            self.trackers.pop(t)
        if not len(detections):
            det_boxes = np.empty(shape=(0, 4))
        else:
            det_boxes = np.array([np.array(det.box) for det in detections])
        matched, unmatched_detections, _ = self.associate(det_boxes, tracklets)
        # update matched trackers with assigned results
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # create and initialise new trackers for unmatched results
        for i in unmatched_detections:
            trk = KalmanBoxTracker(detections[i])
            self.trackers.append(trk)
        self.trackers.sort(key=lambda x: x.get_state().confidence)
        i = len(self.trackers)

        for idx, trk in enumerate(reversed(self.trackers)):
            det = trk.get_state()
            if trk.time_since_update > 1 or trk.hit_streak >= self.min_hits or self.frame_id <= self.min_hits:
                ret.append(det)  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklets
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        return ret
