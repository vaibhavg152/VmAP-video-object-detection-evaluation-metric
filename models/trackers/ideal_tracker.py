import numpy as np
from constants import MIN_OVERLAP
from models.trackers.tracker import MultiObjectTracker


class IdealTracker(MultiObjectTracker):
    name = 'IDEAL'

    def __init__(self, dataset):
        super().__init__(self.name, dataset, MIN_OVERLAP)
        self.matched_objs = []
        self.deleted = []

    def reset_video(self):
        self.matched_objs = []
        self.deleted = []
        self.frame_id = -1

    def update(self, detections, frame_id, video=None):
        """
        Params:
          tracks - a numpy array of tracks in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty tracks.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of tracks provided.
        """

        # get predicted locations from existing trackers.
        if self.frame_id != frame_id:
            print("Warning: frame skipped for tracking.")
            self.frame_id = frame_id
        tracks = []
        for idx, obj in enumerate(self.matched_objs):
            obj_id = obj.object_id
            if obj_id in self.deleted:
                continue
            found = False
            for gt_set in video.gt_objects[obj_id].sets:
                if gt_set.first.number <= self.frame_id <= gt_set.last.number:
                    obj.box = gt_set.frames[self.frame_id]
                    tracks.append(obj)
                    found = True
                    break
            if not found:
                self.deleted.append(obj_id)
        tracks = np.array(tracks).astype(np.float)
        if not len(detections):
            det_boxes = np.empty(shape=(0, 4))
        else:
            det_boxes = np.array([np.array(d.box) for d in detections])
        if not len(tracks):
            track_boxes = np.empty(shape=(0, 4))
        else:
            track_boxes = np.array([np.array(t.box) for t in tracks])
        matched, unmatched_dets, _ = self.associate(det_boxes, track_boxes)
        # update matched trackers with assigned tracks
        for m in matched:
            try:
                for idx in range(len(self.matched_objs)):
                    if self.matched_objs[idx].object_id == m[1]:
                        self.matched_objs[m[1]] = [m[1], detections[m[0]][0], detections[m[0]][1]]
                        self.matched_objs[m[1]].confidence = detections[m[0]].confidence
            except IndexError:
                raise IndexError()

        ret = list(tracks)
        obj_id_fp = len(tracks)
        matched_obj_ids = [t[0] for t in self.matched_objs]
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            # match the new detection with all unmatched ground truth objects.
            is_det_tp = False
            for obj_id, obj in video.gt_objects.items():
                if obj_id in matched_obj_ids:
                    continue
                is_tp, is_fp, _ = obj.match_obj(det, iou_thresh=MIN_OVERLAP)
                if not is_fp:
                    det.object_id = int(obj_id)
                    self.matched_objs.append(det)
                    ret.append(det)
                    is_det_tp = True
                    break
            if not is_det_tp:
                det.object_id = obj_id_fp
                ret.append(det)
                obj_id_fp += 1

        self.frame_id += 1
        return ret
