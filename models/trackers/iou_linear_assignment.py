import numpy as np

from models.trackers.tracker import MultiObjectTracker


class IouLinearAssignment(MultiObjectTracker):
    name = 'IOU_LA'

    def __init__(self, dataset, iou_threshold):
        super().__init__(self.name, dataset, iou_threshold)
        self.tracked_boxes = None
        self.num_objects = 0

    def reset_video(self):
        self.tracked_boxes = None
        self.num_objects = 0
        self.frame_id = -1

    def update(self, detections, frame_id, video):
        self.frame_id += 1
        if self.frame_id != frame_id:
            print("Warning: frame skipped for tracking.", frame_id, self.frame_id)
            self.frame_id = frame_id
        if self.tracked_boxes is None or not self.tracked_boxes:
            self.tracked_boxes = []
            for box in detections:
                box.object_id = self.num_objects
                self.tracked_boxes.append(box)
                self.num_objects += 1
            return self.tracked_boxes

        tracked_boxes = np.array([np.array(det.box) for det in self.tracked_boxes])
        if not len(detections):
            det_boxes = np.empty(shape=(0, 4))
        else:
            det_boxes = np.array([np.array(det.box) for det in detections])
        matched, unmatched_detections, unmatched_tracks = self.associate(det_boxes, tracked_boxes)
        # update matched trackers with assigned results
        for t, _ in enumerate(self.tracked_boxes):
            if t in unmatched_tracks:
                self.tracked_boxes.pop(t)
            else:
                d = matched[np.where(matched[:, 1] == t)[0], 0]
                detections[d[0]].object_id = self.tracked_boxes[t].object_id
                self.tracked_boxes[t] = detections[d[0]]

        # create and initialise new trackers for unmatched results
        for i in unmatched_detections:
            detections[i].object_id = self.num_objects
            self.tracked_boxes.append(detections[i])
            self.num_objects += 1

        return self.tracked_boxes
