import csv

from constants import MIN_OVERLAP, vid_image_path, vid_annotations_path, set_formation_app, \
    set_formation_map, set_formation_app_loc, set_formation_loc, set_formation_time, matching_ln, matching_kf, \
    matching_default, videos_cache_path, gt_cache_path
from video_elements.bndbox import gamma_iou, BoundingBox
from models.utils import get_video_lengths, get_video_shapes


class Frame:
    def __init__(self, number, video_name, bboxes):
        """

        Parameters
        ----------
        number : int
        video_name : str
        bboxes : list(BoundingBox)
        """
        self.number = number
        self.bboxes = bboxes
        self.video_name = video_name
        self.image = None

    def get_image(self):
        if self.image is None:
            import cv2
            self.image = cv2.imread('{}/{}/{}.JPEG'.format(vid_image_path, self.video_name, str(self.number).zfill(6)))
        return self.image


class GTSet:
    def __init__(self, first_frame, set_criteria=set_formation_loc, gamma=10):
        """

        Parameters
        ----------
        first_frame : Frame
        """
        self.object_id = first_frame.bboxes[0].object_id
        self.class_name = first_frame.bboxes[0].class_name
        self.video_name = first_frame.video_name
        self.set_criteria = set_criteria
        self.gamma = gamma
        self.first = first_frame
        self.last = self.first
        self.frames = [self.first]
        self.num_frames = 1
        self.key_frame = None
        self.appearance_values = {}

    def __iter__(self):
        return self.frames.__iter__()

    def add_frame(self, frame):
        if self.can_be_added(frame):
            self.last = frame
            self.frames.append(self.last)
            self.num_frames += 1
            return True
        return False

    def can_be_added(self, frame):
        frame_id, bbox = frame.number, frame.bboxes[0].box
        if frame_id != self.last.number + 1 or self.set_criteria == set_formation_map:
            return False

        elif self.set_criteria == set_formation_loc:
            return gamma_iou(bbox, self.first.bboxes[0].box, self.gamma) > MIN_OVERLAP

        elif self.set_criteria == set_formation_time:
            return frame_id < self.first.number + 40

        elif self.set_criteria == set_formation_app:
            import cv2
            import numpy as np
            if frame_id not in self.appearance_values.keys():
                img_path = '{}/{}/{}.JPEG'.format(vid_image_path, self.video_name, str(frame_id).zfill(6))
                img = cv2.imread(img_path)
                xmin, ymin, xmax, ymax = bbox
                hist = cv2.calcHist([img[ymin:ymax, xmin:xmax]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                self.appearance_values[frame_id] = hist.flatten()
            if self.first not in self.appearance_values.keys():
                img_path = '{}/{}/{}.JPEG'.format(vid_image_path, self.video_name, str(self.first).zfill(6))
                img = cv2.imread(img_path)
                xmin, ymin, xmax, ymax = self.first.bboxes[0].box
                hist = cv2.calcHist([img[ymin:ymax, xmin:xmax]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                self.appearance_values[self.first] = hist.flatten()
            dist = np.linalg.norm(self.appearance_values[self.first] - self.appearance_values[frame_id])
            return dist < 27000

        elif self.set_criteria == set_formation_app_loc:
            if gamma_iou(bbox, self.first.bboxes[0].box, self.gamma) <= MIN_OVERLAP:
                return False
            import numpy as np
            import cv2
            if frame_id not in self.appearance_values.keys():
                img_path = '{}/{}/{}.JPEG'.format(vid_image_path, self.video_name, str(self.first).zfill(6))
                img = cv2.imread(img_path)
                xmin, ymin, xmax, ymax = bbox
                hist = cv2.calcHist([img[ymin:ymax, xmin:xmax]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                self.appearance_values[frame_id] = hist.flatten()
            if self.first not in self.appearance_values.keys():
                img_path = '{}/{}/{}.JPEG'.format(vid_image_path, self.video_name, str(self.first).zfill(6))
                img = cv2.imread(img_path)
                xmin, ymin, xmax, ymax = self.first.bboxes[0].box
                hist = cv2.calcHist([img[ymin:ymax, xmin:xmax]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                self.appearance_values[self.first] = hist.flatten()
            dist = np.linalg.norm(self.appearance_values[self.first] - self.appearance_values[frame_id])
            return dist < 27000

    def __len__(self):
        self.num_frames = len(self.frames)
        return self.num_frames

    def get(self, frame_id):
        assert self.last.number >= frame_id >= self.first.number
        return self.frames[frame_id - self.first.number]

    def match(self, bbox, iou_thresh=None):
        annotation = self.get(bbox.frame_id).bboxes[0]
        iou_thresh = MIN_OVERLAP if iou_thresh is None else iou_thresh
        return gamma_iou(annotation.box, bbox.box) > iou_thresh

    def match_kf(self, bbox, iou_thresh=None):
        annotation = self.get(bbox.frame_id).bboxes[0]
        iou_thresh = MIN_OVERLAP if iou_thresh is None else iou_thresh
        is_kf = bbox.frame_id % 10 == 0
        is_tp = is_kf and gamma_iou(annotation.box, bbox.box) > iou_thresh
        is_fp = is_kf and not is_tp
        return is_tp, is_fp, self.first.number


class GTObject:
    def __init__(self, video_name, obj_id, class_name, first_frame, first_box, gamma, set_criteria, form_sets):
        self.num_frames = 0
        self.video_name = video_name
        self.id = obj_id
        self.class_name = class_name
        self.set_criteria = set_criteria
        self.gamma = gamma
        self.form_sets = form_sets
        self.sets = []
        self.frames = []
        self.add(first_frame, first_box, class_name)

    def __iter__(self):
        return self.frames.__iter__()

    def sets(self):
        return self.sets

    def add(self, frame_id, bbox, cls_id):
        assert self.class_name == cls_id

        self.num_frames += 1
        cur_frame = Frame(frame_id, self.video_name,
                          [BoundingBox(self.video_name, frame_id, cls_id, bbox, object_id=self.id)])
        self.frames.append(cur_frame)

        if self.form_sets:
            for trail in self.sets:
                if trail.add_frame(cur_frame):
                    return
            self.sets.append(GTSet(cur_frame, self.set_criteria, self.gamma))

    def get_set_with_frame(self, frame_id):
        for gt_set in self.sets:
            if gt_set.first.number <= frame_id <= gt_set.last.number:
                return gt_set
        raise ValueError(
            "frame number {} not present in any of the frames for the given object {}.".format(frame_id, self.id))

    def match_obj(self, bbox, iou_thresh=None, match_criteria=matching_default):
        cls_id, frame_id = bbox.class_name, bbox.frame_id
        if self.class_name != cls_id:
            return False, True, -1

        for gt_set in self.sets:
            if frame_id < gt_set.first.number or frame_id > gt_set.last.number:
                continue

            if match_criteria == matching_default:
                det = gt_set.match(bbox, iou_thresh=iou_thresh)
                return det, not det, gt_set.first.number

            elif match_criteria == matching_kf:
                return gt_set.match_kf(bbox, iou_thresh=iou_thresh)
            elif match_criteria == matching_ln:
                if gt_set.match(bbox, iou_thresh=iou_thresh):
                    return 1 / len(gt_set), False, gt_set.first.number
            else:
                raise NotImplementedError("matching criteria must be one of {}, {} {}.")

        return False, True, -1


class Video:
    def __init__(self, video_name, dataset_name, label_map, class_names=None, set_criteria='loc', gamma=10,
                 form_sets=False):
        self.name = video_name
        self.dataset_name = dataset_name
        self.label_map = label_map
        self.num_frames = None
        len(self)
        self.shape = None
        self.get_width_height()
        self.frames = []
        self.per_frame_boxes = {}
        self.set_criteria = set_criteria
        self.gamma = gamma
        self.gt_objects = None
        self.class_names = None
        self.load_annotations(form_sets, class_names)
        self.get_class_names(class_names)

    def __len__(self):
        if self.num_frames is None:
            video_lengths = get_video_lengths()
            try:
                for line in video_lengths:
                    if line[0] == self.name:
                        self.num_frames = int(line[1])
            except IndexError:
                raise ValueError('Incorrect format of video_lengths.csv. '
                                 'Required format: video_name,video_length')
            if self.num_frames is None:
                raise ValueError('No entry found for video length of {}.'.format(self.name))
        return self.num_frames

    def __iter__(self):
        return self.frames.__iter__()

    def __getitem__(self, item):
        return self.frames.__getitem__(item)

    def parse_annotations(self, form_sets=True, cache=False, overwrite_cache=False):
        with open('{}/{}.csv'.format(vid_annotations_path, self.name)) as f:
            self.gt_objects = {}
            frame_wise_boxes = {}
            lines = sorted(list(csv.reader(f)), key=lambda x: int(x[0]))
            for idx, line in enumerate(lines):
                if len(line) < 7:
                    continue

                cls_id = line[2]
                frame_id = int(line[0])
                obj_id = line[1]
                bbox = [int(i) for i in line[3:7]]
                self.per_frame_boxes[frame_id] = 1 + self.per_frame_boxes.get(frame_id, 0)
                frame_wise_boxes[frame_id] = frame_wise_boxes.get(frame_id, []) + [
                    BoundingBox(self.name, frame_id, cls_id, bbox, object_id=obj_id)]

                if obj_id not in self.gt_objects.keys():
                    self.gt_objects[obj_id] = GTObject(self.name, obj_id, cls_id, frame_id, bbox, self.gamma,
                                                       self.set_criteria, form_sets)
                else:
                    self.gt_objects[obj_id].add(frame_id, bbox, cls_id)

        self.frames = [Frame(frame_id, self.name, frame_wise_boxes.get(frame_id, [])) for frame_id in
                       range(self.num_frames)]
        if cache:
            import os
            import pickle
            frames_cache = '{}/{}/frames/'.format(videos_cache_path, self.dataset_name)
            os.makedirs(frames_cache, exist_ok=True)
            if overwrite_cache or not os.path.exists('{}/{}.pkl'.format(frames_cache, self.name)):
                with open('{}/{}.pkl'.format(frames_cache, self.name), 'wb') as f2:
                    pickle.dump(self.frames, f2)

            objects_path = frames_cache.replace('frames', 'objects')
            if form_sets:
                objects_path = '{}/{}/{}_g{}/'.format(gt_cache_path, self.dataset_name, self.set_criteria, self.gamma)
            os.makedirs(objects_path, exist_ok=True)
            if overwrite_cache or not os.path.exists('{}/{}.pkl'.format(objects_path, self.name)):
                print('{}/{}.pkl'.format(objects_path, self.name))
                with open('{}/{}.pkl'.format(objects_path, self.name), 'wb') as f3:
                    pickle.dump(self.gt_objects, f3)

    def load_annotations(self, form_sets, classes=None):
        import os
        frames_cache = '{}/{}/frames/{}.pkl'.format(videos_cache_path, self.dataset_name, self.name)
        if not os.path.exists(frames_cache):
            print("Cached video files not found. Caching now...")
            self.parse_annotations(form_sets, cache=True)
        else:
            import pickle
            with open(frames_cache, 'rb') as f:
                self.frames = pickle.load(f)
            if not form_sets:
                objects_path = frames_cache.replace('frames', 'objects')
                if not os.path.exists(objects_path):
                    self.parse_annotations(form_sets, True)
                with open(objects_path, 'rb') as f2:
                    self.gt_objects = pickle.load(f2)
            else:
                self.load_sets()
        if classes:
            filtered_objects = {}
            for obj_id, obj in self.gt_objects.items():
                if obj.class_name in classes:
                    filtered_objects[obj_id] = obj
            self.gt_objects = filtered_objects
            for frame_id, _ in enumerate(self.frames):
                filtered_boxes = [bbox for bbox in self.frames[frame_id].bboxes if bbox.class_name in classes]
                self.frames[frame_id].bboxes = filtered_boxes

    def load_sets(self):
        import os
        objects_cache_path = '{}/{}/{}_g{}/'.format(gt_cache_path, self.dataset_name, self.set_criteria, self.gamma)
        if os.path.exists('{}/{}.pkl'.format(objects_cache_path, self.name)):
            import pickle
            with open('{}/{}.pkl'.format(objects_cache_path, self.name), 'rb') as f3:
                self.gt_objects = pickle.load(f3)
        else:
            self.parse_annotations(True, cache=True)

    def objects(self):
        if self.gt_objects is None:
            self.parse_annotations(True)
        return self.gt_objects.values()

    def get_width_height(self):
        if self.shape is None:
            for line in get_video_shapes():
                if line[0] == self.name:
                    try:
                        self.shape = (int(line[1]), int(line[2]))
                    except IndexError:
                        raise ValueError('Incorrect format Required format: Video_name,width,height')
            if self.shape is None:
                raise ValueError('No entry found for {} in video shapes.'.format(self.name))
        return self.shape

    def get_class_names(self, class_names=None):
        if self.class_names is None:
            self.class_names = []
            for obj in self.gt_objects.values():
                if class_names and obj.class_name not in class_names:
                    continue
                if obj.class_name not in self.class_names:
                    self.class_names.append(obj.class_name)
        return self.class_names

    def get_num_frames_per_class(self):
        res = {}
        for cls in self.class_names:
            res[cls] = 0
        for obj_id, obj in self.gt_objects.items():
            if obj.class_name in self.class_names:
                res[obj.class_name] += obj.num_frames
        return res

    def get_num_sets_per_class(self):
        res = {}
        for cls in self.class_names:
            res[cls] = 0
        for obj_id, obj in self.gt_objects.items():
            if obj.class_name in self.class_names:
                res[obj.class_name] += len(obj.sets)
        return res

    def match_frames(self, bbox, matched_object_ids=None, iou_thresh=None):
        assert bbox.frame_id < self.num_frames, "Frame number {} not present in video {}.".format(bbox.frame_id,
                                                                                                  self.name)
        if matched_object_ids is None:
            matched_object_ids = []
        for ann in self.frames[bbox.frame_id].bboxes:
            if ann.object_id not in matched_object_ids and bbox.class_name == ann.class_name:
                iou_thresh = iou_thresh if iou_thresh else MIN_OVERLAP
                if gamma_iou(ann.box, bbox.box) > iou_thresh:
                    return True, ann.object_id
        return False, -1

    def match_objects(self, bbox, matched_object_ids=None, iou_thresh=None, match_criteria=matching_default):
        is_fp = True
        matched_obj, matched_set = -1, -1
        for obj_id, obj in self.gt_objects.items():
            if matched_object_ids and obj_id in matched_object_ids:
                continue

            is_tp_obj, is_fp_obj, matched_set_id = obj.match_obj(bbox, iou_thresh, match_criteria)
            if is_tp_obj:
                return is_tp_obj, False, obj_id, matched_set_id
            if not is_fp_obj:
                is_fp = False
                matched_obj = obj_id
                matched_set = matched_set_id

        return False, is_fp, matched_obj, matched_set
