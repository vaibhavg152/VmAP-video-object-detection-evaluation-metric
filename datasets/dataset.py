import json
import os

from constants import set_formation_loc, datasets_cache_path, datasets_path, coco_classes
from models.utils import get_all_video_names, get_video_wise_num_objects, get_class_wise_videos
from video_elements.video_elements import Video


class Dataset:
    def __init__(self, dataset_name, pre_load=True, lazy_load=False, class_names=None, video_names=None,
                 num_objects=None, num_videos=None, set_criteria=set_formation_loc, gamma=10):
        self.name = dataset_name
        self.label_maps_dict = None
        self.get_label_map()
        self.class_wise_videos = None
        self.get_class_wise_videos()
        class_names = [class_names] if isinstance(class_names, str) else class_names
        self.classes = self.all_class_names() if not class_names else class_names
        self.video_names = video_names
        self.num_objects = num_objects
        if self.video_names is None:
            self.video_names = self.get_video_names()
        if self.num_objects:
            self.videos_with_n_objects()
        if num_videos:
            self.video_names = self.video_names[:num_videos]
        self.set_criteria = set_criteria
        self.gamma = gamma
        form_sets = pre_load and not lazy_load
        self.videos = {}
        for vid in self.video_names:
            self.videos[vid] = Video(vid, self.name, self.label_maps_dict, class_names=self.classes,
                                     set_criteria=self.set_criteria, gamma=self.gamma, form_sets=form_sets)
        self.get_class_names()
        self.num_frames_per_class = {}
        self.num_sets_per_class = {}
        self.get_num_frames_per_class()
        self.get_num_sets_per_class()

    def get_num_frames_per_class(self):
        for video in self:
            objs = video.get_num_frames_per_class()
            for cls, num_objs in objs.items():
                self.num_frames_per_class[cls] = self.num_frames_per_class.get(cls, 0) + num_objs
        return self.num_frames_per_class

    def get_num_sets_per_class(self):
        for video in self:
            objs = video.get_num_sets_per_class()
            for cls, num_objs in objs.items():
                self.num_sets_per_class[cls] = self.num_sets_per_class.get(cls, 0) + num_objs
        return self.num_frames_per_class

    def __iter__(self):
        return self.videos.values().__iter__()

    def __getitem__(self, item=0):
        return list(self.videos.values())[item]

    @staticmethod
    def filter(dataset_name, classes=None, video_names=None, num_objects=None, num_videos=None,
               set_criteria=set_formation_loc, gamma=10):
        return Dataset(dataset_name, False, True, classes, video_names, num_objects, num_videos, set_criteria, gamma)

    def get_label_map(self):
        if self.label_maps_dict is None:
            label_map_path = '{}/{}/label_map.json'.format(datasets_path, self.name)
            assert os.path.exists(label_map_path), 'No label maps present for dataset: {}'.format(self.name)
            with open(label_map_path) as f:
                self.label_maps_dict = json.load(f)
            label_keys = [int(k) for k in self.label_maps_dict.keys()]
            label_values = list(self.label_maps_dict.values())
            self.label_maps_dict = dict(zip(label_keys, label_values))

        return self.label_maps_dict

    def all_class_names(self):
        return list(self.label_maps_dict.values())

    def get_class_names(self):
        self.classes = []
        for vid in self.videos.values():
            for cls in vid.class_names:
                if cls not in self.classes:
                    self.classes.append(cls)
        return self.classes

    def get_video_names(self):
        if self.classes == self.all_class_names():
            videos = get_all_video_names(self.name)
        else:
            class_wise_videos = []
            for cls in self.classes:
                for vid in self.class_wise_videos[cls]:
                    if vid not in class_wise_videos:
                        class_wise_videos.append(vid)
            if self.video_names:
                videos = [video_name for video_name in self.video_names if
                                video_name in class_wise_videos]
            else:
                videos = class_wise_videos
        self.video_names = videos
        return videos

    def get_class_wise_videos(self):
        if self.class_wise_videos is None:
            self.class_wise_videos = get_class_wise_videos()
        return self.class_wise_videos

    def class_id_from_name(self, key):
        self.get_label_map()

        if isinstance(key, int) or isinstance(key, float):
            return int(key)
        if key not in self.label_maps_dict.keys():
            return int(float(key))
        idx = list(self.label_maps_dict.values()).index(int(float(key)))
        return list(self.label_maps_dict.keys())[idx]

    def class_name_from_id(self, key):
        self.get_label_map()

        if isinstance(key, str):
            try:
                key = int(float(key))
            except ValueError:
                return key
        return self.label_maps_dict[key]

    def videos_with_n_objects(self):
        if self.num_objects is None:
            return self.video_names

        n_object_videos = get_video_wise_num_objects()[str(self.num_objects)]
        if self.video_names:
            self.video_names = [vid for vid in self.video_names if vid in n_object_videos]
        else:
            self.video_names = n_object_videos


class VID(Dataset):
    def __init__(self, pre_load=True, lazy_load=False, class_names=None, video_names=None, num_objects=None,
                 num_videos=None, set_criteria=set_formation_loc, gamma=10):
        super().__init__('VID', pre_load, lazy_load, class_names, video_names, num_objects, num_videos, set_criteria,
                         gamma)

    @staticmethod
    def filter(classes=None, video_names=None, num_objects=None, num_videos=None, set_criteria=set_formation_loc,
               gamma=10):
        return VID(False, True, classes, video_names, num_objects, num_videos, set_criteria, gamma)


class COCO(Dataset):
    def __init__(self, pre_load=True, lazy_load=False, video_names=None, num_objects=None, num_videos=None,
                 set_criteria=set_formation_loc, gamma=10):
        super().__init__('COCO', pre_load, lazy_load, coco_classes, video_names, num_objects, num_videos, set_criteria,
                         gamma)

    @staticmethod
    def filter(video_names=None, num_objects=None, num_videos=None, set_criteria=set_formation_loc,
               gamma=10):
        return COCO(False, True, video_names, num_objects, num_videos, set_criteria, gamma)


class VIDT(Dataset):
    def __init__(self, pre_load=True, lazy_load=False, class_names=None, num_objects=None, num_videos=None,
                 set_criteria=set_formation_loc, gamma=10):
        with open('{}/VIDT/list_videos.csv'.format(datasets_cache_path)) as f:
            vidt_videos = [line.strip().split()[0] for line in f.readlines()]
        super().__init__('VIDT', pre_load, lazy_load, class_names, vidt_videos, num_objects, num_videos, set_criteria,
                         gamma)

    @staticmethod
    def filter(classes=None, num_objects=None, num_videos=None, set_criteria=set_formation_loc,
               gamma=10):
        return VIDT(False, True, classes, num_objects, num_videos, set_criteria, gamma)
