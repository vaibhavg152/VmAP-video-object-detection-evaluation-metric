import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from constants import vid_annotations_path, vid_xml_annotations_path, datasets_cache_path, vid_image_path

map_dict = {'VID':
                {'n02691156': 'airplane',
                 'n02419796': 'antelope',
                 'n02131653': 'bear',
                 'n02834778': 'bicycle',
                 'n01503061': 'bird',
                 'n02924116': 'bus',
                 'n02958343': 'car',
                 'n02402425': 'cattle',
                 'n02084071': 'dog',
                 'n02121808': 'domestic_cat',
                 'n02503517': 'elephant',
                 'n02118333': 'fox',
                 'n02510455': 'giant_panda',
                 'n02342885': 'hamster',
                 'n02374451': 'horse',
                 'n02129165': 'lion',
                 'n01674464': 'lizard',
                 'n02484322': 'monkey',
                 'n03790512': 'motorcycle',
                 'n02324045': 'rabbit',
                 'n02509815': 'red_panda',
                 'n02411705': 'sheep',
                 'n01726692': 'snake',
                 'n02355227': 'squirrel',
                 'n02129604': 'tiger',
                 'n04468005': 'train',
                 'n01662784': 'turtle',
                 'n04530566': 'watercraft',
                 'n02062744': 'whale',
                 'n02391049': 'zebra'}
            }


def get_all_video_names(dataset_name):
    list_videos_path = '{}/{}/list_videos.csv'.format(datasets_cache_path, dataset_name)
    if not os.path.exists(list_videos_path):
        os.makedirs(datasets_cache_path + '{}/'.format(dataset_name), exist_ok=True)
        video_names = [[Path(vid_name).stem] for vid_name in glob.glob(vid_image_path + '*')]
        with open(list_videos_path, 'w+') as f:
            wr = csv.writer(f)
            wr.writerows(video_names)
    with open(list_videos_path) as f:
        return [l[0].strip() for l in csv.reader(f)]


def get_video_lengths():
    file_path = '{}/VID/video_lengths.csv'.format(datasets_cache_path)
    if not os.path.exists(file_path):
        os.makedirs(datasets_cache_path + 'VID/', exist_ok=True)
        video_len = [[Path(vid_name).stem, len(glob.glob(vid_name + '/*.xml'))] for vid_name in
                     glob.glob(vid_image_path + '*')]
        with open(file_path, 'w+') as f:
            wr = csv.writer(f)
            wr.writerows(video_len)
    with open(file_path) as f:
        return [l for l in csv.reader(f)]


def get_video_shapes():
    file_path = '{}/VID/video_shapes.csv'.format(datasets_cache_path)
    if not os.path.exists(file_path):
        import xml.etree.ElementTree as et
        os.makedirs(datasets_cache_path + 'VID/', exist_ok=True)
        video_shapes = []
        for video_name in get_all_video_names():
            xml_files = glob.glob(vid_xml_annotations_path + video_name + '/*.xml')
            tree = et.parse(xml_files[0])
            root = tree.getroot()
            w, h = root.find('width'), root.find('height')
            video_shapes.append([video_name, w, h])
        with open(file_path, 'w+') as f:
            wr = csv.writer(f)
            wr.writerows(video_shapes)
    with open(file_path) as f:
        return [l for l in csv.reader(f)]


def get_class_wise_videos():
    file_path = '{}/VID/class_wise_videos.json'.format(datasets_cache_path)
    if not os.path.exists(file_path):
        videos = {}
        for vid in get_all_video_names():
            file_path2 = '{}/VID/annotations/{}.csv'.format(datasets_cache_path, vid)
            if not os.path.exists(file_path2):
                cache_vid_annotations()
            with open(file_path2) as f:
                for line in csv.reader(f):
                    cls_name = line[2]
                    if cls_name not in videos.keys():
                        videos[cls_name] = []
                    if vid not in videos[cls_name]:
                        videos[cls_name].append(vid)

        json.dump(videos, open(file_path, 'w'), indent=4)

    with open(file_path) as f:
        return json.load(f)


def get_video_wise_num_objects():
    file_path = '{}/VID/num_objects.json'.format(datasets_cache_path)
    import json
    if not os.path.exists(file_path):
        num_objs = {}
        for vid_name in get_all_video_names():
            object_ids = []
            annotations_path = '{}/{}.csv'.format(vid_annotations_path, vid_name)
            if not os.path.exists(annotations_path):
                cache_vid_annotations()
            with open(annotations_path) as f:
                for line in csv.reader(f):
                    if line[1] not in object_ids:
                        object_ids.append(line[1])
            n1 = len(object_ids)
            num_objs[n1] = num_objs.get(n1, []) + [vid_name]

        json.dump(obj=num_objs, fp=open(file_path, 'w'), indent=4)
    with open(file_path) as f:
        return json.load(f)


def cache_vid_annotations():
    import xml.etree.ElementTree as et

    os.makedirs(vid_annotations_path, exist_ok=True)

    # REQUIRED FORMAT : frame_id, object_id, class_id, xmin, ymin, width, height
    video_names = get_all_video_names('VID')
    for video_name in video_names:
        xml_video_path = vid_xml_annotations_path + '{}.csv'.format(video_name)
        out_path = vid_annotations_path + '{}.csv'.format(video_name)
        print(xml_video_path)
        print(out_path)
        xml_files = glob.glob(xml_video_path + '/*.xml')
        assert len(xml_files)
        annotations = []
        for xml_file in xml_files:
            # print(xml_file)
            tree = et.parse(xml_file)
            root = tree.getroot()
            filename = root.find('outfilename').text
            frame_id = int(filename)
            objects = root.findall('object')
            for track_obj in objects:
                object_id = track_obj.find('trackid').text
                class_id = map_dict['VID'][track_obj.find('name').text]
                xmin = track_obj.find('bndbox/xmin').text
                ymin = track_obj.find('bndbox/ymin').text
                xmax = track_obj.find('bndbox/xmax').text
                ymax = track_obj.find('bndbox/ymax').text
                annotations.append([frame_id, object_id, class_id, xmin, ymin, xmax, ymax])
        with open(out_path, 'w+') as f:
            wr = csv.writer(f)
            wr.writerows(annotations)


def reorder_keys(frame_dict, sort_key=None):
    """ reorder the frames dictionary in a ascending manner """
    keys_int = sorted(list(frame_dict.keys()), key=sort_key)

    new_dict = {}
    for key in keys_int:
        new_dict[key] = frame_dict[key]
    return new_dict


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bounding boxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    # print(bb_gt.shape, bb_test.shape)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def f1(x, y):
    return 2 * x * y / (x + y) if x + y else 0