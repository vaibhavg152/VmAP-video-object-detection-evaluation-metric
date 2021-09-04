import csv
import json
import os
from argparse import ArgumentParser

from constants import custom_datasets_path, experiments_results_path, detector_names
from datasets.dataset import Dataset
from metrics.mAP import MAP
from metrics.vmap import VmAP
from models.detector import Detector
from video_elements.video_elements import Frame

list_videos = ['ILSVRC2015_val_00021000', 'ILSVRC2015_val_00021003', 'ILSVRC2015_val_00021006',
               'ILSVRC2015_val_00021007', 'ILSVRC2015_val_00022000', 'ILSVRC2015_val_00025001',
               'ILSVRC2015_val_00080000', 'ILSVRC2015_val_00080001', 'ILSVRC2015_val_00129000',
               'ILSVRC2015_val_00014000']


def write_datasets_abc():
    os.makedirs(custom_datasets_path, exist_ok=True)
    num_frames = 360

    d = {'small': {'ILSVRC2015_val_00022000': list(range(num_frames // 6))},
         'large': {'ILSVRC2015_val_00025001': list(range(num_frames))}}
    with open('{}/Da.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(fp=f, obj=d, indent=4)

    d = {'small': {'ILSVRC2015_val_00022000': list(range(num_frames // 2))},
         'large': {'ILSVRC2015_val_00025001': list(range(num_frames))}}
    with open('{}/Db.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(fp=f, obj=d, indent=4)

    d = {'small': {'ILSVRC2015_val_00022000': list(range(num_frames))},
         'large': {'ILSVRC2015_val_00025001': list(range(num_frames))}}
    with open('{}/Dc.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(fp=f, obj=d, indent=4)


def write_datasets_123():
    os.makedirs(custom_datasets_path, exist_ok=True)
    num_frames = 600

    d = {'small': {'ILSVRC2015_val_00129000': list(range(num_frames))},
         'large': {'ILSVRC2015_val_00080000': list(range(num_frames))}}
    with open('{}/D1.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(d, f, indent=4)

    num_frames = num_frames // 2
    d = {'small': {'ILSVRC2015_val_00129000': list(range(num_frames)),
                   'ILSVRC2015_val_00021006': list(range(num_frames))},
         'large': {'ILSVRC2015_val_00080000': list(range(num_frames)),
                   'ILSVRC2015_val_00021003': list(range(num_frames))}}
    with open('{}/D2.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(d, f, indent=4)

    num_frames = num_frames // 2
    d = {'small': {'ILSVRC2015_val_00129000': list(range(num_frames)),
                   'ILSVRC2015_val_00021006': list(range(num_frames)),
                   'ILSVRC2015_val_00021007': list(range(num_frames)),
                   'ILSVRC2015_val_00014000': list(range(num_frames))},
         'large': {'ILSVRC2015_val_00080000': list(range(num_frames)),
                   'ILSVRC2015_val_00080001': list(range(num_frames)),
                   'ILSVRC2015_val_00021003': list(range(num_frames)),
                   'ILSVRC2015_val_00021000': list(range(num_frames))}}
    with open('{}/D3.json'.format(custom_datasets_path), 'w+') as f:
        json.dump(d, f, indent=4)


def get_vmap_on_frames(dataset, detector, frames_list):
    # pdb.set_trace()
    new_gts = {}
    small_sets, large_sets = 0, 0
    for video in dataset:
        if video.name not in frames_list['small'] and video.name not in frames_list['large']:
            continue
        new_objs = {}
        for obj_id, obj in video.gt_objects.items():
            new_trails = []
            for set_ in obj.sets:
                # pdb.set_trace()
                for fid in set_.frames:
                    if video.name in frames_list['small'] and fid.number in frames_list['small'][video.name]:
                        small_sets += 1
                        new_trails.append(set_)
                        break
                    elif video.name in frames_list['large'] and fid.number in frames_list['large'][video.name]:
                        large_sets += 1
                        new_trails.append(set_)
                        break
            if len(new_trails) > 0:
                obj.sets = new_trails
                new_objs[obj_id] = obj
        if len(new_objs.keys()) > 0:
            dataset.videos[video.name].gt_objects = new_objs
            new_gts[video.name] = dataset.videos[video.name]

    dataset = Dataset(dataset.name, pre_load=False, class_names=dataset.classes, video_names=list(new_gts.keys()))
    dataset.videos = new_gts
    dataset.get_num_sets_per_class()
    dataset.get_num_frames_per_class()
    vmap = VmAP(dataset, detector, None)
    filtered_dets = {}
    for cls_name in detector.classes:
        filtered_dets[cls_name] = []
        for det_idx, detection in enumerate(vmap.predictions[cls_name]):
            video_name, fid = detection.video_name, detection.frame_id
            if (video_name in frames_list['small'] and fid in frames_list['small'][video_name]) or (
                    video_name in frames_list['large'] and fid in frames_list['large'][video_name]):
                filtered_dets[cls_name].append(detection)
    vmap.set_detections(filtered_dets)
    vmap.evaluate_voc(cache=False)

    return small_sets, large_sets, vmap


def get_map_on_frames(dataset, detector, frames_list):
    new_gts = {}
    num_boxes_per_class = {}
    small_sets, large_sets = 0, 0
    for video in dataset:
        if video.name not in frames_list['small'] and video.name not in frames_list['large']:
            continue
        new_frames = [Frame(i, video.name, []) for i in range(len(video.frames))]
        for frame_id, frame in enumerate(video.frames):
            if video.name in frames_list['small'] and frame.number in frames_list['small'][video.name]:
                small_sets += 1
                new_frames[frame_id] = frame
                for box in frame.bboxes:
                    num_boxes_per_class[box.class_name] = 1 + num_boxes_per_class.get(box.class_name, 0)
            elif video.name in frames_list['large'] and frame.number in frames_list['large'][video.name]:
                large_sets += 1
                new_frames[frame_id] = frame
                for box in frame.bboxes:
                    num_boxes_per_class[box.class_name] = 1 + num_boxes_per_class.get(box.class_name, 0)
        if len(new_frames) > 0:
            dataset.videos[video.name].frames = new_frames
            new_gts[video.name] = dataset.videos[video.name]

    dataset = Dataset(dataset.name, pre_load=False, class_names=dataset.classes, video_names=list(new_gts.keys()))
    dataset.videos = new_gts
    dataset.get_num_sets_per_class()
    dataset.num_frames_per_class = num_boxes_per_class
    vmap = MAP(dataset, detector, None)
    vmap.generate_detections()
    filtered_dets = {}
    for cls_name in detector.classes:
        filtered_dets[cls_name] = []
        for det_idx, detection in enumerate(vmap.predictions[cls_name]):
            video_name, fid = detection.video_name, detection.frame_id
            if (video_name in frames_list['small'] and fid in frames_list['small'][video_name]) or (
                    video_name in frames_list['large'] and fid in frames_list['large'][video_name]):
                filtered_dets[cls_name].append(detection)
    vmap.set_detections(filtered_dets)
    vmap.evaluate_voc(cache=False)

    return small_sets, large_sets, vmap


def eval_custom_datasets(detector_name, video_names=None, verbose=True):
    if video_names is None:
        video_names = list_videos
    results = [
        ['Dataset name', '#small objects', '#large objects', VmAP.name, '#small frames', '#large frames', MAP.name]]
    dataset_names = [1, 2, 3, 'a', 'b', 'c']
    for name in dataset_names:
        classes = ['bear'] if name in [1, 2, 3] else ['domestic_cat']
        dataset = Dataset('COCO', video_names=video_names, class_names=classes)
        detector = Detector(dataset, detector_name, confidence=0.05)

        frames_list = json.load(open('{}/D{}.json'.format(custom_datasets_path, name)))
        small_sets, large_sets, vmap = get_vmap_on_frames(dataset, detector, frames_list=frames_list)
        small_frames, large_frames, map_ = get_map_on_frames(dataset, detector, frames_list=frames_list)
        results.append(
            ['D{}'.format(name), small_sets, large_sets, 100 * vmap.score, small_frames, large_frames,
             100 * map_.score])
        if verbose:
            print("Number of sets:\t small", small_sets, "large", large_sets)
            print("Number of frames:\t small", small_frames, "large", large_frames)
        print("D{}:".format(name), "VmAP: {:.2f}".format(100 * vmap.score), "mAP: {:.2f}".format(100 * map_.score))

    results_path = '{}/custom_dataset/'.format(experiments_results_path)
    import os
    os.makedirs(results_path, exist_ok=True)
    with open('{}/{}.csv'.format(results_path, detector_name), 'w+') as f:
        csv.writer(f).writerows(results)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--verbose', default=True)
    return parser.parse_args()


if __name__ == '__main__':
    write_datasets_123()
    write_datasets_abc()
    for det in detector_names['COCO']:
        print(det)
        eval_custom_datasets(det, list_videos)
