import csv
import os
from collections import defaultdict

import numpy as np

from constants import vid_annotations_path, detections_path
from datasets.dataset import Dataset
from video_elements.bndbox import iou_box


def calc_delay(bboxes_gt, cls_gt, track_ids, bboxes, scores, cls):
    first_frame_pertrack = defaultdict(lambda: 1e9)
    first_detect_pertrack = defaultdict(lambda: 1e9)
    last_frame_pertrack = defaultdict(int)
    class_pertrack = defaultdict(int)
    size_pertrack = defaultdict(list)

    frames = max(max(bboxes_gt.keys()), max(bboxes.keys()))
    # results = [0] * 6
    iou_thresh = 0.5
    for frame in range(frames):
        for c, track_id, bbox in zip(cls_gt[frame], track_ids[frame], bboxes_gt[frame]):
            h, w = (bbox[3] - bbox[1], bbox[2] - bbox[0])
            shorter = min(h, w)
            size_pertrack[track_id].append(shorter)
            first_frame_pertrack[track_id] = min(first_frame_pertrack[track_id], frame)
            last_frame_pertrack[track_id] = max(first_frame_pertrack[track_id], frame)
            if class_pertrack[track_id] == 0:
                class_pertrack[track_id] = c
            else:
                assert class_pertrack[track_id] == c, \
                    "%d vs %d" % (class_pertrack[track_id], c)

        num_gt = len(bboxes_gt[frame])
        for bb, s, c in zip(bboxes[frame], scores[frame], cls[frame]):
            max_overl = 0.0
            track_id = -1
            for id_gt in range(num_gt):

                overl = iou_box(bboxes_gt[frame][id_gt], bb)
                c_gt = cls_gt[frame][id_gt]
                if overl > max_overl and c == c_gt:
                    max_overl = overl
                    track_id = track_ids[frame][id_gt]
            if max_overl > iou_thresh:
                first_detect_pertrack[track_id] = min(first_detect_pertrack[track_id], frame)

    delays = []
    classes = []
    ids = []
    first_frames = []
    sizes = []
    for track_id in first_frame_pertrack:
        size = np.mean(size_pertrack[track_id][:30])
        first_frame = first_frame_pertrack[track_id]
        if track_id in first_detect_pertrack:
            delay = first_detect_pertrack[track_id] - first_frame_pertrack[track_id]
        else:
            delay = last_frame_pertrack[track_id] - first_frame_pertrack[track_id]
        delay = max(delay, 0)

        assert delay >= 0
        delays.append(delay)
        classes.append(class_pertrack[track_id])
        ids.append(track_id)
        first_frames.append(first_frame_pertrack[track_id])
        sizes.append(size)

    return delays, classes, ids, first_frames, sizes


def read_results(filename, conf, condition, p_high, p_low):
    bboxes = defaultdict(list)
    scores = defaultdict(list)
    cls_inds = defaultdict(list)
    if not os.path.exists(filename):
        filename = filename.replace('txt', 'csv')
    for line in open(filename):
        line = line.split() if len(line.split()) >= 7 else line.split(',')
        frame_id, cls, score = int(line[0]), line[1], float(line[2])
        xmin, ymin, xmax, ymax = [float(k) for k in line[3:7]]
        p = np.random.random()
        p_box = p_high if condition(line) else p_low
        if score >= conf and p > p_box:
            frame_id = int(frame_id)
            bboxes[frame_id].append((xmin, ymin, xmax, ymax))
            scores[frame_id].append(score)
            cls_inds[frame_id].append(cls)
    for frame_id in bboxes:
        bboxes[frame_id] = np.array(bboxes[frame_id])
        scores[frame_id] = np.array(scores[frame_id])
        cls_inds[frame_id] = np.array(cls_inds[frame_id])
    return bboxes, scores, cls_inds


def readKITTI(filename):
    bboxes = defaultdict(list)
    cls_inds = defaultdict(list)
    track_ids = defaultdict(list)
    for fields in csv.reader(open(filename)):
        frame_id = int(fields[0])
        xmin, ymin, xmax, ymax = map(float, fields[3:7])
        bboxes[frame_id].append((xmin, ymin, xmax, ymax))
        cls_inds[frame_id].append(fields[2])
        track_ids[frame_id].append(int(fields[1]))
    for frame_id in bboxes:
        bboxes[frame_id] = np.array(bboxes[frame_id])
        cls_inds[frame_id] = np.array(cls_inds[frame_id])
        track_ids[frame_id] = np.array(track_ids[frame_id])
    return bboxes, cls_inds, track_ids


def get_delays(res_files, detector, conf, condition, p_high, p_low):
    delays = []
    classes = []
    first_frames = []
    sizes = []
    tags = []
    for video_name in res_files:
        gt_file = vid_annotations_path + '/{}.csv'.format(video_name)
        bboxes_gt, cls_inds_gt, track_ids = readKITTI(gt_file)
        det_file_path = detections_path + 'synthetic/{}/{}.txt'.format(detector, video_name)
        bboxes, scores, cls_inds = read_results(det_file_path, conf, condition, p_high, p_low)

        delay_tmp, classes_tmp, ids, first_frames_tmp, sizes_tmp = calc_delay(bboxes_gt, cls_inds_gt, track_ids, bboxes,
                                                                              scores, cls_inds)
        delays.extend(delay_tmp)
        classes.extend(classes_tmp)
        first_frames.extend(first_frames_tmp)
        sizes.extend(sizes_tmp)
        tags.extend(map(lambda x: '{}-{}'.format(video_name, x), ids))
    return np.array(delays), np.array(classes), np.array(first_frames), tags, np.array(sizes)


def eval_delay(delays, choice):
    delays = np.array(delays).astype('f')
    if choice == 'mean':
        return np.mean(delays)
    elif choice == 'median':
        return np.median(delays)
    elif choice == 'logmean':
        return np.exp(np.mean(np.log(delays + 1)))
    elif choice == 'harmean':
        return 1.0 / np.mean(1.0 / (delays + 1)) - 1
    elif choice == 'clipmax':
        return np.mean(np.minimum(delays, 30))
    else:
        raise Exception


def get_mD(res_files, detector, confs, choice='mean', condition=lambda x, y: True, p_high=1., p_low=1., verbose=False):
    delays = []
    nonfirst_delays = []
    small_delays = []
    med_delays = []
    large_delays = []
    dataset = Dataset('VID')
    targeted_cls = range(len(dataset.classes))
    delays_per_classes = [[] for _ in targeted_cls]
    for conf in confs:
        delays_atconf, classes, first_frames, tags, sizes = get_delays(res_files, detector, conf, condition, p_high,
                                                                       p_low)
        delays.append(eval_delay(delays_atconf, choice=choice))
        small_delays.append(eval_delay(delays_atconf[np.where(sizes < 40)[0]], choice=choice))
        med_delays.append(eval_delay(delays_atconf[np.where((sizes >= 40) * (sizes < 100))[0]], choice=choice))
        large_delays.append(eval_delay(delays_atconf[np.where(sizes >= 100)[0]], choice=choice))
        nonfirst_delay = [delays_atconf.__getitem__(k) for k in
                          filter(lambda x: first_frames[x] != 0, range(len(delays_atconf)))]
        nonfirst_delays.append(eval_delay(nonfirst_delay, choice=choice))
        for idx, cls in enumerate(targeted_cls):
            delays_per_classes[idx].append(eval_delay(delays_atconf[np.where(classes == cls)[0]], choice=choice))

    if verbose:
        print("Number of GT instances:", len(delays_atconf))
        print("mD for all:  ", eval_delay(delays, choice='harmean'))
        print("mD for small:", eval_delay(small_delays, choice='harmean'))
        print("mD for med:  ", eval_delay(med_delays, choice='harmean'))
        print("mD for large:", eval_delay(large_delays, choice='harmean'))
        for idx, cls in enumerate(targeted_cls):
            print("%10s AD:" % (dataset.class_names()), eval_delay(delays_per_classes[idx], choice='harmean'))

    return delays, eval_delay(delays, choice='harmean')


if __name__ == '__main__':
    pass
