import csv
import os
import random
from argparse import ArgumentParser

from scipy.stats import truncnorm

from datasets.dataset import Dataset
from metrics.mAP import MAP
from metrics.vmap import VmAP
from video_elements.bndbox import BoundingBox
from models.trackers.tracker import track_videos
from synthetic_biased_detectors import SyntheticDetector


class RandomDetector(SyntheticDetector):

    def __init__(self, dataset, threshold):
        super().__init__('RANDOM_{}'.format(threshold), dataset, threshold, precision=1)

    def write_true_positives(self, video):
        gt = video.load_sets(True)
        for obj in gt.objs.all_values():
            for gt_set in obj.sets:
                num_frames = len(gt_set) // self.threshold
                if not num_frames:
                    continue
                # randomly select num_frames results
                out_frame_ids = random.sample(list(gt_set.frames.keys()), num_frames)
                conf_tp = truncnorm(-0.5, 0.5, loc=0.5, scale=1).rvs(num_frames)
                for fid, conf in zip(out_frame_ids, conf_tp):
                    self.frame_wise_dets[fid] = self.frame_wise_dets.get(fid, []) + [
                        BoundingBox(video.name, fid, obj.class_name, gt_set.frames[fid], conf)]

    def get_values(self, video):
        results = []
        for fid in self.frame_wise_dets.keys():
            for dets in self.frame_wise_dets[fid]:
                results.append(list(dets))
        return results, []

    def condition(self, x):
        return True

    def filter_detections(self):
        self.detections = {}
        for cls_name in self.classes:
            self.detections[cls_name] = []

        if not os.path.exists(self.detections_path):
            self.write_detections(False)
        for video_name in self.video_names:
            file_path = self.detections_path + video_name + '.csv'
            if not os.path.exists(file_path):
                self.write_detections(False)
            detections = {}
            with open(file_path) as f:
                for line in csv.reader(f):
                    frame_id = int(line[0])
                    cls_id = self.dataset.class_name_from_id(line[1])
                    bbox = [int(i) for i in line[3:7]]
                    confidence = float(line[2])
                    detections[frame_id] = detections.get(frame_id, []) + [
                        BoundingBox(video_name, frame_id, cls_id, bbox, confidence)]

            for frame_idx, boxes in detections.items():
                for det in boxes:
                    cls_name = det.class_name
                    if cls_name in self.detections:
                        self.detections[cls_name].append(det)

        for cls_name in self.detections.keys():
            self.detections[cls_name] = sorted(self.detections[cls_name], key=lambda x: x.confidence, reverse=True)
        return self.detections


def track_and_evaluate(dataset, detector, tracker, tracker_age, min_hits, verbose):
    track_videos(detector, tracker, dataset=dataset, tracker_age=tracker_age, verbose=verbose, min_hits=min_hits)

    map_before = MAP(dataset, detector, MAP.voc)
    if verbose:
        print("mAP before tracking:", map_before.class_wise_scores['airplane'])

    vmap_before = VmAP(dataset, detector, VmAP.voc)
    if verbose:
        print("VmAP before tracking:", vmap_before.class_wise_scores['airplane'])

    map_ = MAP(dataset, detector, MAP.voc)
    if verbose:
        print("mAP after tracking:", map_.class_wise_scores['airplane'])

    vmap = VmAP(dataset, detector, VmAP.voc)
    if verbose:
        print("VmAP after tracking:", vmap.class_wise_scores['airplane'])

    return map_before.score, vmap_before.score, map_.score, vmap.score


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='VID')
    parser.add_argument('--tracker', default='SORT')
    parser.add_argument('--tracker_max_age', type=int, default=30)
    parser.add_argument('--tracker_min_hits', type=int, default=1)
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = Dataset(args.dataset)
    maps_before, vmaps_before = [], []
    maps_after, vmaps_after = [], []
    result = [['frames', 'map_', 'vmap', 'map+SORT', 'vmap+SORT']]
    x = [25, 20, 15, 12, 10, 7, 5, 4, 3, 2, 1]
    for n in x:
        detector = RandomDetector(dataset, threshold=n)
        detector.write_detections()
        mb, vb, ma, va = track_and_evaluate(dataset, detector, args.tracker, args.tracker_max_age,
                                            args.tracker_min_hits, args.verbose)
        maps_before.append(mb)
        maps_after.append(ma)
        vmaps_before.append(vb)
        vmaps_after.append(va)
        print(n, mb, ma, vb, va)
        result.append([n, mb, vb, ma, va])

    if args.dump:
        import csv

        with open('values.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerows(result)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(x, maps_before, c='r')
        plt.plot(x, vmaps_before, c='g')
        plt.plot(x, maps_after, c='b')
        plt.plot(x, vmaps_after, c='m')
        plt.xlabel('Number of frames per detection.')
        plt.ylabel('Score after tracking.')
        plt.legend(result[0][1:])
        plt.savefig('tracker_with_missing_frames.png')


if __name__ == '__main__':
    main()
