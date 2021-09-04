import csv

from constants import set_formation_criteria, set_formation_ablation_path, detector_names
from datasets.dataset import Dataset
from metrics.mAP import MAP
from metrics.vmap import VmAP
from argparse import ArgumentParser

from models.detector import Detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='VID')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--detectors', nargs='*', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    detectors = args.detectors if args.detectors else detector_names[args.dataset]
    results = [['detector', 'mAP', 'VmAPl', 'VmAPa', 'VmAPal', 'VmAPt']]
    dataset = Dataset(args.dataset, True)
    for det in detectors:
        if args.verbose:
            print(det, end='\t')
        detector = Detector(dataset, det, 0.1, verbose=args.verbose)

        map_ = MAP(dataset, detector, MAP.voc)
        if args.verbose:
            print(map_.score, end='\t')

        res = [det, map_.score]
        for cr in set_formation_criteria:
            dataset = Dataset(args.dataset, True, set_criteria=cr)
            detector = Detector(dataset, det, 0.1, verbose=args.verbose)
            vmap_cr = VmAP(args.dataset, detector, VmAP.voc)
            if args.verbose:
                print(vmap_cr.score, end='\t')
            res.append(vmap_cr.score)
        if args.verbose:
            print()
        results.append(res)
    with open('{}/{}.csv'.format(set_formation_ablation_path, args.dataset), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(results)
