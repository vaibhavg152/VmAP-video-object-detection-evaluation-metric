from argparse import ArgumentParser

from constants import detector_names
from metrics.mAP import MAP
from models.trackers.tracker import track_videos


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tracker', default='goturn')
    parser.add_argument('--precision', default=0.9)
    parser.add_argument('--dataset', default='COCO')
    parser.add_argument('--classes', nargs='*', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for detector_name in detector_names[args.dataset]:
        map_ = MAP(args.dataset)
        op_points = map_.read_op_scores(args.precision)
        track_videos(detector=detector_name, tracker_name=args.tracker, dataset=args.dataset, classes=args.classes,
                     verbose=False)
