from argparse import ArgumentParser

from datasets.dataset import Dataset
from metrics.mAP import mAP
from metrics.vmap import vmap
from models.detector import Detector


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='COCO')
    parser.add_argument('--num_videos', default=None)
    parser.add_argument('--video_names', nargs='+', default=None)
    parser.add_argument('--class_names', nargs='+', default=None)
    parser.add_argument('--detector', required=True)
    parser.add_argument('--nms_num_frames', type=int, default=1)
    parser.add_argument('--tracker', default=None)
    parser.add_argument('--verbose', type=bool, default=True)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--vmap', action='store_true')
    group.add_argument('--map', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--conf', type=float, nargs='+', help='confidence threshold over which to perform tracking.')
    group.add_argument('--OP', action='store_true', help='evaluate at OP')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vid_dataset = Dataset(dataset_name=args.dataset, pre_load=True, num_videos=args.num_videos,
                          video_names=args.video_names, class_names=args.class_names)
    conf = 'max_f1' if args.OP else args.conf
    mega = Detector(dataset=vid_dataset, detector_name=args.detector, confidence=conf, tracker_name=args.tracker,
                    verbose=args.verbose, t_nms_window=args.nms_num_frames)
    if args.vmap:
        x, y = vmap(vid_dataset, mega, args.verbose)
    else:
        x, y = mAP(vid_dataset, mega, args.verbose)

    tps = sum([sc['TP'] for sc in y.values()])
    fps = sum([sc['FP'] for sc in y.values()])
    fns = sum([sc['FN'] for sc in y.values()])
    print(tps, fps, fns, x)
