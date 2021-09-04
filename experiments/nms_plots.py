import matplotlib.pyplot as plt
from constants import pr_plot_path, detector_names
from datasets.dataset import Dataset
from metrics.AP_all_classes import APAll
from metrics.AP_utils import voc_ap
from metrics.vmap import VmAP
from argparse import ArgumentParser

from models.detector import Detector


def nms_plot(num_frames=10, dataset_name='COCO', class_names=None, verbose=False):
    dataset = Dataset(dataset_name, pre_load=False, class_names=class_names)
    for det in detector_names[dataset]:
        print(det)
        if det == 'GT':
            continue
        detector = Detector(dataset, det, 0.1, None, None, 1, True, verbose)
        vmap = VmAP(dataset, detector, VmAP.voc)
        x = vmap.score
        detector = Detector(dataset, det, 0.1, None, None, num_frames, True, verbose)
        vmap = VmAP(dataset, detector, VmAP.voc)
        y = vmap.score
        plt.scatter(x, y, marker='o')
    plt.legend(detector_names[dataset])
    plt.xlim(40, 60)
    plt.ylim(50, 70)
    plt.xlabel('VmAP')
    plt.ylabel('VmAP after t-NMS')
    plt.plot([20, 60], [20, 60], 'k--', alpha=0.4)
    filename = 'results_{}/nms_{}_frames'.format(dataset, num_frames)
    plt.savefig(filename + '.pdf')
    plt.cla()


def nms_pr_curve_case(detectors, dataset_name, class_names=None, verbose=False):
    dataset = Dataset(dataset_name, pre_load=False, class_names=class_names)
    for det in detectors:
        for nms_frames in [1, 10]:
            detector = Detector(dataset, det, 0, None, None, nms_frames, True, verbose)
            vmap = APAll(dataset, detector, None)
            rec, prec, _, _, _ = vmap.get_precision_recall(None)
            ap, m_rec, m_prec = voc_ap(rec, prec)
            plt.plot(m_rec, m_prec)
    plt.xlabel('V-Recall')
    plt.ylabel('V-Precision')
    plt.legend(['Without NMS', 'With NMS'], loc=3)
    filename = '{}/nms_pr_'.format(pr_plot_path, dataset) + str(detectors)
    plt.savefig(filename + ".pdf")
    plt.cla()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='VID')
    parser.add_argument('--nms_window', default=3)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--detectors', nargs='+', default=None)
    parser.add_argument('--classes', nargs='+', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    detectors = args.detectors if args.detectors else detector_names[args.dataset]
    nms_pr_curve_case(detectors, args.dataset, class_names=args.classes)
