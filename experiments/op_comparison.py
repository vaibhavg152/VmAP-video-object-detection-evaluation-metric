import csv
import os
from argparse import ArgumentParser

from constants import experiments_results_path, detector_names, op_scores_cache_path
from metrics.mAP import MAP
from metrics.vmap import VmAP
from models.detector import Detector


def dump_op_v_recalls(dataset, tracker_name=None, verbose=False):
    op_scores = {}
    for det in detector_names[dataset.name]:
        if verbose:
            print(det)
        detector_vmap = Detector(dataset, det, 'max_vf1', tracker_name=tracker_name, verbose=verbose)
        vmap = VmAP(dataset, detector_vmap, None)
        detector = Detector(dataset, det, 'max_f1', tracker_name=tracker_name)
        map_ = VmAP(dataset, detector, None)
        op_scores[det] = {
            vmap.name: vmap.read_scores(
                results_path='{}/{}/{}/OPVideoNone/'.format(op_scores_cache_path, vmap.name, dataset.name)),
            MAP.name: map_.read_scores(
                results_path='{}/{}/{}/OPFrameNone/'.format(op_scores_cache_path, map_.name, dataset.name))}

    for idx, cls_name in enumerate(dataset.classes):
        results = [['Detector', 'mAP', 'VR@mAP', 'FP@mAP', 'VmAP', 'VR@VmAP', 'FP@VmAP']]
        for det in detector_names[dataset.name]:
            res_map = op_scores[det][MAP.name][cls_name]
            res_vmap = op_scores[det][VmAP.name][cls_name]
            results.append([det, res_map['AP'], res_map['Recall'], res_map['FP'], res_vmap['AP'], res_vmap['Recall'],
                            res_vmap['FP']])
            # print(results[-1])

        out_dir = '{}/operating_point_comparison/'.format(experiments_results_path)
        os.makedirs(out_dir, exist_ok=True)
        with open('{}/operating_point_comparison/{}.csv'.format(experiments_results_path, cls_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(results)
            print('Info: results saved in {}'.format(experiments_results_path + 'operating_point_comparison/'))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='COCO')
    parser.add_argument('--verbose', type=bool, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dump_op_v_recalls(args.dataset)
