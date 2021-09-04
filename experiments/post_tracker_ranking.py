import csv
import os
import random
from argparse import ArgumentParser

import numpy as np
import scipy.stats as ss
from sklearn.cluster import KMeans
from sklearn.metrics import ndcg_score

from constants import op_scores_cache_path, ranking_results_path, detector_names
from datasets.dataset import Dataset
from metrics.mAP import MAP
from metrics.vmap import VmAP
from models.detector import Detector
from models.trackers.single_object_tracker import OpenCVTracker


def dump_rankings(dataset, tracker, fix_precision, results_path=None, verbose=False):
    output = []
    dataset_name = '{}_SOT'.format(dataset.name) if dataset.num_objects else dataset.name
    print(dataset_name, fix_precision, tracker)

    for det in detector_names[dataset.name]:
        if verbose:
            print("\n============== {} ==============".format(det))
        detector_op_map = Detector(dataset, det, 'max_f1', fix_precision, eager_load=False)
        detector_op_vmap = Detector(dataset, det, 'max_vf1', fix_precision, eager_load=False)
        detector_op_map_track = Detector(dataset, det, 'max_f1', fix_precision, tracker, eager_load=False)
        if verbose:
            print("loaded detectors")
        map_ = MAP(dataset, detector_op_map, None)
        vmap = VmAP(dataset, detector_op_vmap, None)
        vmap_at_map = VmAP(dataset, detector_op_map, None)
        vmap_at_map_track = VmAP(dataset, detector_op_map_track, None)
        if verbose:
            print("loaded metrics")

        scores_path_vmap = '{}/{}/{}/{}/OP_mAP/'.format(op_scores_cache_path, map_.name, dataset_name, fix_precision)
        if not os.path.exists('{}/{}.json'.format(scores_path_vmap, det)):
            vmap_at_map.evaluate_voc(cache=True, cache_path=scores_path_vmap)

        if not os.path.exists('{}/{}_{}.json'.format(scores_path_vmap, tracker, det)):
            vmap_at_map_track.evaluate_voc(cache=True, cache_path=scores_path_vmap)

        map_op = map_.read_op_scores(fix_precision, scores_path=results_path)
        op_track_map = map_.read_op_scores(fix_precision, scores_path=results_path)
        vmap_at_map = vmap_at_map.read_op_scores(fix_precision, scores_path_vmap)
        vmap_at_map_track = vmap_at_map_track.read_op_scores(fix_precision, scores_path_vmap)
        if verbose:
            print("loaded scores")

        if fix_precision:
            f_op_map = np.average([nums['Recall'] for nums in map_op.values()])
            v_op_map = np.average([nums['Recall'] for nums in vmap_at_map.values()])
            f_op_map_track = np.average([nums['Recall'] for nums in op_track_map.values()])
            v_op_map_track = np.average([nums['Recall'] for nums in vmap_at_map_track.values()])
            output.append(
                [det, round(f_op_map, 2), -1, round(v_op_map, 2), -1, round(f_op_map_track, 2), -1,
                 round(v_op_map_track, 2), -1])
        else:
            detector_op_vmap_track = Detector(dataset, det, 'max_vf1', fix_precision, tracker, eager_load=False)
            vmap_op = vmap.read_op_scores(fix_precision, scores_path=results_path)
            op_track_vmap = vmap.read_op_scores(fix_precision, scores_path=results_path)
            scores_path_map = '{}/{}/{}/{}/OP_VmAP/'.format(op_scores_cache_path, map_.name, dataset_name,
                                                            fix_precision)
            map_at_vmap = MAP(dataset, detector_op_vmap, None)
            if not os.path.exists('{}/{}.json'.format(scores_path_map, det)):
                map_at_vmap.evaluate_voc(cache=True, cache_path=scores_path_map)
            map_at_vmap = map_at_vmap.read_op_scores(fix_precision, scores_path_map)

            map_at_vmap_track = MAP(dataset, detector_op_vmap_track, None)
            if not os.path.exists('{}/{}_{}.json'.format(scores_path_map, tracker, det)):
                map_at_vmap_track.evaluate_voc(cache=True, cache_path=scores_path_map)
            map_at_vmap_track = map_at_vmap_track.read_op_scores(fix_precision, scores_path_map)

            f_op_map = np.average([nums['F1'] for nums in map_op.values()])
            v_op_map = np.average([nums['F1'] for nums in vmap_at_map.values()])
            f_op_map_track = np.average([nums['F1'] for nums in op_track_map.values()])
            v_op_map_track = np.average([nums['F1'] for nums in vmap_at_map_track.values()])

            f_op_vmap = np.average([nums['F1'] for nums in map_at_vmap.values()])
            v_op_vmap = np.average([nums['F1'] for nums in vmap_op.values()])
            f_op_vmap_track = np.average([nums['F1'] for nums in map_at_vmap_track.values()])
            v_op_vmap_track = np.average([nums['F1'] for nums in op_track_vmap.values()])
            output.append(
                [det, round(f_op_map, 2), -1, round(v_op_map, 2), -1, round(f_op_map_track, 2), -1,
                 round(v_op_map_track, 2), -1, round(f_op_vmap, 2), -1, round(v_op_vmap, 2), -1,
                 round(f_op_vmap_track, 2), -1, round(v_op_vmap_track, 2), -1])

    output_np = np.array(output)
    n = len(output)
    for idx in range(2, output_np.shape[1], 2):
        output_np[:, idx] = n + 1 - ss.rankdata(output_np[:, idx - 1].astype(np.float))
    print("{} {}".format(output_np[:, 8], n + 1 - ss.rankdata(output_np[:, 7])))

    if results_path is None:
        os.makedirs(ranking_results_path, exist_ok=True)
        results_path = '{}/{}{}_{}.csv'.format(ranking_results_path, fix_precision if fix_precision else '',
                                               dataset_name, tracker)
    with open(results_path, 'w') as f:
        wr = csv.writer(f)
        header = [
            ['Detector', 'fF1 OPmAP', 'rank', 'vF1 OPmAP', 'rank', 'fF1 OPmAP+tracker', 'rank', 'vF1 OPmAP+tracker',
             'rank', 'fF1 OPVmAP', 'rank', 'vF1 OPVmAP', 'rank', 'fF1 OPVmAP+tracker', 'rank', 'vF1 OPVmAP+tracker',
             'rank']]
        if fix_precision:
            header = [['Detector', 'FR OPmAP', 'rank', 'VR OPmAP', 'rank', 'fF1 OPmAP+tracker', 'rank',
                       'vF1 OPmAP+tracker', 'rank', ]]
        wr.writerows(header + output_np.tolist())


def plot_rankings(dataset, tracker, fix_precision=None, results_path=None):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['figure.figsize'] = 10, 5

    dataset_name = '{}_SOT'.format(dataset.name) if dataset.num_objects else dataset.name
    if results_path is None:
        results_path = '{}/{}{}_{}.csv'.format(ranking_results_path, fix_precision if fix_precision else '',
                                               dataset_name, tracker)

    output_np = np.array(list(csv.reader(open(results_path))))[1:, :]
    output_np = np.array(sorted(output_np, key=lambda x: x[5]))
    plt.scatter(output_np[:, 0], output_np[:, 1].astype(float))
    plt.scatter(output_np[:, 0], output_np[:, 3].astype(float))
    plt.scatter(output_np[:, 0], output_np[:, 5].astype(float))
    plt.scatter(output_np[:, 0], output_np[:, 7].astype(float))
    plt.legend(['mAP', 'VmAP', 'F1@OPmAP+SORT', 'F1@OPVmAP+SORT'])
    plt.savefig(results_path.replace('.csv', '.pdf'), dpi=100, bbox_inches='tight')
    plt.show()
    plt.cla()


def eval_rankings(dataset, tracker, fix_precision=None, results_path=None):
    dataset_name = '{}_SOT'.format(dataset.name) if dataset.num_objects else dataset.name
    if results_path is None:
        results_path = '{}/{}{}_{}.csv'.format(ranking_results_path, fix_precision if fix_precision else '',
                                               dataset_name, tracker)
        if not os.path.exists(results_path):
            dump_rankings(dataset, tracker, fix_precision, results_path)

    output_np = np.array(list(csv.reader(open(results_path))))[1:, :]
    n = output_np.shape[0]
    diff_map = np.abs((output_np[:, 6].astype(float) - output_np[:, 2].astype(float)))
    diff_vmap = np.abs((output_np[:, 6].astype(float) - output_np[:, 4].astype(float)))
    spearman_correlation_map = 1 - ((6 * sum(np.power(diff_map, 2))) / (n * n * n - n))
    spearman_correlation_vmap = 1 - ((6 * sum(np.power(diff_vmap, 2))) / (n * n * n - n))
    print("Dataset:", dataset_name, "Number of detectors:", n)
    print("VmAP rank distance: ", sum(diff_vmap))
    print("mAP rank distance: ", sum(diff_map))
    print("Spearman correlation for VmAP: ", spearman_correlation_vmap)
    print("Spearman correlation for mAP: ", spearman_correlation_map)
    gt_values = [output_np[:, 5].astype(float)]
    print("mAP", ndcg_score(gt_values, [output_np[:, 1].astype(float)]))
    print("VmAP", ndcg_score(gt_values, [output_np[:, 3].astype(float)]))

    return spearman_correlation_map, spearman_correlation_vmap


def eval_k_fold(num_samples, tracker, dataset, fix_precision=None):
    from metrics.Metric import get_video_wise_op_results
    dataset_name = '{}_SOT'.format(dataset.name) if dataset.num_objects else dataset.name
    video_names = dataset.video_names
    random.shuffle(video_names)
    sample_size = len(video_names) // num_samples

    map_accuracy, vmap_accuracy = [], []
    for i in range(num_samples):
        sampled_videos = video_names[i:] if i == num_samples - 1 else video_names[i:i + sample_size]
        dataset = Dataset(dataset.name, True, False, dataset.classes, sampled_videos, dataset.num_objects,
                          set_criteria=dataset.set_criteria, gamma=dataset.gamma)
        output = []
        for det in detector_names[dataset][:]:
            detector = Detector(dataset, det, 0, tracker_name=tracker)
            map_ = MAP(dataset, detector, None)
            op_points = map_.get_operating_points(fix_precision)
            detector = Detector(dataset, det, op_points, tracker_name=tracker)

            map_op = get_video_wise_op_results(dataset, detector, True)
            vmap_op = get_video_wise_op_results(dataset, detector, False)
            op_track_map = get_video_wise_op_results(dataset, detector, True)
            op_track_vmap = get_video_wise_op_results(dataset, detector, False)

            f_op_map = np.average([nums['Recall'] for nums in map_op.values()])
            v_op_map = np.average([nums['Recall'] for nums in vmap_op.values()])
            f_op_map_track = np.average([nums['Recall'] for nums in op_track_map.values()])
            v_op_map_track = np.average([nums['Recall'] for nums in op_track_vmap.values()])
            output.append(
                [det, round(f_op_map, 2), -1, round(v_op_map, 2), -1, round(f_op_map_track, 2), -1,
                 round(v_op_map_track, 2), -1])
        output_np = np.array(output)
        n = len(output)
        for idx in range(2, output_np.shape[1], 2):
            output_np[:, idx] = n + 1 - ss.rankdata(output_np[:, idx - 1].astype(np.float))
        os.makedirs(ranking_results_path, exist_ok=True)
        results_path = '{}/{}{}_{}_{}_{}fold.csv'.format(ranking_results_path, fix_precision if fix_precision else '',
                                                         dataset_name, tracker, i, num_samples)
        with open(results_path, 'w') as f:
            wr = csv.writer(f)
            header = [['Detector', 'FR OPmAP', 'rank', 'VR OPmAP', 'rank', 'fF1 OPmAP+tracker', 'rank',
                       'vF1 OPmAP+tracker', 'rank', ]]
            wr.writerows(header + output_np.tolist())

        acc_map, acc_vmap = eval_rankings(dataset, tracker, results_path=results_path)
        map_accuracy.append(acc_map)
        vmap_accuracy += [acc_vmap]
    return map_accuracy, vmap_accuracy


def eval_clustered(tracker, dataset, gt_col=6, fix_precision=None):
    dataset_name = '{}_SOT'.format(dataset.name) if dataset.num_objects else dataset.name
    results_path = '{}/{}{}_{}.csv'.format(ranking_results_path, fix_precision if fix_precision else '', dataset_name,
                                           tracker)
    if not os.path.exists(results_path):
        dump_rankings(dataset, tracker, fix_precision, results_path)
    output_np = np.array(list(csv.reader(open(results_path))))[1:, :]
    # detectors = output_np[:, 0]
    values = output_np[:, 1:].astype(float)
    # n = values.shape[0]
    rand_state = 7
    clustering_map = KMeans(n_clusters=3, random_state=rand_state).fit(X=values[:, :1]).labels_
    clustering_vmap = KMeans(n_clusters=3, random_state=rand_state).fit(X=values[:, :3]).labels_
    clustering_gt = KMeans(n_clusters=3, random_state=rand_state).fit(X=values[:, :gt_col - 1]).labels_
    print(clustering_gt, clustering_vmap, clustering_map)
    print("mAP diff:\t", np.abs(clustering_gt - clustering_map))
    print("VmAP diff:\t", np.abs(clustering_gt - clustering_vmap))
    print("Spearman for mAP:\t", ss.pearsonr(clustering_gt, clustering_map)[0])
    print("Spearman for VmAP:\t", ss.pearsonr(clustering_gt, clustering_vmap)[0])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--tracker', default='SORT', help='Tracker to be used to evaluate. One of: SORT, IDEAL, IOU_LA,'
                                                          'KF, csrt, kcf, boosting, mil,tld, medianflow, mosse, goturn')
    parser.add_argument('--precision', default=0.9, help='Precision at which to run the tracker.')
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--dataset', default='COCO')
    return parser.parse_args()


def main():
    args = parse_args()
    tracker = args.tracker
    assert tracker.upper() in 'SORT, IDEAL, IOU_LA, KF, csrt, kcf, boosting, mil,tld, medianflow, mosse, goturn'.upper()
    precision = args.precision

    if OpenCVTracker.OPENCV_OBJECT_TRACKERS or args.tracker == 'KF':
        dataset = Dataset(args.dataset, num_objects=1)
    else:
        dataset = Dataset(args.dataset)
    if args.verbose:
        print('Dataset loaded.')

    dump_rankings(dataset, tracker, precision, verbose=args.verbose)
    if args.verbose:
        print('Rankings computed.')

    eval_rankings(dataset, tracker, precision)
    plot_rankings(dataset, tracker, precision)
    if args.verbose:
        print('Results saved.')


if __name__ == '__main__':
    main()