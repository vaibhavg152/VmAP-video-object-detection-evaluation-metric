import csv
import os
from argparse import ArgumentParser

import numpy as np

from constants import ranking_results_path
from datasets.dataset import Dataset
from experiments.post_tracker_ranking import eval_k_fold
from models.trackers.single_object_tracker import OpenCVTracker


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--tracker", action='store', default='SORT')
    parser.add_argument("--precision", type=float, default=0.9)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--dataset", default='COCO')
    return parser.parse_args()


def main():
    args = parse_args()
    if OpenCVTracker.OPENCV_OBJECT_TRACKERS or args.tracker == 'KF':
        dataset = Dataset(args.dataset, num_objects=1)
    else:
        dataset = Dataset(args.dataset)

    map_res, vmap_res = eval_k_fold(args.num_samples, args.tracker, dataset, args.precision)
    print("Mean\tVR: ", np.mean(vmap_res), "\tFR:", np.mean(map_res))
    print("Variance\tVR: ", np.var(vmap_res), "\tFR:", np.var(map_res))
    print("Standard deviation\tVR: ", np.std(vmap_res), "\tFR:", np.std(map_res))
    print("Difference in all values:", [vmap_res[i] - map_res[i] for i in range(len(vmap_res))])

    os.makedirs(ranking_results_path, exist_ok=True)
    with open('{}/k_fold_{}.csv'.format(ranking_results_path, args.num_samples), 'w+') as f:
        wr = csv.writer(f)
        wr.writerows([map_res, vmap_res])


if __name__ == '__main__':
    main()
