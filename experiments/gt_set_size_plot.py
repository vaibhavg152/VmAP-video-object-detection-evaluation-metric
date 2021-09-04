from argparse import ArgumentParser

from datasets.dataset import Dataset
from viz import average_gt_size_plot

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='VID')
    args = parser.parse_args()
    average_gt_size_plot(Dataset(args.dataset))
