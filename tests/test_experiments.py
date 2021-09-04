import csv
import os

from constants import synthetic_detector_results_path, experiments_results_path
from datasets.dataset import COCO, VID


def test_op_comparison():
    from experiments.op_comparison import dump_op_v_recalls
    dataset = COCO(pre_load=True, num_videos=10)
    dump_op_v_recalls(dataset)
    for cls_name in dataset.classes:
        print(cls_name)
        assert os.path.exists('{}/operating_point_comparison/{}.csv'.format(experiments_results_path, cls_name))


def test_op_comparison_correct():
    from experiments.op_comparison import dump_op_v_recalls
    dataset = VID(pre_load=True)
    dump_op_v_recalls(dataset)
    cls_name = 'hamster'
    detectors = ['DFF', 'FGFA', 'MEGA', 'RDN']
    file_path = '{}/operating_point_comparison/{}.csv'.format(experiments_results_path, cls_name)
    assert os.path.exists(file_path)
    with open(file_path) as f:
        for line in csv.reader(f):
            if line[0] == detectors[0]:
                assert int(float(line[2])) == 87
                assert int(float(line[3])) == 108
                assert int(float(line[5])) == 85
                assert int(float(line[6])) == 1
            elif line[0] == detectors[1]:
                assert int(float(line[2])) == 89
                assert int(float(line[3])) == 105
                assert int(float(line[5])) == 83
                assert int(float(line[6])) == 1
            elif line[0] == detectors[2]:
                assert int(float(line[2])) == 91
                assert int(float(line[3])) == 258
                assert int(float(line[5])) == 87
                assert int(float(line[6])) == 3
            elif line[0] == detectors[3]:
                assert int(float(line[2])) == 96
                assert int(float(line[3])) == 109
                assert int(float(line[5])) == 93
                assert int(float(line[6])) == 2
            continue


def test_bias_synthetic_exp():
    from experiments.synthetic_biased_detectors import write_scores, plot_synthetic, BiasedDetectorSize, \
        BiasedDetectorSpeed, BiasedDetectorContrast, BiasedDetectorIllumination, BiasedDetectorHue
    dataset = VID(pre_load=True)
    metrics = ["VmAP", "mAP", "LNmAP", 'KFmAP', 'AD', 'VmAP_20']
    precision = 0.5
    detectors = [BiasedDetectorSize(dataset, precision=precision),
                 BiasedDetectorSpeed(dataset, precision=precision),
                 # BiasedDetectorHue(dataset, precision=precision),
                 # BiasedDetectorContrast(dataset, precision=precision),
                 # BiasedDetectorIllumination(dataset, precision=precision)
                 ]
    write_scores(detectors, dataset, metrics)
    file_path = "{}/all_{}_bias.json".format(synthetic_detector_results_path, BiasedDetectorSpeed.name)
    assert os.path.exists(file_path)
    file_path = 'plot_synthetic.pdf'
    plot_synthetic(detectors, metrics, save_path=file_path)
    assert os.path.exists(file_path)


def test_data_curation_create():
    from experiments.eval_curated_datasets import write_datasets_123, write_datasets_abc
    write_datasets_123()
    write_datasets_abc()


def test_data_curation_eval():
    from experiments.eval_curated_datasets import eval_custom_datasets
    eval_custom_datasets('RDN')


def test_ranking_exp():
    from experiments.post_tracker_ranking import dump_rankings, eval_rankings, plot_rankings
    dataset = COCO(True)
    dump_rankings(dataset, 'SORT', 0.9)
    eval_rankings(dataset, 'SORT', 0.9)
    plot_rankings(dataset, 'SORT', 0.9)
    eval_rankings(dataset, 'SORT', 0.8)
    plot_rankings(dataset, 'SORT', 0.8)


if __name__ == '__main__':
    # test_data_curation_create()
    # test_data_curation_eval()
    test_ranking_exp()
    # test_op_comparison_correct()
