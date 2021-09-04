import csv
import glob
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from constants import synthetic_detections_path, synthetic_detector_results_path, vid_annotations_path, \
    vid_image_path, set_formation_app, set_formation_time, set_formation_app_loc
from datasets.dataset import Dataset, VID
from metrics.average_delay import get_mD
from metrics.kfmap import KFmAP
from metrics.lnmap import LNmAP
from metrics.mAP import MAP
from metrics.tmap import TVmAP
from metrics.vmap import VmAP
from video_elements.bndbox import BoundingBox
from models.detector import Detector
from models.trackers.tracker import track_videos
from video_elements.video_elements import Frame
from viz import draw_box_on_image

p_highs_default = [0.5, 0.57, 0.64, 0.7, 0.77, 0.8, 0.82]
p_lows_default = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0]


class SyntheticDetector(Detector):

    def __init__(self, name, dataset, threshold, precision=0.5):
        self.detections_path = '{}/{}/'.format(synthetic_detections_path, name)
        self.precision = precision
        self.threshold = threshold
        self.frame_wise_dets = None
        self.all_values = None
        self.num_tp = None
        super().__init__(dataset, name, eager_load=False)

    def write_detections(self, plot=False):
        os.makedirs(self.detections_path, exist_ok=True)
        self.all_values = []
        for video in self.dataset:
            self.write_true_positives(video)
            if 0 < self.precision < 1:
                self.write_false_positives(video)
            results, values = self.get_values(video)
            results.sort(key=lambda x: x[0])
            self.all_values += values
            with open(self.detections_path + video.name + '.csv', 'w+') as f:
                writer = csv.writer(f)
                writer.writerows(results)
        if plot:
            import matplotlib.pyplot as plt
            plt.hist(self.all_values, bins=100, color='g')
            plt.show()

    def write_true_positives(self, video):
        self.frame_wise_dets = {}
        with open(vid_annotations_path + '{}.csv'.format(video.name)) as f:
            # write true positives by reading from ground truths
            lines = sorted(list(csv.reader(f)), key=lambda x: int(x[0]))
            self.num_tp = len(lines)
            conf_tp = np.random.normal(size=self.num_tp)
            if max(conf_tp) != min(conf_tp):
                conf_tp = (conf_tp - min(conf_tp)) / (max(conf_tp) - min(conf_tp))
            else:
                conf_tp = np.ones(shape=(self.num_tp,)) / 2
            for idx, line in enumerate(lines):
                fid = int(line[0])
                det = BoundingBox(video.name, fid, self.dataset.class_name_from_id(line[2]),
                                  [int(k) for k in line[3:7]], round(conf_tp[idx], 4), object_id=int(line[1]))
                self.frame_wise_dets[fid] = self.frame_wise_dets.get(fid, []) + [det]

    def write_false_positives(self, video):
        video_size = len(video)
        # write probabilistically generated false positives
        num_fp = int(self.num_tp / self.precision) - self.num_tp
        conf_fp = np.random.lognormal(sigma=0.5, size=num_fp)
        if max(conf_fp) != min(conf_fp):
            conf_fp = (conf_fp - min(conf_fp)) / (max(conf_fp) - min(conf_fp))
        else:
            conf_fp = np.ones(shape=(num_fp,)) / 2

        w, h = video.get_width_height()
        for idx in range(num_fp):
            frame_id = int(np.floor(video_size * np.random.random()))
            cls_id = int(np.ceil((len(self.dataset.classes)) * np.random.random()))

            bbox_float = np.random.random(4)
            bbox = [int(w * min(bbox_float[0], bbox_float[2])), int(h * min(bbox_float[1], bbox_float[3])),
                    int(w * max(bbox_float[0], bbox_float[2])), int(h * max(bbox_float[1], bbox_float[3]))]
            if bbox[1] == bbox[3]:
                bbox[3] += 1
            if bbox[0] == bbox[2]:
                bbox[2] += 1
            det = BoundingBox(video.name, frame_id, self.dataset.classes[cls_id - 1], bbox, round(conf_fp[idx]),
                              object_id=0)
            self.frame_wise_dets[frame_id] = self.frame_wise_dets.get(frame_id, []) + [det]

    def get_values(self, video):
        raise NotImplementedError("Inheriting class should have this abstract method implemented.")

    def condition(self, x):
        raise NotImplementedError("Inheriting class should have this abstract method implemented.")

    def parse_csv(self, p_high=None, p_low=None):
        detector = self.name if self.tracker_name is None else '{}_{}'.format(self.tracker_name, self.name)
        self.detections = {}
        for video_name in self.video_names:
            file_path = '{}/{}/{}.csv'.format(synthetic_detections_path, detector, video_name)
            if not os.path.exists(file_path):
                if self.tracker_name:
                    track_videos(self.name, self.tracker_name, self.dataset, self.classes)
                    print('Tracking results not present in {}.'.format(file_path),
                          'Running tracker now (with default parameters).')
                self.write_detections()

            temp_dets = {}
            num_total, num_low = 0, 0
            with open(file_path) as f:
                r = np.random.random()
                for line in csv.reader(f):
                    if len(line) < (7 if self.tracker_name else 6):
                        line = line[0].split(' ')
                    if len(line) > (7 if self.tracker_name else 6):
                        if self.tracker_name is not None:
                            frame_id, cls, confidence, obj_id = int(line[0]), line[2], float(line[3]), int(line[1])
                            bbox = [int(i) for i in line[4:8]]
                            values = line[8:]
                        else:
                            frame_id, cls, confidence, obj_id = int(line[0]), line[1], float(line[2]), None
                            bbox = [int(i) for i in line[3:7]]
                            values = line[7:]
                        num_total += 1
                        if self.condition(values):
                            p_box = p_high
                        else:
                            num_low += 1
                            p_box = p_low
                        if r >= p_box:
                            continue

                        if self.classes is None or self.dataset.class_name_from_id(cls) in self.classes:
                            thresh_cls = self.conf_thresholds[cls]
                            if confidence >= thresh_cls:
                                temp_dets[frame_id] = temp_dets.get(frame_id, []) + [
                                    BoundingBox(video_name, frame_id, self.dataset.class_name_from_id(cls), bbox,
                                                confidence, obj_id)]

            self.detections[video_name] = [Frame(frame_id, video_name, temp_dets.get(frame_id, [])) for frame_id in
                                           range(len(self.dataset.videos[video_name]))]

            if self.t_nms_window > 1:
                self.suppress()

        # print(num_low, num_total-num_low, round(100*num_low/num_total,2))
        return self.detections


class BiasedDetectorSize(SyntheticDetector):

    def __init__(self, dataset, threshold=0.04, precision=0.5):
        super().__init__('SIZE', dataset, threshold, precision=precision)
        self.detections_path = '{}/{}/'.format(synthetic_detections_path, self.name)

    def condition(self, x):
        return float(x[0]) > self.threshold

    def get_values(self, video):
        w, h = video.get_width_height()
        results, values = [], []
        for fid in self.frame_wise_dets.keys():
            for dets in self.frame_wise_dets[fid]:
                xmin, ymin, xmax, ymax = dets.box
                size = ((xmax - xmin + 1) * (ymax - ymin + 1)) / (w * h)
                values.append(size)
                dets.object_id = None
                results.append(list(dets.to_list()) + [np.round(size, 4)])
        return results, values


class BiasedDetectorSpeed(SyntheticDetector):
    name = 'SPEED'

    def __init__(self, dataset, threshold=40, precision=0.5):
        super().__init__(self.name, dataset, threshold, precision=precision)

    def condition(self, x):
        return float(x[0]) > self.threshold

    def get_values(self, video):
        video.load_sets()
        speeds = {}
        for obj_id, obj in video.gt_objects.items():
            speeds[int(obj_id)] = np.mean([len(t) for t in obj.sets])
        results, values = [], []
        for fid in self.frame_wise_dets.keys():
            for dets in self.frame_wise_dets[fid]:
                values.append(speeds[dets.object_id])
                dets.object_id = None
                results.append(dets.to_list() + [np.round(values[-1])])
        return results, values


class BiasedDetectorIllumination(SyntheticDetector):

    def __init__(self, dataset, threshold=90, precision=0.5):
        super().__init__('ILLUMINATION', dataset, threshold, precision=precision)

    def condition(self, x):
        return float(x[0]) > self.threshold

    def get_values(self, video):
        import cv2
        results, values = [], []
        frames_list = sorted(glob.glob('{}/{}/*.JPEG'.format(vid_image_path, video.name)))
        for frame in frames_list:
            img = cv2.imread(frame)
            fid = int(Path(frame).stem)
            if fid in self.frame_wise_dets.keys():
                for dets in self.frame_wise_dets[fid]:
                    xmin, ymin, xmax, ymax = dets.box
                    values.append(np.mean(img[ymin:ymax, xmin:xmax]))
                    dets.object_id = None
                    results.append(dets.to_list() + [np.round(values[-1])])
        return results, values


class BiasedDetectorHue(SyntheticDetector):
    name = 'HUE'

    def __init__(self, dataset, threshold=300, precision=0.5):
        super().__init__('HUE', dataset, threshold, precision=precision)

    def condition(self, x):
        return float(x[0]) < self.threshold

    def get_values(self, video):
        import cv2
        results, values = [], []
        frames_list = sorted(glob.glob('{}/{}/*.JPEG'.format(vid_image_path, video.name)))
        for idx, frame in enumerate(frames_list):
            img = cv2.imread(frame)
            # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fid = int(Path(frame).stem)
            if fid in self.frame_wise_dets.keys():
                for dets in self.frame_wise_dets[fid]:
                    xmin, ymin, xmax, ymax = dets.box
                    box_hues = []
                    for y in range(ymin, ymax, 1 + (ymax - ymin) // 60):
                        for x in range(xmin, xmax, 1 + (xmax - xmin) // 60):
                            rgb = img[y, x]
                            max_rgb = np.argmax(rgb)
                            min_rgb = np.argmin(rgb)
                            if rgb[max_rgb] == rgb[min_rgb]:
                                hue = 2 * max_rgb
                            elif max_rgb == 0:
                                hue = (rgb[1] / 255 - rgb[2] / 255) / (rgb[max_rgb] / 255 - rgb[min_rgb] / 255)
                            elif max_rgb == 1:
                                hue = 2 + (rgb[2] / 255 - rgb[0] / 255) / (rgb[max_rgb] / 255 - rgb[min_rgb] / 255)
                            else:
                                hue = 4 + (rgb[0] / 255 - rgb[1] / 255) / (rgb[max_rgb] / 255 - rgb[min_rgb] / 255)
                            hue = 360 + hue * 60 if hue < 0 else 60 * hue
                            box_hues.append(hue // 10)
                    values.append(10 * max(set(box_hues), key=box_hues.count))
                    dets.object_id = None
                    results.append(dets.to_list() + [np.round(values[-1])])
        return results, values


class BiasedDetectorContrast(SyntheticDetector):
    def __init__(self, dataset, threshold=0.42, precision=0.5):
        super().__init__('CONTRAST', dataset, threshold, precision=precision)

    def condition(self, x):
        return float(x[0]) > self.threshold

    def get_values(self, video):
        import cv2
        results, values = [], []
        frames_list = sorted(glob.glob('{}/{}/*.JPEG'.format(vid_image_path, video.name)))
        for frame in frames_list:
            img = cv2.imread(frame)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            fid = int(Path(frame).stem)
            if fid in self.frame_wise_dets.keys():
                for dets in self.frame_wise_dets[fid]:
                    xmin, ymin, xmax, ymax = dets.box
                    h_box, w_box = ymax - ymin, xmax - xmin
                    xmin_out, ymin_out = max(0, xmin - w_box // 4), max(0, ymin - h_box // 4)
                    xmax_out, ymax_out = min(img.shape[1], xmax + w_box // 4), min(img.shape[0], ymax + h_box // 4)
                    r_value = abs(
                        np.mean(r[ymin:ymax, xmin:xmax]) / np.mean(r[ymin_out:ymax_out, xmin_out:xmax_out]) - 1)
                    g_value = abs(
                        np.mean(g[ymin:ymax, xmin:xmax]) / np.mean(r[ymin_out:ymax_out, xmin_out:xmax_out]) - 1)
                    b_value = abs(
                        np.mean(b[ymin:ymax, xmin:xmax]) / np.mean(r[ymin_out:ymax_out, xmin_out:xmax_out]) - 1)
                    values.append(r_value + g_value + b_value)
                    dets.object_id = None
                    results.append(dets.to_list() + [round(values[-1], 3)])
        return results, values


def write_scores(detectors, dataset, metrics, p_lows=None, p_highs=None, overwrite=False, verbose=False):
    if p_highs is None:
        p_highs = p_highs_default
    if p_lows is None:
        p_lows = p_lows_default
    os.makedirs(synthetic_detector_results_path, exist_ok=True)
    for detector in detectors:
        if verbose:
            print(detector.name)
        file_path = "{}/all_{}_bias.json".format(synthetic_detector_results_path, detector.name)
        metric_values = json.load(open(file_path)) if os.path.exists(file_path) else {}
        already_evaluated = [] if overwrite else list(metric_values.keys())
        print(already_evaluated)
        for p_h, p_low in zip(p_highs, p_lows):
            if verbose:
                print("Computing for the detection probabilities:", p_h, p_low)

            detector.parse_csv(p_h, p_low)
            if 'AD' in metrics and 'AD' not in already_evaluated:
                if verbose:
                    print('Evaluating AD')
                delays, m_d = get_mD(dataset.video_names, detector.name, [0], 'clipmax', detector.condition,
                                     p_h, p_low, verbose=False)
                m_d = 100 / (1 + np.exp(-m_d))
                metric_values['AD'] = metric_values.get('AD', []) + [100 * m_d]

            if 'LNmAP' in metrics and 'LNmAP' not in already_evaluated:
                if verbose:
                    print('Evaluating LNmAP')
                vmap = LNmAP(dataset, detector, evaluate=LNmAP.voc)
                metric_values['LNmAP'] = metric_values.get('LNmAP', []) + [100 * vmap.score]

            if 'KFmAP' in metrics and 'KFmAP' not in already_evaluated:
                if verbose:
                    print('Evaluating KFmAP')
                vmap = KFmAP(dataset, detector, evaluate=KFmAP.voc)
                metric_values['KFmAP'] = metric_values.get('KFmAP', []) + [100 * vmap.score]

            if 'mAP' in metrics and 'mAP' not in already_evaluated:
                if verbose:
                    print('Evaluating mAP')
                vmap = MAP(dataset, detector, evaluate=MAP.voc)
                metric_values['mAP'] = metric_values.get('mAP', []) + [100 * vmap.score]

            if 'VmAP' in metrics and 'VmAP' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP')
                vmap = VmAP(dataset, detector, evaluate=VmAP.voc)
                metric_values['VmAP'] = metric_values.get('VmAP', []) + [100 * vmap.score]

            if 'VmAP_5' in metrics and 'VmAP_5' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_5')
                tmap = TVmAP(dataset.name, detector, 0.05, evaluate=TVmAP.voc)
                metric_values['VmAP_5'] = metric_values.get('VmAP_5', []) + [100 * tmap.score]

            if 'VmAP_10' in metrics and 'VmAP_10' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_10')
                tmap = TVmAP(dataset, detector, 0.1, evaluate=TVmAP.voc)
                metric_values['VmAP_10'] = metric_values.get('VmAP_10', []) + [100 * tmap.score]

            if 'VmAP_20' in metrics and 'VmAP_20' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_20')
                tmap = TVmAP(dataset, detector, 0.2, evaluate=TVmAP.voc)
                metric_values['VmAP_20'] = metric_values.get('VmAP_20', []) + [100 * tmap.score]

            if 'VmAP_50' in metrics and 'VmAP_50' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_50')
                tmap = TVmAP(dataset, detector, 0.5, evaluate=TVmAP.voc)
                metric_values['VmAP_50'] = metric_values.get('VmAP_50', []) + [100 * tmap.score]

            if 'VmAP_a' in metrics and 'VmAP_a' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_a')
                dataset = Dataset(dataset.name, True, class_names=dataset.classes, video_names=dataset.video_names,
                                  num_objects=dataset.num_objects, set_criteria=set_formation_app, gamma=dataset.gamma)
                vmap = VmAP(dataset, detector, evaluate=VmAP.voc)
                metric_values['VmAP_a'] = metric_values.get('VmAP_a', []) + [100 * vmap.score]

            if 'VmAP_al' in metrics and 'VmAP_al' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_al')
                dataset = Dataset(dataset.name, True, class_names=dataset.classes, video_names=dataset.video_names,
                                  num_objects=dataset.num_objects, set_criteria=set_formation_app_loc,
                                  gamma=dataset.gamma)
                vmap = VmAP(dataset, detector, evaluate=VmAP.voc)
                metric_values['VmAP_al'] = metric_values.get('VmAP_al', []) + [100 * vmap.score]

            if 'VmAP_t' in metrics and 'VmAP_t' not in already_evaluated:
                if verbose:
                    print('Evaluating VmAP_t')
                dataset = Dataset(dataset.name, True, class_names=dataset.classes, video_names=dataset.video_names,
                                  num_objects=dataset.num_objects, set_criteria=set_formation_time, gamma=dataset.gamma)
                vmap = VmAP(dataset, detector, evaluate=VmAP.voc)
                metric_values['VmAP_t'] = metric_values.get('VmAP_t', []) + [100 * vmap.score]

            if verbose:
                print('Caching...')

            with open(file_path, 'w+') as f:
                json.dump(metric_values, fp=f, indent=4)


def plot_synthetic(detectors, metrics, p_lows=None, p_highs=None, save_path=None, show_plot=False):
    if p_lows is None:
        p_lows = p_lows_default
    if p_highs is None:
        p_highs = p_highs_default
    import matplotlib.pyplot as plt

    for detector in detectors:
        file_path = "{}/all_{}_bias.json".format(synthetic_detector_results_path, detector.name)
        metric_values = json.load(open(file_path)) if os.path.exists(file_path) else {}
        metrics = [m for m in metrics if m in list(metric_values.keys())]
        x = range(len(metric_values[metrics[0]]))
        for m in metrics:
            plt.plot(x, metric_values[m])
        plt.legend(metrics)
        plt.xticks(x, list(zip(p_lows, p_highs)))
        plt.yticks(list(range(0, 105, 20)))
        plt.ylim(-5, 105)
        if save_path is None:
            save_path = '{}/{}_{}.pdf'.format(synthetic_detector_results_path, save_path, detector.name)
        plt.savefig(save_path, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.cla()


def vis_contrast():
    from viz import draw_box_on_image
    import cv2
    video_names = sorted([Path(k).stem for k in glob.glob('results_VIDT/gt_sets/*')])[:]
    output_path = 'visualisations/contrast/'
    os.makedirs(output_path + 'low/', exist_ok=True)
    os.makedirs(output_path + 'mid/', exist_ok=True)
    os.makedirs(output_path + 'high/', exist_ok=True)
    for video_name in video_names:
        detections_path = 'alert_generator/detector/tracked_detections/ARTIFICIAL/CONTRAST/'
        file_path = detections_path + video_name + '.csv'
        with open(file_path) as f:
            for line in csv.reader(f):
                ill_val = sum([float(k) for k in line[7:]])
                if 0.38 > ill_val > 0.35:
                    img_path = output_path + 'mid/{}_{}.JPEG'.format(video_name, line[0].zfill(6))
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                    else:
                        img = cv2.imread('results_VIDT/gt_sets/{}/{}.JPEG'.format(video_name, line[0].zfill(6)))
                    bbox = [int(k) for k in line[3:7]]
                    text = "{:.3f}".format(ill_val)
                    img = draw_box_on_image(img, bbox, text, thickness=2)
                    cv2.imwrite(img_path, img)
                    print(video_name, line, ill_val)


def vis_ill():
    import cv2
    video_names = sorted([Path(k).stem for k in glob.glob('results_VIDT/gt_sets/*')])[:]
    output_path = 'visualisations/illumination/'
    os.makedirs(output_path + 'low/', exist_ok=True)
    os.makedirs(output_path + 'mid/', exist_ok=True)
    os.makedirs(output_path + 'high/', exist_ok=True)
    for video_name in video_names:
        detections_path = 'alert_generator/detector/tracked_detections/ARTIFICIAL/ILL/'
        file_path = detections_path + video_name + '.csv'
        with open(file_path) as f:
            for line in csv.reader(f):
                ill_val = sum([float(k) for k in line[7:]])
                if 80 < ill_val < 90:
                    img_path = output_path + 'mid/{}_{}.JPEG'.format(video_name, line[0].zfill(6))
                elif ill_val < 30:
                    img_path = output_path + 'low/{}_{}.JPEG'.format(video_name, line[0].zfill(6))
                elif ill_val > 150:
                    img_path = output_path + 'high/{}_{}.JPEG'.format(video_name, line[0].zfill(6))
                else:
                    continue
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                else:
                    img = cv2.imread('results_VIDT/gt_sets/{}/{}.JPEG'.format(video_name, line[0].zfill(6)))
                bbox = [int(k) for k in line[3:7]]
                text = "{:.3f}".format(ill_val)
                img = draw_box_on_image(img, bbox, text, thickness=2)
                cv2.imwrite(img_path, img)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--metrics', nargs='+', default=['VmAP', 'mAP'],
                        help='list of metrics to be evaluated. Available metrics are: VmAP, mAP, LNmAP, KFmAP, AD, '
                             'VmAP_n (n=5, 10, 15, 20), VmAP_a, VmAP_al, VmAP_t')
    parser.add_argument('--biases', nargs='+', default=['SIZE', 'SPEED', 'ILLUMINATION', 'HUE', 'CONTRAST'],
                        help='biases over which to evaluate the metrics. Available biases are: SIZE, SPEED, HUE, '
                             'CONTRAST, ILLUMINATION')
    parser.add_argument('--precision', default=0.5, help='frame level precision of the synthetic biased detector.')
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--plot', action='store_true', help='whether to save the plot or not.')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite currently cached scores.')
    parser.add_argument('--plot_file_name', default='bias_sensitivity', help='name of the plot to be saved.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert len(args.metrics), "At least one metric required."
    np.random.seed(5)
    metrics = args.metrics
    precision = args.precision
    verbose = args.verbose
    detectors = []
    dataset = VID(pre_load=True)
    if verbose:
        print('Dataset loaded.')

    if 'SIZE' in args.biases:
        detectors.append(BiasedDetectorSize(dataset, precision=precision))
    if 'SPEED' in args.biases:
        detectors.append(BiasedDetectorSpeed(dataset, precision=precision))
    if 'HUE' in args.biases:
        detectors.append(BiasedDetectorHue(dataset, precision=precision))
    if 'CONTRAST' in args.biases:
        detectors.append(BiasedDetectorContrast(dataset, precision=precision))
    if 'ILLUMINATION' in args.biases:
        detectors.append(BiasedDetectorIllumination(dataset, precision=precision))

    assert len(detectors), "at least one bias required. Available biases are: SIZE, SPEED, HUE, CONTRAST, ILLUMINATION"
    if verbose:
        print('Detectors loaded.')

    write_scores(detectors, dataset, args.metrics, p_lows_default, p_highs_default, args.overwrite, verbose=verbose)
    if args.plot:
        plot_synthetic(detectors, args.metrics, p_lows_default, p_highs_default, args.plot_file_name)
