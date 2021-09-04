import os

import numpy as np

from constants import pr_plot_path, detector_names
from datasets.dataset import COCO, VID
from metrics.AP_utils import voc_ap
from metrics.Metric import Metric
from models.detector import Detector


class VmAP(Metric):
    name = 'VmAP'

    def __init__(self, dataset, detector, evaluate, verbose=False):
        super().__init__(self.name, dataset, detector, evaluate, verbose)

    def get_precision_recall(self, cls_name, thresh=None):
        num_det = len(self.predictions[cls_name])
        num_obj = self.dataset.num_sets_per_class.get(cls_name, 0)
        assert num_obj, "No GT objects of class {} in given videos.".format(cls_name)
        if not num_det:
            print("INFO: No detections found for {}.".format(cls_name))
            return [0], [0], 0, 0, num_obj
        tp, fp = np.zeros(num_det), np.zeros(num_det)
        num_tp, num_fp = 0, 0
        for det_idx, detection in enumerate(self.predictions[cls_name]):
            vid, fid = detection.video_name, detection.frame_id
            video = self.dataset.videos[vid]
            # match boxes
            is_tp, is_fp, matched_obj, matched_set = video.match_objects(detection, self.matched_objs[vid].get(fid, []),
                                                                         thresh)
            if is_fp:
                self.frame_wise_fp[vid][fid] = 1 + self.frame_wise_fp[vid].get(fid, 0)
                num_fp += is_fp
            if is_tp:
                self.matched_objs[vid][fid] = self.matched_objs[vid].get(fid, []) + [matched_obj]
                if matched_obj not in self.matched_sets[vid]:
                    self.matched_sets[vid][matched_obj] = {}
                self.matched_sets[vid][matched_obj][matched_set] = self.matched_sets[vid][matched_obj].get(matched_set,
                                                                                                           0) + 1
                num_tp += int(self.matched_sets[vid][matched_obj][matched_set] == 1)
            fp[det_idx] = num_fp
            tp[det_idx] = num_tp

        rec = tp / num_obj if num_obj > 0 else np.zeros(num_det)
        prec = tp / (tp + fp) if num_det > 0 else np.zeros(num_det)
        return rec, prec, tp[-1], fp[-1], num_obj


def vmap(ground_truths, detector, verbose=False):
    from video_elements.video_elements import Video
    if isinstance(ground_truths, Video):
        ground_truths = VID(pre_load=True, video_names=[ground_truths.name], set_criteria=ground_truths.set_criteria,
                            gamma=ground_truths.gamma, class_names=ground_truths.class_names)
    metric = VmAP(ground_truths, detector, evaluate=VmAP.voc, verbose=verbose)
    return metric.score, metric.class_wise_scores


def plot_pr_curves(detector, dataset, t_nms_window=1):
    metric = VmAP(dataset, detector, evaluate=VmAP.voc)
    if t_nms_window > 1:
        results_path = pr_plot_path + '{}/{}/{}_nms/'.format(dataset.name, metric.name, detector.name)
    else:
        results_path = None
    metric.pr_plots(results_path=results_path)


def plot_pr_all(dataset=None, metric="VmAP"):
    import matplotlib.pyplot as plt

    detectors = detector_names[dataset.name]
    detectors.remove("GT")
    for cls in dataset.class_names():
        results_path = "{}/{}/{}/all_detectors/".format(pr_plot_path, dataset.name, VmAP.name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        for detector in detectors:
            vmap_scores = VmAP(dataset, Detector(dataset, detector), evaluate=VmAP.voc)
            rec, prec, tp, fp, num_obj = vmap_scores.get_precision_recall(cls)
            _, m_rec, m_prec = voc_ap(rec, prec)
            plt.plot(m_rec, m_prec)

        plt.xlabel("V-Recall")
        plt.ylabel("V-Precision")
        plt.legend(detectors, loc=3)
        plt.title(metric + ' ' + cls)
        file_name = results_path + cls + ".png"
        print(file_name)
        plt.savefig(file_name)
        # plt.show()
        plt.cla()


def dump_class_wise_scores(dataset):
    for detector in detector_names[dataset.name]:
        vmap_scores = VmAP(dataset, Detector(dataset, detector), VmAP.voc)
        vmap_scores.evaluate_voc(True, True)


if __name__ == '__main__':
    dump_class_wise_scores(COCO(pre_load=True))
