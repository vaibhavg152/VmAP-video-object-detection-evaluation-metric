import os

import numpy as np

from constants import pr_plot_path, detector_names
from datasets.dataset import Dataset
from metrics.AP_utils import voc_ap
from metrics.Metric import Metric
from models.detector import Detector


class MAP(Metric):
    name = 'mAP'

    def __init__(self, dataset, detector, evaluate, verbose=False):
        super().__init__(self.name, dataset, detector, evaluate, verbose)

    def get_precision_recall(self, cls_name, thresh=None):
        num_det = len(self.predictions[cls_name])
        num_obj = self.dataset.num_frames_per_class.get(cls_name, 0)
#        print("Num detections", cls_name, num_det)

        assert num_obj, "No GT objects of class {} in given videos.".format(cls_name)
        if not num_det:
            print("INFO: No detections found for {}.".format(cls_name))
            return [0], [0], 0, 0, num_obj
        tp, fp = np.zeros(num_det), np.zeros(num_det)
        num_tp, num_fp = 0, 0
        for det_idx, detection in enumerate(self.predictions[cls_name]):
            vid, fid = detection.video_name, detection.frame_id
            # match boxes
            matched, obj_id = self.dataset.videos[vid].match_frames(detection, self.matched_objs[vid].get(fid, []),
                                                                    iou_thresh=thresh)
            if not matched:
                self.frame_wise_fp[vid][fid] = 1 + self.frame_wise_fp[vid].get(fid, 0)
                num_fp += 1
            else:
                num_tp += 1
                self.matched_objs[vid][fid] = self.matched_objs[vid].get(fid, []) + [obj_id]
            fp[det_idx] = num_fp
            tp[det_idx] = num_tp

        rec = tp / num_obj if num_obj > 0 else np.zeros(num_det)
        prec = tp / (tp + fp) if num_det > 0 else np.zeros(num_det)
        return rec, prec, tp[-1], fp[-1], num_obj


def plot_pr_curves(detector, dataset):
    metric = MAP(dataset, detector, evaluate=MAP.voc)
    metric.pr_plots()


def plot_pr_all(dataset=None, metric="VmAP"):
    import matplotlib.pyplot as plt

    detectors = detector_names[dataset.name]
    detectors.remove("GT")
    for cls in dataset.class_names():
        results_path = "{}/{}/{}/all_detectors/".format(pr_plot_path, dataset.name, MAP.name)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        for detector in detectors:
            vmap = MAP(dataset, Detector(dataset, detector), evaluate=MAP.voc)
            rec, prec, tp, fp, num_obj = vmap.get_precision_recall(cls)
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


def mAP(ground_truths, detector, verbose=False):
    from video_elements.video_elements import Video
    if isinstance(ground_truths, Video):
        ground_truths = Dataset('VID', pre_load=False, video_names=[ground_truths.name])
    map_ = MAP(ground_truths, detector, evaluate=MAP.voc, verbose=verbose)
    return map_.score, map_.class_wise_scores


def dump_class_wise_scores(dataset):
    for detector in detector_names[dataset.name]:
        vmap = MAP(dataset, Detector(dataset, detector), evaluate=MAP.voc)
        vmap.generate_detections()
        vmap.evaluate_voc(True, True)


if __name__ == '__main__':
    dump_class_wise_scores(Dataset('COCO'))
