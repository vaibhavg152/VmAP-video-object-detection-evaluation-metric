import numpy as np

from constants import matching_ln
from metrics.Metric import Metric


class LNmAP(Metric):
    name = 'LNmAP'

    def __init__(self, dataset, detector, evaluate):
        super().__init__(self.name, dataset, detector, evaluate)

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
                                                                         thresh, match_criteria=matching_ln)
            if is_fp:
                self.frame_wise_fp[vid][fid] = 1 + self.frame_wise_fp[vid].get(fid, 0)
                num_fp += is_fp
            if is_tp:
                num_tp += is_tp
                self.matched_objs[vid][fid] = self.matched_objs[vid].get(fid, []) + [matched_obj]
                self.matched_sets[vid][matched_set] = self.matched_sets[vid].get(matched_set, 0) + 1
                # print(fid, matched_obj, matched_set, self.matched_objs[vid][fid])
            fp[det_idx] = num_fp
            tp[det_idx] = num_tp

        rec = tp / num_obj if num_obj > 0 else np.zeros(num_det)
        prec = tp / (tp + fp) if num_det > 0 else np.zeros(num_det)
        return rec, prec, tp[-1], fp[-1], num_obj


def ln_map(ground_truths, detector):
    metric = LNmAP(ground_truths, detector, LNmAP.voc)
    return metric.score
