import numpy as np

from metrics.Metric import Metric


class TVmAP(Metric):
    name = 'VmAP_t'

    def __init__(self, dataset, detector, match_threshold, evaluate):
        self.match_threshold = match_threshold
        super().__init__(self.name, dataset, detector, evaluate)

    def get_precision_recall(self, cls_name, thresh=None):
        num_det = len(self.predictions[cls_name])
        num_obj = self.dataset.num_frames_per_class.get(cls_name, 0)
        tp, fp = np.zeros(num_det), np.zeros(num_det)
        num_tp, num_fp = 0, 0
        if num_det == 0:
            print("No results found for class: ", cls_name)
            return [0], [0], 0, 0, num_obj

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
                match_before = self.matched_sets[vid].get(matched_set, 0)
                match_req = self.match_threshold * len(video.gt_objects[matched_obj].get_set_with_frame(fid))
                if match_before <= match_req < match_before+1:
                    num_tp += 1
                self.matched_sets[vid][matched_set] = match_before + 1
            fp[det_idx] = num_fp
            tp[det_idx] = num_tp
        rec = tp / num_obj if num_obj > 0 else np.zeros(num_det)
        prec = tp / (tp + fp) if num_det > 0 else np.zeros(num_det)
        return rec, prec, tp[-1], fp[-1], num_obj


def t_vmap(ground_truths, detector, match_threshold, verbose=True):
    metric = TVmAP(ground_truths, detector, match_threshold, evaluate=None)
    return metric.evaluate_voc(False)
