import numpy as np

from metrics.Metric import Metric


class APAll(Metric):
    name = 'AP_all_classes'

    def __init__(self, dataset, detector, evaluate):
        super().__init__(self.name, dataset, detector, evaluate)
        self.all_classes_scores = {}

    def get_precision_recall(self, cls_name, thresh=None):
        try:
            tp = self.class_wise_scores['TP']
            fp = self.class_wise_scores['FP']
            num_obj = self.class_wise_scores['FN'] + tp
            return self.all_classes_scores['Recall'], self.all_classes_scores['Precision'], tp, fp, num_obj
        except KeyError:
            detections_all = []
            for cls_name in self.class_names:
                for detection in self.predictions[cls_name]:
                    detection.class_name = cls_name
                    detections_all.append(detection)
            detections_all.sort(key=lambda x: x.confidence, reverse=True)

            # find true positives and false positives
            matched_objs = dict([(vid, {}) for vid in self.video_names])
            num_det = len(detections_all)
            tp, fp = np.zeros(num_det), np.zeros(num_det)
            num_tp, num_fp = 0, 0
            num_obj = self.dataset.num_frames_per_class.get(cls_name, 0)
            for det_idx, detection in enumerate(detections_all):
                vid, fid = detection.video_names, detection.frame_id
                is_tp, is_fp, matched_obj_id = self.dataset[vid].match_objects(detection, matched_objs[vid][fid],
                                                                               0, iou_thresh=thresh)
                if is_tp:
                    num_tp += is_tp
                    matched_objs[vid][fid] = matched_objs[vid].get(fid, []) + [matched_obj_id]
                if is_fp:
                    num_fp += is_fp
                    self.frame_wise_fp[vid][fid] = 1 + self.frame_wise_fp[vid].get(fid, 0)
                fp[det_idx] = num_fp
                tp[det_idx] = num_tp

            rec = tp / num_obj if num_obj > 0 else np.zeros(num_det)
            prec = tp / (tp + fp) if num_det > 0 else np.zeros(num_det)
            self.all_classes_scores = {'TP': tp[-1], 'FP': fp[-1], 'FN': num_obj - tp[-1],
                                       'NumDetections': tp[-1] + fp[-1], 'Recall': rec, 'Precision': prec}

            return rec, prec, tp[-1], fp[-1], num_obj


def ap_all(ground_truths, detector, verbose=True):
    metric = APAll(ground_truths, detector, evaluate=APAll.voc)
    return metric.evaluate_voc(False)
