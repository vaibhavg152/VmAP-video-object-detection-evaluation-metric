import json
import os

from constants import op_cache_path, op_scores_cache_path, pr_plot_path, eval_cache_path, MIN_OVERLAP_COCO
from datasets.dataset import Dataset
from metrics.AP_utils import voc_ap
from models.utils import f1
from video_elements.video_elements import Video


class Metric:
    voc = 'VOC'
    ms_coco = 'MS_COCO'

    def __init__(self, name, dataset, detector, evaluate, verbose=False):
        self.verbose = verbose
        self.recalls = {}
        self.precisions = {}
        if isinstance(dataset, Video):
            dataset = Dataset(dataset.dataset_name, video_names=[dataset.name], class_names=dataset.class_names,
                              set_criteria=dataset.set_criteria, gamma=dataset.gamma)
        self.name = name
        self.dataset = dataset
        self.detector = detector
        self.tracker = self.detector.tracker_name
        self.video_names = [vid for vid in self.detector.video_names if vid in self.dataset.video_names]
        self.class_names = [cls for cls in self.detector.classes if cls in self.dataset.classes]
        self.frame_wise_fp = dict([(vid, {}) for vid in self.video_names])
        self.predictions = None
        self.score = None
        self.class_wise_scores = None
        self.matched_objs = dict([(vid, {}) for vid in self.video_names])
        self.matched_sets = dict([(vid, {}) for vid in self.video_names])
        if evaluate:
            self.generate_detections()
            if evaluate == self.voc:
                self.evaluate_voc()
            elif evaluate == self.ms_coco:
                self.evaluate_coco()

    def generate_detections(self):
        if self.detector.detections is None:    # if detector was loaded with lazy-load
            self.detector.parse_csv()

        self.predictions = {}
        for cls_name in self.class_names:
            self.predictions[cls_name] = []
        for vid_name in self.video_names:
            for idx, frame in enumerate(self.detector.detections[vid_name]):
                for det in frame.bboxes:
                    if det.class_name in self.class_names:
                        self.predictions[det.class_name] += [det]

        # sorting detections according to confidence values
        for cls_name, predict in self.predictions.items():
            self.predictions[cls_name] = sorted(predict, key=lambda x: x.confidence, reverse=True)
        # total_dets = sum([len(dets) for dets in self.predictions.values()])
        # print("Total detections:", total_dets)

    def set_detections(self, detections):
        self.predictions = detections

    def evaluate_coco(self, cache=True, overwrite_cache=False):
        self.score = 0
        self.class_wise_scores = {}
        self.matched_objs = dict([(vid, {}) for vid in self.video_names])
        if self.predictions is None:
            self.generate_detections()
        for cls_name in self.class_names:
            aps = []
            self.class_wise_scores[cls_name] = {}
            for thresh in MIN_OVERLAP_COCO:
                rec, prec, tp, fp, num_obj = self.get_precision_recall(cls_name, thresh)
                aps.append(voc_ap(list(rec), list(prec))[0])
                self.class_wise_scores[cls_name][thresh] = {'TP': tp, 'FP': fp, 'FN': num_obj - tp, 'AP': 100 * aps[-1],
                                                            'Recall': 100 * rec[-1], 'Precision': 100 * prec[-1],
                                                            'F-1': 100 * f1(rec[-1], prec[-1])}
            ap_coco = sum(aps) / len(aps)
            self.class_wise_scores[cls_name]['AP'] = 100 * ap_coco
            self.score += ap_coco
            if self.verbose:
                print("CLASS:", cls_name, "AP: {:.2f}".format(100 * ap_coco), end='\t')
        self.score /= len(self.class_names)
        if cache:
            results_path = '{}/{}/{}/MS_COCO/'.format(eval_cache_path, self.name, self.dataset.name)
            import os
            os.makedirs(results_path, exist_ok=True)
            if not os.path.exists('{}/{}.json'.format(results_path, self.detector)) or overwrite_cache:
                json.dump(self.class_wise_scores, open('{}/{}.json'.format(results_path, self.detector), 'w'), indent=4)

        return self.score, self.class_wise_scores

    def evaluate_voc(self, cache=True, overwrite_cache=False, save_plots=False, cache_path=None):
        self.score = 0
        self.class_wise_scores = {}
        self.matched_objs = dict([(vid, {}) for vid in self.video_names])
        if self.predictions is None:
            self.generate_detections()
        for cls_name in self.class_names:
            rec, prec, tp, fp, num_obj = self.get_precision_recall(cls_name)
            ap, m_rec, m_prec = voc_ap(list(rec), list(prec))
            self.class_wise_scores[cls_name] = {'TP': tp, 'FP': fp, 'FN': num_obj - tp, 'AP': 100 * ap,
                                                'Recall': 100 * rec[-1], 'Precision': 100 * prec[-1],
                                                'F1': f1(rec[-1], prec[-1])}
            self.recalls[cls_name] = rec
            self.precisions[cls_name] = prec
            self.score += ap
            if self.verbose:
                print("CLASS:", cls_name, "AP: {:.2f}".format(100 * ap), end='\t')
                print("precision:\t {:.2f}%".format(100 * prec[-1]), end='\t')
                print("recall:\t {:.2f}%".format(100 * rec[-1]))
            if save_plots:
                filename = '{}/{}/{}/{}/{}.png'.format(pr_plot_path, self.dataset.name, self.name,
                                                       self.detector, cls_name)
                self.draw_pr_plot(m_rec, m_prec, filename, "{} AP={}".format(self.detector, ap), show=False)
        self.score /= len(self.class_names)
        if cache:
            results_path = cache_path if cache_path else '{}/{}/{}/VOC/'.format(eval_cache_path,
                                                                                self.name, self.dataset.name)
            import os
            os.makedirs(results_path, exist_ok=True)
            detector_name = self.detector.name if self.detector.tracker_name is None else '{}_{}'.format(
                self.detector.tracker_name, self.detector.name)
            if not os.path.exists('{}/{}.json'.format(results_path, detector_name)) or overwrite_cache:
                json.dump(self.class_wise_scores, open('{}/{}.json'.format(results_path, detector_name), 'w'), indent=4)
        return self.score, self.class_wise_scores

    def read_scores(self, voc=True, results_path=None):
        import os
        if voc:
            if results_path is None:
                results_path = '{}/{}/{}/VOC/'.format(eval_cache_path, self.name, self.dataset.name)
            if not os.path.exists('{}/{}.json'.format(results_path, self.detector.name)):
                self.evaluate_voc(cache_path=results_path)
            return json.load(open('{}/{}.json'.format(results_path, self.detector.name)))
        else:
            if results_path is None:
                results_path = '{}/{}/{}/MS_COCO/{}.json'.format(eval_cache_path, self.name, self.dataset.name,
                                                                 self.detector)
            if not os.path.exists(results_path):
                self.evaluate_coco()
            return json.load(open(results_path))

    def pr_plots(self, class_names=None, results_path=None, show=False):
        if class_names is None:
            class_names = self.class_names
        if results_path is None:
            results_path = '{}/{}/{}/{}/'.format(pr_plot_path, self.dataset.name, self.name, self.detector)
        if self.predictions is None:
            self.generate_detections()
            # raise ValueError("Detections not loaded for the metric")
        self.matched_objs = dict([(vid, {}) for vid in self.video_names])

        for cls in class_names:
            if cls in self.recalls:
                rec, prec = self.recalls[cls], self.precisions[cls]
            else:
                rec, prec, tp, fp, num_obj = self.get_precision_recall(cls)
            ap, m_rec, m_prec = voc_ap(list(rec), list(prec))

            filename = results_path
            if results_path is None:
                filename = '{}/{}.png'.format(self.detector.name, cls)
            self.draw_pr_plot(m_rec, m_prec, filename, "{} AP={}".format(self.detector, ap), show)

    def draw_pr_plot(self, m_rec, m_prec, filename, title, show):
        """
         Draw plot
        """
        import matplotlib.pyplot as plt
        plt.plot(m_rec, m_prec, '-o')
        # add a new penultimate point to the list (m_rec[-2], 0.0)
        # since the last line segment (and respective area) do not affect the AP value
        area_under_curve_x = m_rec[:-1] + [m_rec[-2]] + [m_rec[-1]]
        area_under_curve_y = m_prec[:-1] + [0.0] + [m_prec[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        plt.xlabel('Recall' if self.name == "mAP" else "V-Recall")
        plt.ylabel('Precision' if self.name == "mAP" else "V-Precision")
        axes = plt.gca()
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space
        if filename:
            from pathlib import Path
            os.makedirs(Path(filename).parent, exist_ok=True)
            plt.savefig(filename)
        plt.title(title)
        if show:
            plt.show()
        plt.cla()  # clear axes for next plot

    def get_precision_recall(self, cls_name, thresh=None):
        raise NotImplementedError("get_precision_recall() should be implemented in the metric subclass.")

    def cache_operating_points(self, fix_precision=None, overwrite=True, results_path=None):
        import numpy as np
        import os
        results = {}
        if self.predictions is None:
            self.generate_detections()
        self.matched_objs = dict([(vid, {}) for vid in self.video_names])
        for cls in self.class_names:
            rec, prec, tp, fp, fn = self.get_precision_recall(cls)
            if rec[-1] == 0 and prec[-1] == 0:
                results[cls] = 1
            elif fix_precision is None:
                fsc = [rec[idx] * prec[idx] / (rec[idx] + prec[idx]) for idx in range(len(prec))]
                results[cls] = min(self.predictions[cls][np.nanargmax(fsc)].confidence, 0.99)
            else:
                for det_idx, precision in enumerate(reversed(prec)):
                    if precision > fix_precision:
                        results[cls] = self.predictions[cls][-1 - det_idx].confidence
                        break
                if cls not in results:
                    results[cls] = 1

        if results_path is None:
            results_path = '{}/{}'.format(op_cache_path, self.dataset.name)
        os.makedirs(results_path, exist_ok=True)

        detector_name = self.detector.name if self.detector.tracker_name is None else '{}_{}'.format(
            self.detector.tracker_name, self.detector.name)
        file_path = '{}/{}{}.json'.format(results_path, detector_name, fix_precision)
        if overwrite or not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(results, f, indent=4)
        return results

    def get_operating_points(self, fix_precision=None, op_path=None):
        import os
        if op_path is None:
            op_path = '{}/{}/{}/'.format(op_cache_path, self.dataset.name, self.name)
        os.makedirs(op_path, exist_ok=True)
        detector_name = '{}_{}'.format(self.detector.tracker_name,
                                       self.detector.name) if self.detector.tracker_name else self.detector.name
        file_path = '{}/{}{}.json'.format(op_path, detector_name, fix_precision)
        if not os.path.exists(file_path):
            print("Operating points not found. Computing and caching...")
            self.cache_operating_points(fix_precision=fix_precision, results_path=op_path)
        with open(file_path) as f:
            results = json.load(f)
            # print("metric 222:", file_path, results)
        return results

    def cache_op_scores(self, fix_precision=None, cache_path=None, overwrite_cache=False, frame_level=True):
        if cache_path is None:
            op_name = 'OPFrame' if frame_level else 'OPVideo'
            cache_path = '{}/{}/{}/{}{}/'.format(op_scores_cache_path, self.name, self.dataset.name, op_name,
                                                 fix_precision)
        self.evaluate_voc(overwrite_cache=overwrite_cache, cache_path=cache_path)

    def read_op_scores(self, fix_precision=None, scores_path=None, frame_level=True):
        detector = self.detector.name if self.detector.tracker_name is None else '{}_{}'.format(
            self.detector.tracker_name, self.detector.name)
        import os
        if scores_path is None:
            op_name = 'OPFrame' if frame_level else 'OPVideo'
            scores_path = '{}/{}/{}/{}{}/'.format(op_scores_cache_path, self.name, self.dataset.name, op_name,
                                                  fix_precision)
        if not os.path.exists(scores_path + detector + '.json'):
            print("cached results not found. Computing and caching...\n")
            self.cache_op_scores(fix_precision, cache_path=scores_path)
        return json.load(open(scores_path + detector + '.json'))

    def timeline_plot(self, class_names, file_name, title=None, video=None, show=False):
        if video is None:
            false_positives = {}
            num_frames = 0
            for video in self.dataset:
                fps = self.frame_wise_fp[video.name]
                for i in range(len(video)):
                    false_positives[i + num_frames] = fps.get(i, 0)
                num_frames += len(video)
            gts = self.dataset.videos
        else:
            assert video in self.video_names
            false_positives = self.frame_wise_fp[video]
            # print(self.frame_wise_fp)
            gts = {video: self.dataset.videos[video]}

        import matplotlib
        from matplotlib.ticker import MaxNLocator
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        matplotlib.rcParams['ps.useafm'] = True
        matplotlib.rcParams['pdf.use14corefonts'] = True
        # matplotlib.rcParams['text.usetex'] = True

        fig = plt.figure(figsize=(6, 3))
        gs = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0, height_ratios=[1, 3])
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        num_frames = 0
        x_matched, y_matched, x_unmatched, y_unmatched = [], [], [], []
        class_ids = []
        for video in gts.values():
            for obj_id, obj in video.gt_objects.items():
                class_ids.append(obj.class_name)
                if class_names and obj.class_name not in class_names:
                    continue
                for idx, gt_set in enumerate(obj.sets):
                    col = 'g' if self.matched_sets[video.name][obj_id].get(gt_set.first.number, 0) > 0 else 'r'
                    start, end = gt_set.first.number + num_frames, gt_set.last.number + num_frames
                    patch1 = plt.Rectangle((start - 0.4, int(obj_id) - 0.4), end - start + 0.8, 0.8, color=col,
                                           alpha=0.3)
                    patch2 = plt.Rectangle((start - 0.4, int(obj_id) - 0.4), end - start + 0.8, 0.8, color='k',
                                           linewidth=0.5, fill=False)
                    ax1.add_patch(patch1)
                    ax1.add_patch(patch2)

                    for frame_id, matched_objs in self.matched_objs[video.name].items():
                        if obj_id in matched_objs:
                            x_matched.append(num_frames + frame_id)
                            y_matched.append(int(obj_id))
                        else:
                            x_unmatched.append(num_frames + frame_id)
                            y_unmatched.append(int(obj_id))
            num_frames += len(video)
            ax1.plot([num_frames, num_frames], [0, 2 + int(max(video.gt_objects.keys()))], 'k--')
            ax0.plot([num_frames, num_frames], [0, 2 + int(max(video.gt_objects.keys()))], 'k--')

        ax0.scatter(list(false_positives.keys()), list(false_positives.values()), s=3, c='r')
        ax0.fill_between(list(false_positives.keys()), list(false_positives.values()), color='r', alpha=0.4)
        ax1.scatter(x_unmatched, y_unmatched, s=3, c='r')
        ax1.scatter(x_matched, y_matched, s=4, c='g')

        ax0.set_ylim(ymin=0, ymax=max(false_positives.values()) + 1)
        ax0.set_ylabel('# FP')
        ax1.set_ylabel('Object ID')
        ax0.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax0.set_xticks([], minor=True)
        plt.ylim(-0.3)
        plt.xlabel('Frame number')
        if title is not None:
            fig.suptitle(title)
        plt.savefig(file_name, bbox_inches="tight", pad_inches=0, dpi=200)
        if show:
            plt.show()
        plt.close(fig)

        return false_positives


def get_video_wise_op_results(dataset, detector, frame_level=False):
    from metrics.mAP import mAP
    avg_cls_scores = {}
    num_videos = {}
    for video in dataset:
        if frame_level:
            _, cls_wise_score = mAP(video, detector)
            for cls, score in cls_wise_score.items():
                num_videos[cls] = num_videos.get(cls, 0) + 1
                for key in score:
                    avg_cls_scores[key] = avg_cls_scores.get(key, 0) + score[key]
    return avg_cls_scores

