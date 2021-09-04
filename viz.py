import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

from constants import vid_image_path, timeline_plot_path, gamma_vis_path, set_formation_criteria, \
    experiments_results_path, plotted_detections_path
from datasets.dataset import Dataset
from metrics.mAP import MAP
from metrics.vmap import VmAP
from models.detector import Detector


def make_video(video, frames, save_path=None):
    import cv2
    temp_dir = './temp/'
    os.makedirs(temp_dir, exist_ok=True)
    # if isinstance(frames, list):
    #     frames =
    for frame in frames:
        for annotation in frame.bboxes:
            file_name = '{}/{}.JPEG'.format(temp_dir, str(annotation.frame_id).zfill(6))
            if os.path.isfile(file_name):
                img = cv2.imread(file_name)
            else:
                img = cv2.imread(vid_image_path + '{}/{}.JPEG'.format(video.name, str(annotation.frame_id).zfill(6)))
            text = "{}".format(annotation.class_name)
            img = draw_box_on_image(img, annotation.box, text, color=colors[0], thickness=2)
            cv2.imwrite(file_name, img)
    if save_path is None:
        save_path = './video_{}.mp4'.format(video.name)
    make_video_from_frames(temp_dir, save_path)
    os.rmdir(temp_dir)


def timeline_plot(video, detector, save_path=None, show=False):
    print(video.name, end=" ")
    if save_path is None:
        name = detector.name + '_' + detector.tracker_name if detector.tracker_name else detector.name
        save_path = "{}/{}/{}/{}.pdf".format(timeline_plot_path, video.dataset_name, name, video.name)
    os.makedirs(Path(save_path).parent, exist_ok=True)

    metric = MAP(video, detector, evaluate=MAP.voc)
    num_tp = sum([metric.class_wise_scores[cls]['TP'] for cls in video.class_names])
    num_fp = sum([metric.class_wise_scores[cls]['FP'] for cls in video.class_names])
    num_fn = sum([metric.class_wise_scores[cls]['FN'] for cls in video.class_names])

    vmap = VmAP(video, detector, evaluate=VmAP.voc)
    v_rec = np.average([vmap.class_wise_scores[cls]['Recall'] for cls in video.class_names])
    v_prec = np.average([vmap.class_wise_scores[cls]['Precision'] for cls in video.class_names])

    title = "VmAP={:.1f} mAP={:.1f} VR={:.1f} VP={:.1f} tp={} fp={} fn={}".format(vmap.score, metric.score, v_rec,
                                                                                  v_prec, num_tp, num_fp, num_fn)
    print(title)
    vmap.timeline_plot(detector.classes, save_path, title, show=show)


def pr_curve(video, detector, classes=None, save_path=None, show=False):
    metric = MAP(video, detector, evaluate=MAP.voc)
    metric.pr_plots(classes, save_path, show)


def vp_vr_curve(video, detector, classes=None, save_path=None, show=False):
    metric = VmAP(video, detector, evaluate=VmAP.voc)
    metric.pr_plots(classes, save_path, show)


colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (255, 255, 0), (0, 255, 255)]


def plot_gamma_boxes(video_name, frame_id, boxes, alpha=0.4):
    img = cv2.imread('{}/{}/{}.JPEG'.format(vid_image_path, video_name, frame_id))
    for gamma in range(5, 21, 50):
        print(gamma)
        for bbox in boxes:
            left = [max(0, bbox[0] - gamma), max(0, bbox[1] - gamma), bbox[0] + gamma, bbox[3] + gamma]
            up = [bbox[0], max(0, bbox[1] - gamma), bbox[2], bbox[1] + gamma]
            down = [bbox[0], bbox[3] - gamma, bbox[2], bbox[3] + gamma]
            right = [bbox[2] - gamma, max(0, bbox[1] - gamma), bbox[2] + gamma, bbox[3] + gamma]
            left_down = [max(0, bbox[0] - gamma), bbox[1] + gamma, max(0, bbox[2] - gamma), bbox[3] + gamma]
            right_down = [bbox[0] + gamma, bbox[1] + gamma, bbox[2] + gamma, bbox[3] + gamma]
            right_up = [bbox[0] + gamma, max(0, bbox[1] - gamma), bbox[2] + gamma, max(0, bbox[3] - gamma)]
            left_up = [max(0, bbox[0] - gamma), max(0, bbox[1] - gamma), max(0, bbox[2] - gamma),
                       max(0, bbox[3] - gamma)]
            overlay = img.copy()
            img = draw_box_on_image(img, left, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, up, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, down, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, right, color=(0, 0, 255), thickness=-1)

            img = draw_box_on_image(img, left_down, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, left_up, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, right_down, color=(0, 0, 255), thickness=-1)
            img = draw_box_on_image(img, right_up, color=(0, 0, 255), thickness=-1)

            img = draw_box_on_image(img, bbox, color=(0, 255, 0), thickness=10)
            img = cv2.addWeighted(img, alpha, overlay, 1 - alpha, 0)
            cv2.imwrite('{}/{}_{}_gamma_{}.png'.format(gamma_vis_path, video_name, frame_id, gamma), img)
            cv2.imwrite('{}/color/{}_{}.png'.format(gamma_vis_path, video_name, frame_id), img)
            print("saving")


def average_gt_size_plot(dataset, n_splits=3, save_path=None, show=True):
    all_class_names = sorted(dataset.classes)
    import numpy
    for split_i in range(n_splits):
        split_size = int(numpy.ceil(len(all_class_names) / n_splits))
        classes = all_class_names[split_i * split_size:(split_i + 1) * split_size]
        width = 0.2
        ticks = np.arange(len(classes)) + width
        for idx, cr in enumerate(set_formation_criteria):
            cls_wise_sizes = {}
            for cls in classes:
                cls_wise_sizes[cls] = []
            for video in dataset:
                gt = video.gt_objects
                for obj_id, obj in gt.items():
                    for trail in obj.sets:
                        if obj.class_name in classes:
                            cls_wise_sizes[obj.class_name].append(len(trail))
            mean_sizes = {}
            for cls in classes:
                mean_sizes[cls] = np.mean(cls_wise_sizes[cls])
            x = np.arange(len(classes)) + idx * width
            plt.barh(x[:10], width=list(mean_sizes.values()), height=width)
        plt.yticks(ticks=ticks, labels=classes)
        plt.xlabel('Average Set Length (#frames)')
        plt.ylabel('Class Names')
        plt.legend(set_formation_criteria)
        if save_path is None:
            save_path = "{}/avg_set_size_{}.png".format(experiments_results_path, split_i)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
        if show:
            plt.show()


def draw_box_on_image(image, box, text='', color=(255, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=.5,
                      thickness=1):
    xmin, ymin, xmax, ymax = [int(i) for i in box]
    top_left = (int(xmin), int(ymin))
    bottom_right = (int(xmax), int(ymax))

    text_location = (int(top_left[0]), int(top_left[1]) - 2)
    image = cv2.rectangle(img=image, pt1=top_left, pt2=bottom_right, color=color, thickness=thickness)
    image = cv2.putText(image, text, text_location, font, font_scale, color, thickness, cv2.LINE_AA)
    return image


def make_video_from_frames(in_dir, out_dir):
    frames = [cv2.imread(img) for img in sorted(glob.glob(in_dir + '/*.*'))]
    h, w, _ = frames[0].shape
    vid_wr = cv2.VideoWriter(out_dir, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f in frames:
        vid_wr.write(f)
    cv2.destroyAllWindows()
    vid_wr.release()


def plot_tracked_video(detectors, dataset, confidence, tracker_name):
    for det_name in detectors:
        print(det_name)

        for video in dataset:
            print(video.name)

            detector = Detector(dataset, det_name, confidence)
            output_path = '{}/{}/{}/{}.mp4'.format(plotted_detections_path, dataset.name, det_name, video.name)
            make_video(video, detector.detections[video.name], save_path=output_path)

            detector = Detector(dataset, det_name, confidence, tracker_name=tracker_name)
            output_path = '{}/{}/{}_{}/{}.mp4'.format(plotted_detections_path, dataset.name, tracker_name, det_name,
                                                      video.name)
            make_video(video, detector.detections[video.name], save_path=output_path)


def frame_wise_confidence(detector_name='DFF', vid_name='ILSVRC2015_val_00000000'):
    d = {}

    for line in csv.reader(open('../detector/tracked_detections/{}/{}.csv'.format(detector_name, vid_name))):
        frame_id = int(line[0])
        conf = float(line[2])
        d[frame_id] = max(d.get(frame_id, 0), conf)

    plt.scatter(list(d.keys()), list(d.values()))
    print(len(d.keys()))
    plt.xlim(0, 100)
    plt.ylim(0.7, 1.2)
    plt.xlabel('frame-id')
    plt.ylabel('confidence value for object-0')
    plt.show()


if __name__ == '__main__':
    average_gt_size_plot(Dataset('VID'))
