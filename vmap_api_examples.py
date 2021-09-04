from datasets.dataset import VID
from metrics.mAP import mAP
from metrics.vmap import vmap, VmAP
from models.detector import Detector

vid_dataset = VID(lazy_load=True)  # Load only when necessary. Alternatively, use preload=True

animal_dataset = VID.filter(classes=["elephant", "horse", "dog"])
single_obj_dataset = VID.filter(num_objects=1)
print(single_obj_dataset[0].gt_objects)
small_dataset = VID(num_videos=10)

frcnn = Detector(dataset=vid_dataset, detector_name="MEGA")  # Should error out if detections aren't available
frcnn_sort = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.7,
                      tracker_name="SORT")  # Should error out if detections aren't available

frcnn_max_f1 = Detector(dataset=vid_dataset, detector_name="MEGA",
                        confidence='max_f1')  # This could be default (Uses class-wise max f1)
frcnn_conf_05 = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.5)
frcnn_conf_list = Detector("MEGA", vid_dataset,
                           confidence=[0.5, 0.1, 0.1, 0.2])  # Error since len(class_names) != len(conf)

# Dataset info
for vid in animal_dataset:
    print(vid.name)
    print(vid.shape)
    print(vid.num_frames)
    for frame_gt in vid:
        print(frame_gt.number)
        print(frame_gt.image.shape)
        print(frame_gt.bboxes)

    for obj in vid.objects():
        print(obj.num_frames)  # How many frames did it stay in video
        print(len(obj.sets))  # All sets. Returns empty if form_sets = False during load.
        for gt_set in obj.sets():
            print(gt_set.first)  # Returns an object of frame class
            print(gt_set.last)  # Returns an object of frame class
            print(gt_set.num_frames)  # last.number - first.number

# Find mAP on the entire dataset
print(mAP(vid_dataset, frcnn))

# Find mAP on animal dataset
print(mAP(animal_dataset, frcnn))

# Find video-wise mAP
for video in vid_dataset:
    print(mAP(video, frcnn))
    print(vmap(video, frcnn))

# Find mAP for detectors in a list
for det_name in ["SSD", "FGFA", "MEGA"]:
    det = Detector(vid_dataset, det_name)
    print(mAP(vid_dataset, det))

# Generate videos for detectors
from viz import make_video, timeline_plot

video = animal_dataset[0]
make_video(video, video, save_path='ground_truth_on_animal.avi')  # Draw ground truth on video

make_video(video, frcnn)  # Note that frcnn[video.name] and video are the same type of object

# Compare gt and det
for frame_gt, frame_dets in zip(video, frcnn(video)):
    print(frame_gt.bboxes)
    print(frame_dets.bboxes)  # frame object should have vid_name

for frame_gt in frcnn[video.name]:
    print(frame_gt.number)
    print(frame_gt.detections[:])

# Timeline plot
# PR Curve
from viz import pr_curve, vp_vr_curve

timeline_plot(video, frcnn, save_path=video.name + '_frcnn.pdf')
pr_curve(video, frcnn, save_path="pr_{}_frcnn.pdf")
vp_vr_curve(video, frcnn, save_path="vp_vr_{}_frcnn.pdf")

match_results = VmAP(vid_dataset, frcnn, evaluate=VmAP.voc)
print(match_results.class_wise_scores)  # contains a dictionary with key=class_name
for cls_name in vid_dataset.classes:
    print(match_results.class_wise_scores[cls_name])
    print(match_results.class_wise_scores[cls_name]['TP'])
    print(match_results.class_wise_scores[cls_name]['FP'])
    print(match_results.class_wise_scores[cls_name]['FN'])
    print(match_results.class_wise_scores[cls_name]['Recall'])
    print(match_results.class_wise_scores[cls_name]['Precision'])
    print(match_results.class_wise_scores[cls_name]['F1'])
    print(match_results.class_wise_scores[cls_name]['AP'])
print(match_results.score)
match_results.pr_plots(show=True)
match_results.timeline_plot('plot.pdf', show=True, video='ILSVRC...', file_name=None)

match_results = mAP(vid_dataset, frcnn)
print(match_results.class_wise_scores)  # contains a dictionary with key=class_name
for cls_name in vid_dataset.classes:
    print(match_results.class_wise_scores[cls_name])
    print(match_results.class_wise_scores[cls_name]['TP'])
    print(match_results.class_wise_scores[cls_name]['FP'])
    print(match_results.class_wise_scores[cls_name]['FN'])
    print(match_results.class_wise_scores[cls_name]['Recall'])
    print(match_results.class_wise_scores[cls_name]['Precision'])
    print(match_results.class_wise_scores[cls_name]['F1'])
    print(match_results.class_wise_scores[cls_name]['AP'])
print(match_results.score)
match_results.pr_plots(show=True)
match_results.timeline_plot('plot.pdf', show=True, video='ILSVRC...')

# frcnn -> dict of video, detections
# video detection -> list of frames
# frame -> image + annotations/detections
# detection -> conf, bbox, cls
