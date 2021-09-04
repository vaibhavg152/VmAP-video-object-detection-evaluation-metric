import os

from datasets.dataset import VID, Dataset
from models.detector import Detector


def test_data_lazy():
    vid_dataset = VID(lazy_load=True)  # Load only when necessary. Alternatively, use preload=True
    assert not len(vid_dataset[0].gt_objects['0'].sets)


def test_data_preload():
    vid_dataset = VID(pre_load=True)
    assert len(vid_dataset[0].gt_objects['0'].sets)


def test_data_filter_classes():
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"])
    assert animal_dataset.classes == ["elephant", "horse", "dog"]


def test_data_filter_single():
    single_obj_dataset = VID.filter(num_objects=1)
    assert len(single_obj_dataset[0].gt_objects) == 1


def test_data_filter_num_videos():
    small_dataset = VID(num_videos=10)
    assert len(small_dataset.videos) == 10


def test_detector_load():
    vid_dataset = VID(pre_load=True)
    det = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0)
    assert det.classes
    assert det.video_names
    assert isinstance(det.conf_thresholds, dict)
    assert isinstance(list(det.conf_thresholds.values())[0], float) or isinstance(
        list(det.conf_thresholds.values())[0], int)


def test_detector_load_params():
    small_dataset = VID(num_videos=10)
    det = Detector(dataset=small_dataset, detector_name="MEGA", confidence=0.5)
    assert isinstance(list(det.conf_thresholds.values())[0], float) or isinstance(
        list(det.conf_thresholds.values())[0], int)

    det = Detector(dataset=VID.filter(classes=["dog", "car", "bus", "motorcycle"]), detector_name="MEGA",
                   confidence=[0.5, 0.1, 0.15, 0.2])
    assert det.conf_thresholds["dog"] == 0.5
    assert det.conf_thresholds["car"] == 0.1
    assert det.conf_thresholds["bus"] == 0.15
    assert det.conf_thresholds["motorcycle"] == 0.2


def test_detector_load_tracker():
    vid_dataset = VID(pre_load=True)
    det = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.7, tracker_name="SORT")
    assert det.tracker_name
    assert isinstance(list(det.conf_thresholds.values())[0], float) or isinstance(
        list(det.conf_thresholds.values())[0], int)


def test_detector_load_op():
    small_dataset = VID(num_videos=10)
    det = Detector(dataset=small_dataset, detector_name="MEGA", confidence='max_f1')
    assert isinstance(list(det.conf_thresholds.values())[0], float) or isinstance(
        list(det.conf_thresholds.values())[0], int)


# Dataset info
def test_video_elements():
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"], num_videos=3)
    for vid in animal_dataset:
        assert isinstance(vid.name, str)
        print(vid.shape)
        print(vid.num_frames)
        assert isinstance(vid.num_frames, int) and vid.num_frames > 0
        vid.load_sets()
        for fid, frame_gt in enumerate(vid):
            print(frame_gt.number)
            # print(frame_gt.image.shape)
            print(frame_gt.bboxes)
            if fid > 2:
                break

        for obj_id, obj in enumerate(vid.objects()):
            print(obj.num_frames)  # How many frames did it stay in video
            assert obj.num_frames
            print(len(obj.sets))  # All sets. Returns empty if form_sets = False during load.
            assert len(obj.sets)
            for gt_set in obj.sets[:3]:
                print(gt_set.first)  # Returns an object of frame class
                print(gt_set.last)  # Returns an object of frame class
                print(gt_set.num_frames)  # last.number - first.number
                assert gt_set.num_frames
            if obj_id > 2:
                break


# Find mAP on the entire dataset
def test_map_entire():
    from metrics.mAP import mAP
    vid_dataset = VID(lazy_load=True)
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0)
    assert mAP(vid_dataset, mega)


# Find mAP on animal dataset
def test_map_subset():
    from metrics.mAP import mAP
    animal_dataset = VID.filter(classes=["sheep", "horse", "squirrel"])
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0)
    assert mAP(animal_dataset, mega)


# Find video-wise mAP
def test_map_video_wise():
    from metrics.mAP import mAP
    from metrics.vmap import vmap
    from constants import set_formation_map
    n = 3
    vid_dataset = VID(pre_load=True, num_videos=n)
    vid_dataset2 = VID(pre_load=True, num_videos=n, set_criteria=set_formation_map)
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.2)

    for video in list(vid_dataset)[:n]:
        print(video.name, mAP(video, mega))
    for video in list(vid_dataset2)[:n]:
        print(video.name, vmap(video, mega))


# Find mAP for detectors in a list
def test_map_multiple():
    from metrics.mAP import mAP
    vid_dataset = VID(lazy_load=True, num_videos=10)
    for det_name in ["RDN", "FGFA", "MEGA"]:
        det = Detector(vid_dataset, det_name)
        print(mAP(vid_dataset, det))


# Find mAP for detectors in a list
def test_vmap_multiple():
    from metrics.vmap import vmap
    vid_dataset = VID(pre_load=True, num_videos=10)
    for det_name in ["RDN", "FGFA", "MEGA"]:
        det = Detector(vid_dataset, det_name)
        print(vmap(vid_dataset, det))


def test_make_video_gt():
    # Generate videos for detectors
    from viz import make_video
    animal_dataset = VID.filter(classes=["elephant", "hamster", "dog"], num_videos=10)
    video = animal_dataset[0]
    make_video(video, video, save_path='ground_truth_on_animal.avi')  # Draw ground truth on video
    assert os.path.exists('ground_truth_on_animal.avi')


def test_make_video_det():
    from viz import make_video
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"], num_videos=10)
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0)
    video = animal_dataset[0]
    make_video(video, mega[video.name], save_path='mega_on_animal.avi')
    assert os.path.exists('mega_on_animal.avi')


# Compare gt and det
def test_compare_gt_det():
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"], num_videos=10)
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0)
    video = animal_dataset[0]
    i = 0
    for frame_gt, frame_dets in zip(video, mega(video)):
        print(frame_gt.bboxes)
        print(frame_dets.bboxes)
        assert type(frame_gt) == type(frame_dets)
        i += 1
        if i > 3:
            break


def test_print_dets():
    from video_elements.video_elements import Frame
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"], num_videos=10)
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0)
    video = animal_dataset[0]
    for frame_gt in mega[video.name]:
        assert isinstance(frame_gt, Frame)
        print(frame_gt.number)
        print(frame_gt.bboxes[:])


def test_timeline_plot():
    from viz import timeline_plot
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"], num_videos=10)
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0.8)
    video = animal_dataset[0]
    timeline_plot(video, mega, save_path=video.name + '_mega.pdf')
    assert os.path.exists(video.name + '_mega.pdf')


def test_pr_curve():
    from viz import pr_curve, vp_vr_curve
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"])
    mega = Detector(dataset=animal_dataset, detector_name="MEGA", confidence=0)
    video = animal_dataset[0]
    pr_curve(video, mega, save_path="pr_mega.pdf")
    assert os.path.exists("pr_mega.pdf")
    vp_vr_curve(video, mega, save_path="vp_vr_mega.pdf")
    assert os.path.exists("vp_vr_mega.pdf")


def test_vmap_results():
    from metrics.vmap import VmAP
    vid_dataset = Dataset('VID', pre_load=True, class_names=["hamster", "lion", "car"])
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.8)
    match_results = VmAP(vid_dataset, mega, evaluate=VmAP.voc)
    print(match_results.class_wise_scores)
    assert match_results.class_wise_scores  # contains a dictionary with key=class_name
    for cls_name in vid_dataset.classes:
        assert (match_results.class_wise_scores[cls_name])
        assert (match_results.class_wise_scores[cls_name]['TP'] is not None)
        assert (match_results.class_wise_scores[cls_name]['FP'] is not None)
        assert (match_results.class_wise_scores[cls_name]['FN'] is not None)
        assert (match_results.class_wise_scores[cls_name]['Recall'] is not None)
        assert (match_results.class_wise_scores[cls_name]['Precision'] is not None)
        assert (match_results.class_wise_scores[cls_name]['F1'] is not None)
        assert (match_results.class_wise_scores[cls_name]['AP'] is not None)
    assert match_results.score
    match_results.pr_plots(["car"], 'vp_vr_curve2.pdf', show=False)
    assert os.path.exists('vp_vr_curve2.pdf')
    match_results.timeline_plot("car", show=False, video=vid_dataset[1].name, file_name="timeline_vmap.pdf")
    assert os.path.exists("timeline_vmap.pdf")


def test_map_results():
    from metrics.mAP import MAP
    vid_dataset = VID(pre_load=False, class_names=["hamster", "lion"])
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0)
    match_results = MAP(vid_dataset, mega, evaluate=MAP.voc)
    assert match_results.class_wise_scores  # contains a dictionary with key=class_name
    for cls_name in vid_dataset.classes:
        assert (match_results.class_wise_scores[cls_name])
        assert (match_results.class_wise_scores[cls_name]['TP'])
        assert (match_results.class_wise_scores[cls_name]['FP'])
        assert (match_results.class_wise_scores[cls_name]['FN'])
        assert (match_results.class_wise_scores[cls_name]['Recall'])
        assert (match_results.class_wise_scores[cls_name]['Precision'])
        assert (match_results.class_wise_scores[cls_name]['F1'])
        assert (match_results.class_wise_scores[cls_name]['AP'])
    match_results.pr_plots(["hamster"], 'pr_curve2.pdf', show=False)
    assert os.path.exists('pr_curve2.pdf')
