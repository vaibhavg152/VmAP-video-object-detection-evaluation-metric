# VmAP: video object detection evaluation metric
This repository contains the code for the paper: "VmAP: a fair evaluation metric for video object detection" at 
ACM Multimedia, 2021. 

## API examples

### Load a dataset
    from datasets.dataset import VID, COCO
    vid_dataset = VID(lazy_load=True)  # Load frames, sets only when necessary. Alternatively, use preload=True
    coco_dataset = COCO(lazy_load=True)  # Load frames, sets only when necessary. Alternatively, use preload=True

    # load only particular classes
    animal_dataset = VID.filter(classes=["elephant", "horse", "dog"])
    # load videos which contain only a single object (useful with single object trackers)
    single_obj_dataset = VID.filter(num_objects=1)
    print(single_obj_dataset[0].gt_objects)
    small_dataset = VID(num_videos=10)  # sample of the dataset

### Load a detector
    from models.detector import Detector
    frcnn = Detector(dataset=vid_dataset, detector_name="MEGA")  # Should error out if detections aren't available
    frcnn_sort = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.7,
                          tracker_name="SORT")  # Should error out if detections aren't available
    
    frcnn_max_f1 = Detector(dataset=vid_dataset, detector_name="MEGA",
                            confidence='max_f1')  # This is default (Uses class-wise max f1)
    frcnn_conf_05 = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.5)
    frcnn_conf_list = Detector("MEGA", vid_dataset,
                               confidence=[0.5, 0.1, 0.1, 0.2])  # Error since len(class_names) != len(conf)

### Dataset info
    for video in animal_dataset:
        print(video.name)
        print(video.shape)
        print(video.num_frames)
        for frame_gt in video:
            print(frame_gt.number)
            print(frame_gt.image.shape)
            print(frame_gt.bboxes)
    
        for obj in video.objects():
            print(obj.num_frames)  # How many frames did it stay in video
            print(len(obj.sets))  # All sets. Returns empty if form_sets = False during load.
            for gt_set in obj.sets():
                print(gt_set.first)  # Returns an object of frame class
                print(gt_set.last)  # Returns an object of frame class
                print(gt_set.num_frames)  # last.number - first.number

### Find mAP on the entire dataset
    from metrics.mAP import mAP
    print(mAP(vid_dataset, frcnn))

### Find mAP on animal dataset
    from metrics.vmap import vmap, VmAP
    print(mAP(animal_dataset, frcnn))

### Find video-wise mAP
    for video in vid_dataset:
        print(mAP(video, frcnn))
        print(vmap(video, frcnn))

### Find mAP for detectors in a list
    for det_name in ["SSD", "FGFA", "MEGA"]:
        det = Detector(vid_dataset, det_name)
        print(mAP(vid_dataset, det))

### Generate videos for detectors
    from viz import make_video
    
    video = animal_dataset[0]
    make_video(video, video, save_path='ground_truth_on_animal.avi')  # Draw ground truth on video
    make_video(video, frcnn)  # Note that frcnn[video.name] and video are the same type of object

### Compare gt and det
    for frame_gt, frame_dets in zip(video, frcnn(video)):
        print(frame_gt.bboxes)
        print(frame_dets.bboxes)  # frame object should have vid_name
    
    for frame_gt in frcnn[video.name]:
        print(frame_gt.number)
        print(frame_gt.detections[:])

### Timeline plot
    from viz import timeline_plot
    timeline_plot(video, frcnn, save_path=video.name + '_frcnn.pdf')

### PR Curve
    from viz import pr_curve, vp_vr_curve
    
    pr_curve(video, frcnn, save_path="pr_{}_frcnn.pdf")
    vp_vr_curve(video, frcnn, save_path="vp_vr_{}_frcnn.pdf")

### Format of evaluation results
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

## Dataset

The datasets used are ILSVRC2015 VID val videos (555) ,a subset of VID, VIDT (149), and a subset of objects with COCO 
classes (346 videos).

#### Add a new dataset 
* create a directory $ROOT/datasets/{dataset_name}/ containing the label-map (label_map.json) for the dataset.

## Detections
#### Available detectors
 * VID (555 videos)
     * MEGA
     * MEGA-BASE
     * RDN
     * FGFA
     * DFF

 * COCO classes (347 videos)
     * CENTRIPETAL
     * CORNERNET
     * DETR
     * FCOS
     * FRCNN_R50
     * HTC
     * RETINANET_R50
     * SSD
     * YOLOV3

 * VIDT (149 videos)
     * CATDET
     * FRCNN
     * RFCN
     * RFCN

#### Add a new detector
 * In $ROOT/constants.py, add the {detector_name} to the dictionary detector_classes.
 * Create a directory: $ROOT/detector/results/{detector_name}/
 * In the directory, create a file for each video named {video_name}.txt
 * Each line of the file contains a detection on the video. The format of each line is:
 ``` text
    frame_id(int) class_id(int) confidence_score(float) xmin(int) ymin(int) xmax(int) ymax(int)
```

## Evaluation

Available metrics:
 * Average Delay
 * KeyFrame-mAP
 * LengthNormalised-mAP
 * VmAP
 * mAP
 * VmAP_n


## Experiments

#### synthetic detectors
  * experiments/synthetic_biased_detectors.py: writes the detections of biased detectors at `$ROOT/detector/ARTIFICIAL/{bias}`.
    The biases in currently implemented detectors are:
    * Contrast: (I-Ib)/Ib where I is the illumination inside bounding box and Ib is the illumination in the region
    surrounding the bounding box.
    * Illumination: average pixel values inside the bounding box.
    * Hue: biased against a particular part of the hue circle (red).
    * Size: biased against small objects.
    * Speed: biased against fast moving objects.
  * custom false positives are generated randomly and added with the ground truth boxes to keep the precision at 0.5. 
  * evaluates the synthetic detectors and plots the figure as in the paper.

#### custom datasets
   * dataset_curation/make_datasets.py: creates the custom datasets for fooling mAP as an evaluation metric and saves in 
    $ROOT/dataset_curation/custom_datasets/{dataset}.csv.
        * write_datasets_123(): creates the datasets D1, D2 and D3 as described in the paper.
        * write_datasets_abc(): creates the datasets Da, Db and Dc as described in the paper.
   * dataset_curation/eval_datasets.py: evaluates various detectors on the custom datasets on VmAP and mAP and writes the result in 
    $ROOT/dataset_curation/table1.csv
    
#### ablation_studies
   * ablation_studies/eval_set_criteria.py : Evaluate various detectors using different set-formation criteria and saves to 
   $ROOT/ablation_studies/table3.csv. Current criteria are:
        * Location (VmAPl): a new ground truth set is formed whenever the location of the bounding box in the frame 
        changes significantly. (when IoU overlap of the new box with the first frame of the set is greater than the 
        threshold.)
        * Appearance (VmAPa): a new ground truth set is formed whenever the appearance of the bounding box changes 
        significantly. (when the L-2 distance between the histogram feature vectors is greater than the threshold.)
        * Location + Appearance (VmAPal): a new ground truth set is formed whenever the appearance or the location of 
        the bounding box changes significantly.
        * Time (VmAPt): a new ground truth set is formed after a constant time (40 frames) from the first frame of the set. 
   *  ablation_studies/locvsapp.py : code to visualize the sets formed by the location and appearance criteria for the train class. 
   Reads from the path ($DATA)/Data/VID/val/ and writes to ($ROOT)/ablatation_studies/set_visuals/{criterion}/{class}/
