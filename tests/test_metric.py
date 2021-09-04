from datasets.dataset import VID
from models.detector import Detector


def test_map_video_wise():
    from metrics.mAP import mAP
    from metrics.vmap import vmap
    vid_dataset = VID(pre_load=True)
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0.2)
    print(mAP(vid_dataset, mega))
    x, y = vmap(vid_dataset, mega)
    tps = sum([sc['TP'] for sc in y.values()])
    fps = sum([sc['FP'] for sc in y.values()])
    fns = sum([sc['FN'] for sc in y.values()])
    print(tps, fps, fns, x)
    n = 30
    for video in list(vid_dataset)[:n]:
        print(video.name, vmap(video, mega))


def op():
    from metrics.mAP import MAP
    vid_dataset = VID(pre_load=True, video_names=['ILSVRC2015_val_00000000', 'ILSVRC2015_val_00001000'])
    print(vid_dataset.videos.keys())
    mega = Detector(dataset=vid_dataset, detector_name="MEGA", confidence=0)
    m = MAP(vid_dataset, mega, None)
    m.cache_operating_points(None, True, './')


if __name__ == '__main__':
    # test_map_video_wise()
    op()
