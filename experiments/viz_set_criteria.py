import os
from argparse import ArgumentParser

import cv2

from constants import experiments_results_path, vid_image_path, set_formation_criteria
from datasets.dataset import Dataset
from viz import draw_box_on_image


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('classes', nargs='+')
    parser.add_argument('--dataset', default='VID')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    criteria = set_formation_criteria[:2]
    alpha = 0.3
    for cr in criteria:
        dataset = Dataset(args.dataset, pre_load=True, class_names=args.classes, set_criteria=cr)
        for name, vid in dataset:
            print(vid)
            for object_id, obj in vid.gt_objects:
                out_path = '{}/set_visuals/{}/{}/'.format(experiments_results_path, cr, obj.class_name)
                os.makedirs(out_path, exist_ok=True)
                print(out_path)
                for set_idx, t in enumerate(obj.sets):
                    output_path = out_path + '{}_obj{}_set{}_frame{}.jpg'.format(vid, object_id, set_idx, t.first)
                    if os.path.exists(output_path):
                        continue
                    img_start = cv2.imread(vid_image_path + name + '/{}.JPEG'.format(str(t.first).zfill(6)))
                    img_last = cv2.imread(vid_image_path + name + '/{}.JPEG'.format(str(t.last.number).zfill(6)))
                    print(t.last.number)
                    img_start = draw_box_on_image(img_start, t.frames[t.first], color=(0, 0, 0), thickness=5)
                    img_last = draw_box_on_image(img_last, t.frames[t.last.number], color=(0, 0, 0), thickness=5)
                    img_last = cv2.addWeighted(img_last, alpha, img_start, 1 - alpha, 0)
                    for bbox in t.frames.all_values():
                        xc, yc = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                        img_last = cv2.circle(img_last, (xc, yc), 3, (0, 0, 255), thickness=-1)
                    cv2.imwrite(output_path, img_last)
