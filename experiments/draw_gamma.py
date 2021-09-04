import csv
from argparse import ArgumentParser
from constants import vid_annotations_path
from viz import plot_gamma_boxes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--video_name', type=str)
    parser.add_argument('--frame_number', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    vid_name = args.video_name
    frame_id = str(args.frame_number).zfill(6)
    boxes = []
    for line in csv.reader(open('{}/{}.csv'.format(vid_annotations_path, vid_name))):
        if int(line[0]) == int(frame_id):
            boxes.append([int(x) for x in line[3:]])

    print(boxes)
    plot_gamma_boxes(vid_name, frame_id, boxes)


if __name__ == '__main__':
    main()