# modified from bruce yang2012/nms_python/nms_utils.py
import numpy as np

from constants import NMS_THRESHOLD
from video_elements.bndbox import gamma_iou


def non_max_suppression_gamma_iou(boxes, scores, gamma):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    indices = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(indices) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(indices) - 1
        i = indices[last]
        pick.append(indices[last])
        suppress = [last]

        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = indices[pos]
            # if there is sufficient overlap, suppress the
            # current bounding box
            overlap = gamma_iou(boxes[i], boxes[j], gamma)
            if overlap > NMS_THRESHOLD:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        indices = np.delete(indices, suppress)

    # return only the bounding boxes that were picked
    return pick


def non_max_suppression_fast(boxes, scores=None):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = y2

    # if probabilities are provided, sort on them instead
    if scores is not None:
        indices = scores

    # sort the indexes
    indices = np.argsort(indices)

    # keep looping while some indexes still remain in the indexes list
    while len(indices) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(indices) - 1
        i = indices[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[indices[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        indices = np.delete(indices, np.concatenate(([last], np.where(overlap > NMS_THRESHOLD)[0])))

    # return only the bounding boxes that were picked
    return pick
