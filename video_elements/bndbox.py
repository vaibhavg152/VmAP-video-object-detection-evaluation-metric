class BoundingBox:
    def __init__(self, video_name, frame_id, class_name, box, confidence=1, object_id=None):
        self.video_name = video_name
        self.frame_id = frame_id
        self.class_name = class_name
        self.box = [int(float(k)) for k in box]
        self.object_id = object_id
        self.confidence = confidence

    def get_class(self):
        return self.class_name

    def __str__(self):
        return "Video name: {}\tFrame number: {}\t Object id: {}\tClass name: {}\t Confidence score:" \
               " {:.3f}\tBounding box {}\n".format(self.video_name, self.frame_id, self.object_id, self.class_name,
                                                   self.confidence, self.box)

    def to_list(self):
        if self.object_id:
            return [self.frame_id, self.object_id, self.class_name, self.confidence] + self.box
        return [self.frame_id, self.class_name, self.confidence] + self.box


def gamma_iou(bbox1, bbox2, gamma=0):
    """
    Util function for calculating gamma-relaxed-IoU.

    Parameters
    -----------
    bbox1: list(int)
        (required)bounding boxes in format [xmin, ymin, xmax, ymax]
    bbox2: list(int)
        (required)bounding boxes in format [xmin, ymin, xmax, ymax]
    gamma: int
        (default=0) the value of relaxation factor in pixels. Gamma=0 returns IoU

    Returns
    --------
    maxIoU: float
        the maximum IoU after gamma relaxation in range [0,gamma]
    """
    if gamma == 0:
        return iou_box(bbox1, bbox2)

    max_iou = iou_box(bbox1, bbox2)
    for gamma in range(1, gamma + 1):
        # shift the box-2 left,right,up,down by gamma-pixels and store the maximum
        left = iou_box(bbox1, [max(0, bbox2[0] - gamma), bbox2[1], max(0, bbox2[2] - gamma), bbox2[3]])
        right = iou_box(bbox1, [bbox2[0] + gamma, bbox2[1], bbox2[2] + gamma, bbox2[3]])
        down = iou_box(bbox1, [bbox2[0], bbox2[1] + gamma, bbox2[2], bbox2[3] + gamma])
        up = iou_box(bbox1, [bbox2[0], max(0, bbox2[1] - gamma), bbox2[2], max(0, bbox2[3] - gamma)])
        left_down = iou_box(bbox1,
                            [max(0, bbox2[0] - gamma), bbox2[1] + gamma, max(0, bbox2[2] - gamma), bbox2[3] + gamma])
        right_down = iou_box(bbox1, [bbox2[0] + gamma, bbox2[1] + gamma, bbox2[2] + gamma, bbox2[3] + gamma])
        right_up = iou_box(bbox1,
                           [bbox2[0] + gamma, max(0, bbox2[1] - gamma), bbox2[2] + gamma, max(0, bbox2[3] - gamma)])
        left_up = iou_box(bbox1, [max(0, bbox2[0] - gamma), max(0, bbox2[1] - gamma), max(0, bbox2[2] - gamma),
                                  max(0, bbox2[3] - gamma)])
        curr_max = max(left, right, up, down, left_up, left_down, right_up, right_down)
        if curr_max > max_iou:
            max_iou = curr_max
    return max_iou


def iou_box(bbox1, bbox2):
    """
    Util function for calculating IoU.

    Parameters
    -----------
    bbox1: list(int)
        bounding boxes in format [xmin, ymin, xmax, ymax]
    bbox2: list(int)
        bounding boxes in format [xmin, ymin, xmax, ymax]

    Returns
    --------
    IoU: float
        IoU of the 2 bounding boxes. returns -1 if boxes do not overlap
    """
    iw = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]) + 1
    ih = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]) + 1
    if iw <= 0 or ih <= 0:
        return 0
    area_int = iw * ih
    area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    area_uni = area1 + area2 - area_int
    return area_int / area_uni
