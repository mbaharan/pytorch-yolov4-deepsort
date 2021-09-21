from .YOLOV4 import yolo
from .YOLOV4.yolo import YOLO


__all__ = ['build_detector']


def build_detector(model_path, anchors_path, classes_path):
    return YOLO(model_path, anchors_path, classes_path)
