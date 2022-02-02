from .detector import build_detector
from .deep_sort import build_tracker


def build_yolo_ds(args, client_cfg, use_cuda=True):
    detector = build_detector(
        args.yolo_model_path, args.yolo_anchors_path, client_cfg.DeepSORT.YOLO_Classes_Path)
    deepsort = build_tracker(client_cfg, use_cuda)

    return detector, deepsort
