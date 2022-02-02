from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(client_cfg, use_cuda):
    return DeepSort(client_cfg.DeepSORT.DS_Model_Path,
                    # namesfile=cfg.DEEPSORT.CLASS_NAMES,
                    namesfile=client_cfg.DeepSORT.YOLO_Classes_Path,
                    max_dist=client_cfg.DeepSORT.Max_Dist,
                    min_confidence=client_cfg.DeepSORT.Min_Confidence,
                    nms_max_overlap=client_cfg.DeepSORT.NMS_Max_Overlap,
                    max_iou_distance=client_cfg.DeepSORT.Max_IOU_Distance,
                    max_age=client_cfg.DeepSORT.Max_Age,
                    n_init=client_cfg.DeepSORT.N_Init,
                    nn_budget=client_cfg.DeepSORT.NN_Budget,
                    use_osnet=not(
                        client_cfg.DeepSORT.Original_Feature_Extractor),
                    client_cfg=client_cfg,
                    use_cuda=use_cuda)


# def build_tracker(cfg, use_cuda):
#     return DeepSort(cfg.DEEPSORT.REID_CKPT,# namesfile=cfg.DEEPSORT.CLASS_NAMES,
#                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
#                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
