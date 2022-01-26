from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(args, client_cfg, use_cuda):
    return DeepSort(args.ds_model_path, namesfile=args.yolo_classes_path,  # namesfile=cfg.DEEPSORT.CLASS_NAMES,
                    max_dist=args.max_dist, min_confidence=args.min_confidence,
                    nms_max_overlap=args.nms_max_overlap, max_iou_distance=args.max_iou_distance,
                    max_age=args.max_age, n_init=args.n_init, nn_budget=args.nn_budget,
                    use_osnet=not(args.original_feature_extractor), 
                    client_cfg=client_cfg,
                    use_cuda=True)


# def build_tracker(cfg, use_cuda):
#     return DeepSort(cfg.DEEPSORT.REID_CKPT,# namesfile=cfg.DEEPSORT.CLASS_NAMES,
#                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
#                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
#                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
