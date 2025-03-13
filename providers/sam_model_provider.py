import os
import torch
import multiprocessing
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

class SAMModelProvider:
    _model = None
    _predictor = None
    _auto_mask_generator = None
    _device = None
    _points_per_side = 64
    _pred_iou_thresh = 0.75

    @classmethod
    def get_device(cls):
        if cls._device is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls._device

    @classmethod
    def get_model(cls):
        if cls._model is None:
            device = cls.get_device()
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            if device == "cuda":
                SAM_CHECKPOINT = os.path.join(base_dir, "sam_vit_h_4b8939.pth")
                MODEL_TYPE = "vit_h"
            else:
                SAM_CHECKPOINT = os.path.join(base_dir, "sam_vit_b_01ec64.pth")
                MODEL_TYPE = "vit_b"
                cpu_core_count = multiprocessing.cpu_count()
                torch.set_num_threads(cpu_core_count)
                print(f"Detected {cpu_core_count} CPU cores. Configured PyTorch to use {cpu_core_count} threads.")
            cls._model = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
            cls._model.to(device)
            cls._model.eval()
        return cls._model

    @classmethod
    def get_predictor(cls):
        if cls._predictor is None:
            cls._predictor = SamPredictor(cls.get_model())
        return cls._predictor

    @classmethod
    def set_auto_mask_generator_config(cls, points_per_side, pred_iou_thresh):
        cls._points_per_side = points_per_side
        cls._pred_iou_thresh = pred_iou_thresh
        cls._auto_mask_generator = None  # Reset so new config takes effect.

    @classmethod
    def get_auto_mask_generator(cls):
        if cls._auto_mask_generator is None:
            cls._auto_mask_generator = SamAutomaticMaskGenerator(
                cls.get_model(),
                points_per_side=cls._points_per_side,
                pred_iou_thresh=cls._pred_iou_thresh,
                stability_score_thresh=0.95,
                min_mask_region_area=200
            )
        return cls._auto_mask_generator
