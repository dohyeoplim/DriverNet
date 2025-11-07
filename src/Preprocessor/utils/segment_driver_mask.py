import numpy as np
import torch
from PIL import Image
from typing import Any

def segment_driver_mask(
    image: Image.Image,
    bbox: list,
    sam_model: Any,
    sam_processor: Any,
    device: str
) -> np.ndarray:

    inputs_sam = sam_processor(
        image,
        input_boxes=[[bbox]],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs_sam = sam_model(**inputs_sam)

    best_mask_idx = torch.argmax(outputs_sam.iou_scores).item()

    mask = sam_processor.post_process_masks(
        outputs_sam.pred_masks,
        inputs_sam["original_sizes"].cpu(),
        inputs_sam["reshaped_input_sizes"].cpu()
    )[0][0, best_mask_idx, :, :]

    return (mask > 0).cpu().numpy().astype(np.uint8)
