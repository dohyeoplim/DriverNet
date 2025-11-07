import os
import cv2
import numpy as np
from PIL import Image
from typing import Any, Dict
from src.Preprocessor.utils import get_driver_bbox, segment_driver_mask, post_process_mask, apply_background_effects

def process_one_image(
    image_path: str,
    out_dir: str,
    detector: Any,
    detector_processor: Any,
    sam_model: Any,
    sam_processor: Any,
    device: str,
    smooth_k: int,
    feather_k: int,
    bg_blur_k: int,
    bg_darken_factor: float
) -> Dict[str, Any]:

    os.makedirs(out_dir, exist_ok=True)

    try:
        image_pil = Image.open(image_path).convert("RGB")
        cv_image_rgb = np.array(image_pil)
    except Exception as e:
        return {"ok": False, "error": f"read_fail: {e}", "file": image_path}

    best_box = get_driver_bbox(image_pil, detector, detector_processor, device)

    if best_box is None:
        return {"ok": False, "error": "no_person_found", "file": image_path}

    binary_mask = segment_driver_mask(image_pil, best_box, sam_model, sam_processor, device)

    alpha_mask = post_process_mask(binary_mask, smooth_k, feather_k)

    processed_image = apply_background_effects(cv_image_rgb, alpha_mask, bg_blur_k, bg_darken_factor)

    base = os.path.splitext(os.path.basename(image_path))[0]
    # mask_path = os.path.join(out_dir, f"{base}_mask.png")
    final_image_path = os.path.join(out_dir, f"{base}_processed.png")

    # cv2.imwrite(mask_path, alpha_mask)
    cv2.imwrite(final_image_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

    return {"ok": True, "processed_image": final_image_path}
