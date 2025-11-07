import cv2
import numpy as np

def post_process_mask(
    mask: np.ndarray,
    smooth_k: int,
    feather_k: int
) -> np.ndarray:

    mask_255 = (mask * 255).astype(np.uint8)

    if smooth_k > 0:
        smooth_k = smooth_k if smooth_k % 2 == 1 else smooth_k + 1
        mask_255 = cv2.medianBlur(mask_255, smooth_k)

    if feather_k > 0:
        feather_k = feather_k if feather_k % 2 == 1 else feather_k + 1
        mask_255 = cv2.GaussianBlur(mask_255, (feather_k, feather_k), 0)

    return mask_255
