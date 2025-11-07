import cv2
import numpy as np

def apply_background_effects(
    cv_image: np.ndarray,
    mask_255: np.ndarray,
    bg_blur_k: int,
    bg_darken_factor: float
) -> np.ndarray:

    background = cv_image.copy()

    if bg_blur_k > 0:
        bg_blur_k = bg_blur_k if bg_blur_k % 2 == 1 else bg_blur_k + 1
        background = cv2.GaussianBlur(background, (bg_blur_k, bg_blur_k), 0)

    if bg_darken_factor < 1.0:
        background = (background.astype(np.float32) * bg_darken_factor)
        background = np.clip(background, 0, 255)

    mask_norm = mask_255[..., None].astype(np.float32) / 255.0
    inv_mask_norm = 1.0 - mask_norm

    processed_image = (cv_image.astype(np.float32) * mask_norm) + (background.astype(np.float32) * inv_mask_norm)

    return np.clip(processed_image, 0, 255).astype(np.uint8)
