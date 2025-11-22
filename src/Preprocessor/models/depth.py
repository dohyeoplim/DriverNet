import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthMap:
    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device) # type: ignore
        self.model.eval()

    @torch.no_grad()
    def create_depth_map(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_width, original_height = image.size

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_height, original_width),
            mode="bicubic",
            align_corners=False,
        )

        output = prediction.squeeze().cpu().numpy()

        formatted = (output - output.min()) / (output.max() - output.min()) * 255.0
        depth_map = formatted.astype(np.uint8)

        return Image.fromarray(depth_map)
