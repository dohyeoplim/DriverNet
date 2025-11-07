from src.Preprocessor.core import process_one_image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SamModel, SamProcessor

def main():
    data_dir = Path("input/imgs/train")
    out_dir = Path("input/processed/train")
    out_dir.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in sorted(data_dir.glob("c*")) if d.is_dir()]

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")

    rows = []
    all_images = [(cdir, img) for cdir in class_dirs for img in cdir.glob("*.jpg")]

    use_autocast = device.type in ("cuda", "mps")

    with torch.inference_mode():
        autocast_ctx = torch.autocast(device_type=device.type, enabled=use_autocast)
        with autocast_ctx:
            for class_dir, img_file in tqdm(all_images, desc="Processing images", unit="img"):
                class_out_dir = out_dir / class_dir.name
                class_out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = process_one_image(
                        str(img_file),
                        str(class_out_dir),
                        detector=detector,
                        detector_processor=detector_processor,
                        sam_model=sam_model,
                        sam_processor=sam_processor,
                        device=device.type,
                        smooth_k=5,
                        feather_k=15,
                        bg_blur_k=25,
                        bg_darken_factor=0.5,
                    )
                    if result.get("ok", False):
                        rows.append({
                            "image_path": result.get("image_path", str(img_file)),
                            "mask_path": result.get("mask_path", ""),
                            "ok": True,
                            "error": "",
                        })
                    else:
                        rows.append({
                            "image_path": str(img_file),
                            "mask_path": "",
                            "ok": False,
                            "error": result.get("error", "unknown"),
                        })
                except Exception as e:
                    rows.append({
                        "image_path": str(img_file),
                        "mask_path": "",
                        "ok": False,
                        "error": repr(e),
                    })

    result_df = pd.DataFrame(rows, columns=["image_path", "mask_path", "ok", "error"])
    result_df.to_csv(out_dir / "processing_results.csv", index=False)

if __name__ == "__main__":
    main()
