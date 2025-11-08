from src.Preprocessor.core import process_one_image
import argparse
import os
from glob import glob
from tqdm import tqdm
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from transformers import SamModel, SamProcessor

def main():
    parser = argparse.ArgumentParser(description="Batch process images using Detector + SAM.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save processed masks.")
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--feather_k", type=int, default=15)
    parser.add_argument("--bg_blur_k", type=int, default=25)
    parser.add_argument("--bg_darken_factor", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    detector = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device) # pyright: ignore[reportArgumentType]

    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device) # pyright: ignore[reportArgumentType]

    print(f"Smoothing kernel: {args.smooth_k}, Feathering kernel: {args.feather_k}")

    search_pattern = os.path.join(args.input_dir, f"**/*.jpg")
    image_files = glob(search_pattern, recursive=True)

    if not image_files:
        print(f"No images found for pattern: {search_pattern}")
        return

    print(f"Found {len(image_files)} images.")
    fail_count = 0

    for f in tqdm(image_files, desc="Processing"):
        try:
            result = process_one_image(
                f,
                args.output_dir,
                detector,
                detector_processor,
                sam_model,
                sam_processor,
                device.type,
                args.smooth_k,
                args.feather_k,
                args.bg_blur_k,
                args.bg_darken_factor,
            )
            if not result["ok"]:
                print(f"Failed to process {f}: {result['error']}")
                fail_count += 1
        except Exception as e:
            print(f"Critical error on {f}: {e}")
            fail_count += 1

    print(f"\nSuccessfully processed: {len(image_files) - fail_count}")
    print(f"\nFailed to process: {fail_count}")

if __name__ == "__main__":
    main()
