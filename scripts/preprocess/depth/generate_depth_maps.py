from src.Preprocessor.models.depth import DepthMap
import argparse
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from pathlib import Path

def process_one_image(image_path: str, output_dir: str, depth_creator: DepthMap):
    try:
        relative_path = Path(image_path).parent.relative_to(Path(output_dir).parent.parent)

        output_path = Path(output_dir) / relative_path
        output_path.mkdir(parents=True, exist_ok=True)

        base_name = Path(image_path).stem
        output_filename = output_path / f"{base_name}_depth.png"

        image = Image.open(image_path)

        depth_map = depth_creator.create_depth_map(image)

        depth_map.save(output_filename)

        return {"ok": True, "path": str(output_filename)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Batch process images to generate depth maps.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing input images (e.g., 'input/train').")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save generated depth maps.")
    args = parser.parse_args()

    depth_creator = DepthMap()

    search_pattern = os.path.join(args.input_dir, f"**/*.jpg")
    image_files = glob(search_pattern, recursive=True)

    if not image_files:
        print(f"No images found for pattern: {search_pattern}")
        return

    print(f"Found {len(image_files)} images.")
    fail_count = 0

    for f in tqdm(image_files, desc="Processing"):
        try:
            result = process_one_image(f, args.output_dir, depth_creator)
            if not result["ok"]:
                print(f"Failed to process {f}: {result['error']}")
                fail_count += 1
        except Exception as e:
            print(f"Critical error on {f}: {e}")
            fail_count += 1

    print(f"\nSuccessfully processed: {len(image_files) - fail_count}")
    print(f"Failed to process: {fail_count}")

if __name__ == "__main__":
    main()
