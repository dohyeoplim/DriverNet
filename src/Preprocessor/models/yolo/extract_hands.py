import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

from .yolo_model import YOLOHandDetector


def ensure_min_size(image: Image.Image, bbox: Tuple[int, int, int, int], min_size: int = 112) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    w, h = image.size

    if x2 - x1 < min_size:
        x2 = min(x1 + min_size, w)
        if x2 - x1 < min_size:
            x1 = max(0, x2 - min_size)

    if y2 - y1 < min_size:
        y2 = min(y1 + min_size, h)
        if y2 - y1 < min_size:
            y1 = max(0, y2 - min_size)

    return (x1, y1, x2, y2)


def extract_hand_from_image(image: Image.Image, detector: YOLOHandDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Image.Image:
    w, h = image.size
    hand_bbox = detector.detect_hand(image)

    if hand_bbox is None:
        return Image.new('RGB', (112, 112), color=(0, 0, 0))

    x1, y1, x2, y2 = ensure_min_size(image, hand_bbox, min_size)
    hand_roi = image.crop((x1, y1, x2, y2))
    return hand_roi.resize((112, 112), Image.Resampling.LANCZOS)


def extract_hand_with_bbox(image: Image.Image, detector: YOLOHandDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    w, h = image.size
    hand_bbox = detector.detect_hand(image)

    if hand_bbox is None:
        black_image = Image.new('RGB', (112, 112), color=(0, 0, 0))
        return black_image, (0, 0, 112, 112), image.copy()

    final_bbox = ensure_min_size(image, hand_bbox, min_size)
    x1, y1, x2, y2 = final_bbox
    hand_roi = image.crop((x1, y1, x2, y2)).resize((112, 112), Image.Resampling.LANCZOS)

    return hand_roi, final_bbox, image.copy()


def extract_hands_from_train(csv_path: str, train_root: str, output_root: str, detector: YOLOHandDetector, failed_csv_path: Optional[str] = None):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    print("1단계: 손 감지 시도 및 성공 케이스 수집...")
    successful_bboxes = []
    failed_cases = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 1: Detection"):
        image_path = os.path.join(train_root, row['classname'], row['img'])

        if not os.path.exists(image_path):
            failed_cases.append({
                'classname': row['classname'],
                'img': row['img'],
                'reason': 'file_not_found'
            })
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            hand_bbox = detector.detect_hand(image)

            if hand_bbox is not None:
                successful_bboxes.append(hand_bbox)
            else:
                failed_cases.append({
                    'classname': row['classname'],
                    'img': row['img'],
                    'reason': 'detection_failed'
                })
        except Exception as e:
            failed_cases.append({
                'classname': row['classname'],
                'img': row['img'],
                'reason': f'error: {str(e)}'
            })

    if len(successful_bboxes) > 0:
        avg_bbox = (
            int(np.mean([b[0] for b in successful_bboxes])),
            int(np.mean([b[1] for b in successful_bboxes])),
            int(np.mean([b[2] for b in successful_bboxes])),
            int(np.mean([b[3] for b in successful_bboxes]))
        )
        print(f"평균 BBox 좌표 계산 완료: {avg_bbox} (성공 케이스: {len(successful_bboxes)}개)")
    else:
        avg_bbox = None
        print("성공한 케이스가 없습니다. 기본 좌표 사용")

    print(f"\n2단계: 손 추출 및 저장 (실패 케이스: {len(failed_cases)}개)...")

    success_count = 0
    fail_count = 0
    extraction_failed_cases = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pass 2: Extraction"):
        image_path = os.path.join(train_root, row['classname'], row['img'])

        if not os.path.exists(image_path):
            fail_count += 1
            continue

        try:
            image = Image.open(image_path).convert('RGB')
            hand_roi = extract_hand_from_image(image, detector)

            output_dir = os.path.join(output_root, row['classname'])
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, row['img'])
            if output_path.lower().endswith('.png'):
                output_path = output_path[:-4] + '.jpg'
            if not output_path.lower().endswith('.jpg'):
                output_path += '.jpg'

            hand_roi.save(output_path, 'JPEG', quality=95)
            success_count += 1
        except Exception as e:
            fail_count += 1
            extraction_failed_cases.append({
                'classname': row['classname'],
                'img': row['img'],
                'reason': f'extraction_error: {str(e)}'
            })

    all_failed_cases = failed_cases + extraction_failed_cases

    if len(all_failed_cases) > 0:
        failed_df = pd.DataFrame(all_failed_cases)
        if failed_csv_path is None:
            failed_csv_path = os.path.join(output_root, 'failed_cases.csv')
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"실패 케이스 CSV 저장 완료: {failed_csv_path} ({len(all_failed_cases)}개)")

    print(f"훈련 손 추출 완료: {success_count}/{len(df)} 성공, {fail_count} 실패")


def extract_hands_from_test(test_root: str, output_root: str, detector: YOLOHandDetector, failed_csv_path: Optional[str] = None):
    os.makedirs(output_root, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(test_root).glob(ext))
        image_files.extend(Path(test_root).glob(ext.upper()))

    print("1단계: 손 감지 시도 및 성공 케이스 수집...")
    successful_bboxes = []
    failed_cases = []

    for image_path in tqdm(image_files, desc="Pass 1: Detection"):
        try:
            image = Image.open(image_path).convert('RGB')
            hand_bbox = detector.detect_hand(image)

            if hand_bbox is not None:
                successful_bboxes.append(hand_bbox)
            else:
                failed_cases.append({
                    'img': image_path.name,
                    'reason': 'detection_failed'
                })
        except Exception as e:
            failed_cases.append({
                'img': image_path.name,
                'reason': f'error: {str(e)}'
            })

    if len(successful_bboxes) > 0:
        avg_bbox = (
            int(np.mean([b[0] for b in successful_bboxes])),
            int(np.mean([b[1] for b in successful_bboxes])),
            int(np.mean([b[2] for b in successful_bboxes])),
            int(np.mean([b[3] for b in successful_bboxes]))
        )
        print(f"평균 BBox 좌표 계산 완료: {avg_bbox} (성공 케이스: {len(successful_bboxes)}개)")
    else:
        avg_bbox = None
        print("성공한 케이스가 없습니다. 기본 좌표 사용")

    print(f"\n2단계: 손 추출 및 저장 (실패 케이스: {len(failed_cases)}개)...")

    success_count = 0
    fail_count = 0
    extraction_failed_cases = []

    for image_path in tqdm(image_files, desc="Pass 2: Extraction"):
        try:
            image = Image.open(image_path).convert('RGB')
            hand_roi = extract_hand_from_image(image, detector)

            img_filename = image_path.name
            if img_filename.lower().endswith('.png'):
                img_filename = img_filename[:-4] + '.jpg'
            if not img_filename.lower().endswith('.jpg'):
                img_filename += '.jpg'

            hand_roi.save(os.path.join(output_root, img_filename), 'JPEG', quality=95)
            success_count += 1
        except Exception as e:
            fail_count += 1
            extraction_failed_cases.append({
                'img': image_path.name,
                'reason': f'extraction_error: {str(e)}'
            })

    all_failed_cases = failed_cases + extraction_failed_cases

    if len(all_failed_cases) > 0:
        failed_df = pd.DataFrame(all_failed_cases)
        if failed_csv_path is None:
            failed_csv_path = os.path.join(output_root, 'failed_cases.csv')
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"실패 케이스 CSV 저장 완료: {failed_csv_path} ({len(all_failed_cases)}개)")

    print(f"테스트 손 추출 완료: {success_count}/{len(image_files)} 성공, {fail_count} 실패")


def visualize_hand_extraction(
    image_paths: List[str],
    detector: YOLOHandDetector,
    fallback_bbox: Optional[Tuple[int, int, int, int]] = None,
    save_path: Optional[str] = None
):
    num_samples = min(5, len(image_paths))
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for idx in range(num_samples):
        image_path = image_paths[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            hand_bbox = detector.detect_hand(image)

            if hand_bbox is None and fallback_bbox is not None:
                hand_roi, final_bbox, _ = extract_hand_with_bbox(image, detector, fallback_bbox=fallback_bbox)
                detected = False
            else:
                hand_roi, final_bbox, _ = extract_hand_with_bbox(image, detector, fallback_bbox=fallback_bbox)
                detected = hand_bbox is not None

            original_with_bbox = image.copy()
            from PIL import ImageDraw
            draw = ImageDraw.Draw(original_with_bbox)
            x1, y1, x2, y2 = final_bbox
            draw.rectangle([x1, y1, x2, y2], outline='red' if detected else 'yellow', width=3)

            axes[0, idx].imshow(original_with_bbox)
            axes[0, idx].set_title(f'Original ({"Detected" if detected else "Fallback"})', fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')

            axes[1, idx].imshow(hand_roi)
            axes[1, idx].set_title(f'Extracted Hand', fontsize=12, fontweight='bold')
            axes[1, idx].axis('off')

        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center')
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: {save_path}")
    else:
        plt.show()

    plt.close()

