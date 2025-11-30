from ultralytics import YOLO
import numpy as np
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List

model = YOLO(os.path.join(os.path.dirname(__file__), 'yolov8n-face-keypoints.pt'))


def get_face_data(image_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    results = model(image_path, verbose=False)

    if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        box = results[0].boxes[0].xyxy.cpu().numpy()[0]

        if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]
            return box, keypoints
        else:
            return box, None

    return None, None


def visualize_face_keypoints(
    image_path: str,
    box: Optional[np.ndarray] = None,
    keypoints: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    image = Image.open(image_path).convert('RGB')

    if box is None or keypoints is None:
        box, keypoints = get_face_data(image_path)

    if box is None:
        print("얼굴을 감지하지 못했습니다.")
        return

    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

    if keypoints is not None and len(keypoints) > 0:
        for point in keypoints:
            px, py = int(point[0]), int(point[1])
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill='blue', outline='blue')

        if len(keypoints) >= 5:
            left_eye = keypoints[0]
            right_eye = keypoints[1]
            nose = keypoints[2]
            left_mouth = keypoints[3]
            right_mouth = keypoints[4]

            draw.line([left_eye[0], left_eye[1], right_eye[0], right_eye[1]], fill='green', width=2)
            draw.line([left_eye[0], left_eye[1], nose[0], nose[1]], fill='green', width=2)
            draw.line([right_eye[0], right_eye[1], nose[0], nose[1]], fill='green', width=2)
            draw.line([nose[0], nose[1], left_mouth[0], left_mouth[1]], fill='green', width=2)
            draw.line([nose[0], nose[1], right_mouth[0], right_mouth[1]], fill='green', width=2)
            draw.line([left_mouth[0], left_mouth[1], right_mouth[0], right_mouth[1]], fill='green', width=2)

    face_roi = image.crop((x1, y1, x2, y2))
    face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(annotated_image)
    axes[0].set_title('Face with Keypoints', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    if keypoints is not None:
        face_with_keypoints = face_roi_resized.copy()
        draw_roi = ImageDraw.Draw(face_with_keypoints)

        scale_x = 112 / (x2 - x1)
        scale_y = 112 / (y2 - y1)

        for point in keypoints:
            px = int((point[0] - x1) * scale_x)
            py = int((point[1] - y1) * scale_y)
            if 0 <= px < 112 and 0 <= py < 112:
                draw_roi.ellipse([px - 2, py - 2, px + 2, py + 2], fill='blue', outline='blue')

        axes[1].imshow(face_with_keypoints)
    else:
        axes[1].imshow(face_roi_resized)

    axes[1].set_title('Extracted Face ROI', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def batch_visualize_keypoints(
    image_dir: str,
    num_samples: int = 5,
    save_path: Optional[str] = None
):
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(image_dir).glob(ext))
        image_files.extend(Path(image_dir).glob(ext.upper()))

    if len(image_files) == 0:
        print(f"이미지를 찾을 수 없습니다: {image_dir}")
        return

    image_files = image_files[:num_samples]
    image_paths = [str(f) for f in image_files]

    num_samples = min(num_samples, len(image_paths))
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    if num_samples == 1:
        axes = axes.reshape(2, 1)

    for idx, image_path in enumerate(image_paths):
        try:
            box, keypoints = get_face_data(image_path)

            if box is None:
                axes[0, idx].text(0.5, 0.5, 'No face detected', ha='center', va='center', fontsize=12)
                axes[0, idx].axis('off')
                axes[1, idx].axis('off')
                continue

            image = Image.open(image_path).convert('RGB')
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

            if keypoints is not None and len(keypoints) > 0:
                for point in keypoints:
                    px, py = int(point[0]), int(point[1])
                    draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill='blue', outline='blue')

            axes[0, idx].imshow(annotated_image)
            title = 'Face with Keypoints' if keypoints is not None else 'Face (No Keypoints)'
            axes[0, idx].set_title(title, fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')

            face_roi = image.crop((x1, y1, x2, y2))
            face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

            if keypoints is not None:
                face_with_keypoints = face_roi_resized.copy()
                draw_roi = ImageDraw.Draw(face_with_keypoints)

                scale_x = 112 / (x2 - x1)
                scale_y = 112 / (y2 - y1)

                for point in keypoints:
                    px = int((point[0] - x1) * scale_x)
                    py = int((point[1] - y1) * scale_y)
                    if 0 <= px < 112 and 0 <= py < 112:
                        draw_roi.ellipse([px - 2, py - 2, px + 2, py + 2], fill='blue', outline='blue')

                axes[1, idx].imshow(face_with_keypoints)
            else:
                axes[1, idx].imshow(face_roi_resized)

            axes[1, idx].set_title('Extracted Face ROI', fontsize=12, fontweight='bold')
            axes[1, idx].axis('off')

        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center', fontsize=10)
            axes[0, idx].axis('off')
            axes[1, idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: {save_path}")
    else:
        plt.show()

    plt.close()


def extract_keypoints_to_csv(
    csv_path: str,
    train_root: str,
    output_csv_path: str,
    max_keypoints: int = 5
):
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(csv_path)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keypoints"):
        classname = row['classname']
        img_filename = row['img']
        subject_id = row['subject']

        image_path = os.path.join(train_root, classname, img_filename)

        result_row = {
            'classname': classname,
            'img': img_filename,
            'subject': subject_id,
        }

        if not os.path.exists(image_path):
            result_row['bbox_x1'] = None
            result_row['bbox_y1'] = None
            result_row['bbox_x2'] = None
            result_row['bbox_y2'] = None
            for i in range(max_keypoints):
                result_row[f'keypoint_{i}_x'] = None
                result_row[f'keypoint_{i}_y'] = None
            results.append(result_row)
            continue

        try:
            box, keypoints = get_face_data(image_path)

            if box is not None:
                result_row['bbox_x1'] = float(box[0])
                result_row['bbox_y1'] = float(box[1])
                result_row['bbox_x2'] = float(box[2])
                result_row['bbox_y2'] = float(box[3])
            else:
                result_row['bbox_x1'] = None
                result_row['bbox_y1'] = None
                result_row['bbox_x2'] = None
                result_row['bbox_y2'] = None

            if keypoints is not None and len(keypoints) > 0:
                for i in range(max_keypoints):
                    if i < len(keypoints):
                        result_row[f'keypoint_{i}_x'] = float(keypoints[i][0])
                        result_row[f'keypoint_{i}_y'] = float(keypoints[i][1])
                    else:
                        result_row[f'keypoint_{i}_x'] = None
                        result_row[f'keypoint_{i}_y'] = None
            else:
                for i in range(max_keypoints):
                    result_row[f'keypoint_{i}_x'] = None
                    result_row[f'keypoint_{i}_y'] = None

            results.append(result_row)

        except Exception as e:
            print(f"오류 발생 ({image_path}): {e}")
            result_row['bbox_x1'] = None
            result_row['bbox_y1'] = None
            result_row['bbox_x2'] = None
            result_row['bbox_y2'] = None
            for i in range(max_keypoints):
                result_row[f'keypoint_{i}_x'] = None
                result_row[f'keypoint_{i}_y'] = None
            results.append(result_row)
            continue

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Keypoints CSV 저장 완료: {output_csv_path}")
    print(f"총 {len(results)}개 이미지 처리 완료")
    print(f"BBox 추출 성공: {result_df['bbox_x1'].notna().sum()}개")
    print(f"Keypoints 추출 성공: {result_df['keypoint_0_x'].notna().sum()}개")


def extract_test_keypoints_to_csv(
    test_root: str,
    output_csv_path: str,
    max_keypoints: int = 5
):
    import pandas as pd
    from tqdm import tqdm

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(test_root).glob(ext))
        image_files.extend(Path(test_root).glob(ext.upper()))

    image_files = sorted(image_files)

    if len(image_files) == 0:
        print(f"이미지를 찾을 수 없습니다: {test_root}")
        return

    results = []

    for image_path in tqdm(image_files, desc="Extracting test keypoints"):
        img_filename = image_path.name

        result_row = {
            'img': img_filename,
        }

        try:
            box, keypoints = get_face_data(str(image_path))

            if box is not None:
                result_row['bbox_x1'] = float(box[0])
                result_row['bbox_y1'] = float(box[1])
                result_row['bbox_x2'] = float(box[2])
                result_row['bbox_y2'] = float(box[3])
            else:
                result_row['bbox_x1'] = None
                result_row['bbox_y1'] = None
                result_row['bbox_x2'] = None
                result_row['bbox_y2'] = None

            if keypoints is not None and len(keypoints) > 0:
                for i in range(max_keypoints):
                    if i < len(keypoints):
                        result_row[f'keypoint_{i}_x'] = float(keypoints[i][0])
                        result_row[f'keypoint_{i}_y'] = float(keypoints[i][1])
                    else:
                        result_row[f'keypoint_{i}_x'] = None
                        result_row[f'keypoint_{i}_y'] = None
            else:
                for i in range(max_keypoints):
                    result_row[f'keypoint_{i}_x'] = None
                    result_row[f'keypoint_{i}_y'] = None

            results.append(result_row)

        except Exception as e:
            print(f"오류 발생 ({image_path}): {e}")
            result_row['bbox_x1'] = None
            result_row['bbox_y1'] = None
            result_row['bbox_x2'] = None
            result_row['bbox_y2'] = None
            for i in range(max_keypoints):
                result_row[f'keypoint_{i}_x'] = None
                result_row[f'keypoint_{i}_y'] = None
            results.append(result_row)
            continue

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)
    print(f"Test Keypoints CSV 저장 완료: {output_csv_path}")
    print(f"총 {len(results)}개 이미지 처리 완료")
    print(f"BBox 추출 성공: {result_df['bbox_x1'].notna().sum()}개")
    print(f"Keypoints 추출 성공: {result_df['keypoint_0_x'].notna().sum()}개")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_image_path = "data/imgs/train/c9/img_55.jpg"
        if os.path.exists(test_image_path):
            print("얼굴 데이터 추출 중...")
            box, keypoints = get_face_data(test_image_path)

            if box is not None:
                print(f"Bounding Box: {box}")
                if keypoints is not None:
                    print(f"Keypoints shape: {keypoints.shape}")
                    print(f"Keypoints:\n{keypoints}")
                else:
                    print("Keypoints를 추출할 수 없습니다.")

            print("\n시각화 중...")
            visualize_face_keypoints(test_image_path, box, keypoints)
        else:
            print(f"테스트 이미지를 찾을 수 없습니다: {test_image_path}")
    elif len(sys.argv) > 1 and sys.argv[1] == 'extract_test':
        test_root = "data/imgs/test"
        output_csv_path = "data/test_face_keypoints.csv"

        extract_test_keypoints_to_csv(test_root, output_csv_path)
    else:
        print("Train 데이터 추출 시작...")
        csv_path = "data/driver_imgs_list.csv"
        train_root = "data/imgs/train"
        output_csv_path = "data/train_face_keypoints.csv"

        extract_keypoints_to_csv(csv_path, train_root, output_csv_path)

        print("\nTest 데이터 추출 시작...")
        test_root = "data/imgs/test"
        test_output_csv_path = "data/test_face_keypoints.csv"

        extract_test_keypoints_to_csv(test_root, test_output_csv_path)

        print("\n모든 추출 완료!")
