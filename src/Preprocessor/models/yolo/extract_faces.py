import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

from .yolo_model import YOLOFaceDetector

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


def extract_face_from_image(image: Image.Image, detector: YOLOFaceDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Image.Image:
    w, h = image.size
    face_bbox = detector.detect_face(image)

    if face_bbox is None:
        x1, y1 = 0, 0
        x2 = int(w * 0.5)
        y2 = int(h * 0.5)
    else:
        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size)

    face_roi = image.crop((x1, y1, x2, y2))
    return face_roi.resize((112, 112), Image.Resampling.LANCZOS)


def extract_face_with_bbox(image: Image.Image, detector: YOLOFaceDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    w, h = image.size
    face_bbox = detector.detect_face(image)

    if face_bbox is None:
        x1, y1 = 0, 0
        x2 = int(w * 0.5)
        y2 = int(h * 0.5)
        final_bbox = (x1, y1, x2, y2)
    else:
        final_bbox = ensure_min_size(image, face_bbox, min_size)

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

from .yolo_model import YOLOFaceDetector

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


def extract_face_from_image(image: Image.Image, detector: YOLOFaceDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Image.Image:
    w, h = image.size
    face_bbox = detector.detect_face(image)

    if face_bbox is None:
        x1, y1 = 0, 0
        x2 = int(w * 0.5)
        y2 = int(h * 0.5)
    else:
        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size)

    face_roi = image.crop((x1, y1, x2, y2))
    return face_roi.resize((112, 112), Image.Resampling.LANCZOS)


def extract_face_with_bbox(image: Image.Image, detector: YOLOFaceDetector, min_size: int = 112, fallback_bbox: Tuple[int, int, int, int] = None) -> Tuple[Image.Image, Tuple[int, int, int, int], Image.Image]:
    w, h = image.size
    face_bbox = detector.detect_face(image)

    if face_bbox is None:
        x1, y1 = 0, 0
        x2 = int(w * 0.5)
        y2 = int(h * 0.5)
        final_bbox = (x1, y1, x2, y2)
    else:
        final_bbox = ensure_min_size(image, face_bbox, min_size)

    x1, y1, x2, y2 = final_bbox
    face_roi = image.crop((x1, y1, x2, y2)).resize((112, 112), Image.Resampling.LANCZOS)

    return face_roi, final_bbox, image.copy()


def extract_faces_from_train(csv_path: str, train_root: str, output_root: str, detector: YOLOFaceDetector, keypoints_csv_path: Optional[str] = None, batch_size: int = 32):
    df = pd.read_csv(csv_path)
    os.makedirs(output_root, exist_ok=True)

    keypoints_results = []
    max_keypoints = 5

    print(f"얼굴 및 키포인트 추출 중... (배치 크기: {batch_size})")

    success_count = 0
    fail_count = 0

    for batch_start in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]

        batch_images = []
        batch_data = []

        for _, row in batch_df.iterrows():
            image_path = os.path.join(train_root, row['classname'], row['img'])

            if not os.path.exists(image_path):
                fail_count += 1
                if keypoints_csv_path:
                    result_row = {
                        'classname': row['classname'],
                        'img': row['img'],
                        'subject': row['subject'],
                        'bbox_x1': None,
                        'bbox_y1': None,
                        'bbox_x2': None,
                        'bbox_y2': None,
                    }
                    for i in range(max_keypoints):
                        result_row[f'keypoint_{i}_x'] = None
                        result_row[f'keypoint_{i}_y'] = None
                    keypoints_results.append(result_row)
                continue

            try:
                image = Image.open(image_path).convert('RGB')
                batch_images.append(image)
                batch_data.append((image_path, row))
            except Exception:
                fail_count += 1
                if keypoints_csv_path:
                    result_row = {
                        'classname': row['classname'],
                        'img': row['img'],
                        'subject': row['subject'],
                        'bbox_x1': None,
                        'bbox_y1': None,
                        'bbox_x2': None,
                        'bbox_y2': None,
                    }
                    for i in range(max_keypoints):
                        result_row[f'keypoint_{i}_x'] = None
                        result_row[f'keypoint_{i}_y'] = None
                    keypoints_results.append(result_row)
                continue

        if len(batch_images) == 0:
            continue

        try:
            batch_results = detector.detect_face_with_keypoints_batch(batch_images)

            for i, (face_bbox, keypoints) in enumerate(batch_results):
                image_path, row = batch_data[i]
                image = batch_images[i]

                try:
                    if face_bbox is not None:
                        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size=112)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)
                    else:
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                    output_dir = os.path.join(output_root, row['classname'])
                    os.makedirs(output_dir, exist_ok=True)

                    output_path = os.path.join(output_dir, row['img'])
                    if output_path.lower().endswith('.png'):
                        output_path = output_path[:-4] + '.jpg'
                    if not output_path.lower().endswith('.jpg'):
                        output_path += '.jpg'

                    face_roi_resized.save(output_path, 'JPEG', quality=95)
                    success_count += 1

                    if keypoints_csv_path:
                        result_row = {
                            'classname': row['classname'],
                            'img': row['img'],
                            'subject': row['subject'],
                        }

                        if face_bbox is not None:
                            x1, y1, x2, y2 = face_bbox
                            result_row['bbox_x1'] = float(x1)
                            result_row['bbox_y1'] = float(y1)
                            result_row['bbox_x2'] = float(x2)
                            result_row['bbox_y2'] = float(y2)
                        else:
                            result_row['bbox_x1'] = None
                            result_row['bbox_y1'] = None
                            result_row['bbox_x2'] = None
                            result_row['bbox_y2'] = None

                        if keypoints is not None and len(keypoints) > 0:
                            for j in range(max_keypoints):
                                if j < len(keypoints):
                                    result_row[f'keypoint_{j}_x'] = float(keypoints[j][0])
                                    result_row[f'keypoint_{j}_y'] = float(keypoints[j][1])
                                else:
                                    result_row[f'keypoint_{j}_x'] = None
                                    result_row[f'keypoint_{j}_y'] = None
                        else:
                            for j in range(max_keypoints):
                                result_row[f'keypoint_{j}_x'] = None
                                result_row[f'keypoint_{j}_y'] = None

                        keypoints_results.append(result_row)
                except Exception as e:
                    fail_count += 1
                    # 예외 발생 시에도 좌상단 fallback 이미지 저장
                    try:
                        image = Image.open(image_path).convert('RGB')
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                        output_dir = os.path.join(output_root, row['classname'])
                        os.makedirs(output_dir, exist_ok=True)

                        output_path = os.path.join(output_dir, row['img'])
                        if output_path.lower().endswith('.png'):
                            output_path = output_path[:-4] + '.jpg'
                        if not output_path.lower().endswith('.jpg'):
                            output_path += '.jpg'

                        face_roi_resized.save(output_path, 'JPEG', quality=95)
                    except Exception:
                        pass

                    if keypoints_csv_path:
                        result_row = {
                            'classname': row['classname'],
                            'img': row['img'],
                            'subject': row['subject'],
                            'bbox_x1': None,
                            'bbox_y1': None,
                            'bbox_x2': None,
                            'bbox_y2': None,
                        }
                        for j in range(max_keypoints):
                            result_row[f'keypoint_{j}_x'] = None
                            result_row[f'keypoint_{j}_y'] = None
                        keypoints_results.append(result_row)

        except Exception as e:
            for image_path, row in batch_data:
                try:
                    image = Image.open(image_path).convert('RGB')
                    face_bbox, keypoints = detector.detect_face_with_keypoints(image)

                    if face_bbox is not None:
                        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size=112)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)
                    else:
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                    output_dir = os.path.join(output_root, row['classname'])
                    os.makedirs(output_dir, exist_ok=True)

                    output_path = os.path.join(output_dir, row['img'])
                    if output_path.lower().endswith('.png'):
                        output_path = output_path[:-4] + '.jpg'
                    if not output_path.lower().endswith('.jpg'):
                        output_path += '.jpg'

                    face_roi_resized.save(output_path, 'JPEG', quality=95)
                    success_count += 1

                    if keypoints_csv_path:
                        result_row = {
                            'classname': row['classname'],
                            'img': row['img'],
                            'subject': row['subject'],
                        }

                        if face_bbox is not None:
                            x1, y1, x2, y2 = face_bbox
                            result_row['bbox_x1'] = float(x1)
                            result_row['bbox_y1'] = float(y1)
                            result_row['bbox_x2'] = float(x2)
                            result_row['bbox_y2'] = float(y2)
                        else:
                            result_row['bbox_x1'] = None
                            result_row['bbox_y1'] = None
                            result_row['bbox_x2'] = None
                            result_row['bbox_y2'] = None

                        if keypoints is not None and len(keypoints) > 0:
                            for j in range(max_keypoints):
                                if j < len(keypoints):
                                    result_row[f'keypoint_{j}_x'] = float(keypoints[j][0])
                                    result_row[f'keypoint_{j}_y'] = float(keypoints[j][1])
                                else:
                                    result_row[f'keypoint_{j}_x'] = None
                                    result_row[f'keypoint_{j}_y'] = None
                        else:
                            for j in range(max_keypoints):
                                result_row[f'keypoint_{j}_x'] = None
                                result_row[f'keypoint_{j}_y'] = None

                        keypoints_results.append(result_row)
                except Exception:
                    fail_count += 1
                    if keypoints_csv_path:
                        result_row = {
                            'classname': row['classname'],
                            'img': row['img'],
                            'subject': row['subject'],
                            'bbox_x1': None,
                            'bbox_y1': None,
                            'bbox_x2': None,
                            'bbox_y2': None,
                        }
                        for j in range(max_keypoints):
                            result_row[f'keypoint_{j}_x'] = None
                            result_row[f'keypoint_{j}_y'] = None
                        keypoints_results.append(result_row)

    print(f"훈련 얼굴 추출 완료: {success_count}/{len(df)} 성공, {fail_count} 실패")

    if keypoints_csv_path and len(keypoints_results) > 0:
        result_df = pd.DataFrame(keypoints_results)
        result_df.to_csv(keypoints_csv_path, index=False)
        print(f"키포인트 CSV 저장 완료: {keypoints_csv_path}")
        print(f"BBox 추출 성공: {result_df['bbox_x1'].notna().sum()}개")
        print(f"Keypoints 추출 성공: {result_df['keypoint_0_x'].notna().sum()}개")


def extract_faces_from_test(test_root: str, output_root: str, detector: YOLOFaceDetector, keypoints_csv_path: Optional[str] = None, batch_size: int = 32):
    os.makedirs(output_root, exist_ok=True)

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(test_root).glob(ext))
        image_files.extend(Path(test_root).glob(ext.upper()))

    keypoints_results = []
    max_keypoints = 5

    print(f"얼굴 및 키포인트 추출 중... (배치 크기: {batch_size})")

    success_count = 0
    fail_count = 0

    # 배치 단위로 처리
    for batch_start in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]

        batch_images = []
        batch_data = []

        # 배치 데이터 준비
        for image_path in batch_files:
            try:
                image = Image.open(image_path).convert('RGB')
                batch_images.append(image)
                batch_data.append(image_path)
            except Exception:
                fail_count += 1
                if keypoints_csv_path:
                    img_filename = image_path.name
                    if img_filename.lower().endswith('.png'):
                        img_filename = img_filename[:-4] + '.jpg'
                    if not img_filename.lower().endswith('.jpg'):
                        img_filename += '.jpg'

                    result_row = {'img': img_filename}
                    result_row['bbox_x1'] = None
                    result_row['bbox_y1'] = None
                    result_row['bbox_x2'] = None
                    result_row['bbox_y2'] = None
                    for j in range(max_keypoints):
                        result_row[f'keypoint_{j}_x'] = None
                        result_row[f'keypoint_{j}_y'] = None
                    keypoints_results.append(result_row)
                continue

        if len(batch_images) == 0:
            continue

        try:
            batch_results = detector.detect_face_with_keypoints_batch(batch_images)

            for i, (face_bbox, keypoints) in enumerate(batch_results):
                image_path = batch_data[i]
                image = batch_images[i]

                try:
                    if face_bbox is not None:
                        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size=112)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)
                    else:
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                    img_filename = image_path.name
                    if img_filename.lower().endswith('.png'):
                        img_filename = img_filename[:-4] + '.jpg'
                    if not img_filename.lower().endswith('.jpg'):
                        img_filename += '.jpg'

                    face_roi_resized.save(os.path.join(output_root, img_filename), 'JPEG', quality=95)
                    success_count += 1

                    if keypoints_csv_path:
                        result_row = {'img': img_filename}

                        if face_bbox is not None:
                            x1, y1, x2, y2 = face_bbox
                            result_row['bbox_x1'] = float(x1)
                            result_row['bbox_y1'] = float(y1)
                            result_row['bbox_x2'] = float(x2)
                            result_row['bbox_y2'] = float(y2)
                        else:
                            result_row['bbox_x1'] = None
                            result_row['bbox_y1'] = None
                            result_row['bbox_x2'] = None
                            result_row['bbox_y2'] = None

                        if keypoints is not None and len(keypoints) > 0:
                            for j in range(max_keypoints):
                                if j < len(keypoints):
                                    result_row[f'keypoint_{j}_x'] = float(keypoints[j][0])
                                    result_row[f'keypoint_{j}_y'] = float(keypoints[j][1])
                                else:
                                    result_row[f'keypoint_{j}_x'] = None
                                    result_row[f'keypoint_{j}_y'] = None
                        else:
                            for j in range(max_keypoints):
                                result_row[f'keypoint_{j}_x'] = None
                                result_row[f'keypoint_{j}_y'] = None

                        keypoints_results.append(result_row)
                except Exception as e:
                    fail_count += 1
                    # 예외 발생 시에도 좌상단 fallback 이미지 저장
                    try:
                        image = Image.open(image_path).convert('RGB')
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                        img_filename = image_path.name
                        if img_filename.lower().endswith('.png'):
                            img_filename = img_filename[:-4] + '.jpg'
                        if not img_filename.lower().endswith('.jpg'):
                            img_filename += '.jpg'

                        face_roi_resized.save(os.path.join(output_root, img_filename), 'JPEG', quality=95)
                    except Exception:
                        pass

                    if keypoints_csv_path:
                        img_filename = image_path.name
                        if img_filename.lower().endswith('.png'):
                            img_filename = img_filename[:-4] + '.jpg'
                        if not img_filename.lower().endswith('.jpg'):
                            img_filename += '.jpg'

                        result_row = {'img': img_filename}
                        result_row['bbox_x1'] = None
                        result_row['bbox_y1'] = None
                        result_row['bbox_x2'] = None
                        result_row['bbox_y2'] = None
                        for j in range(max_keypoints):
                            result_row[f'keypoint_{j}_x'] = None
                            result_row[f'keypoint_{j}_y'] = None
                        keypoints_results.append(result_row)

        except Exception as e:
            # 배치 실패 시 개별 처리로 폴백
            for image_path in batch_data:
                try:
                    image = Image.open(image_path).convert('RGB')
                    face_bbox, keypoints = detector.detect_face_with_keypoints(image)

                    if face_bbox is not None:
                        x1, y1, x2, y2 = ensure_min_size(image, face_bbox, min_size=112)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)
                    else:
                        x1, y1 = 0, 0
                        x2 = int(image.size[0] * 0.5)
                        y2 = int(image.size[1] * 0.5)
                        face_roi = image.crop((x1, y1, x2, y2))
                        face_roi_resized = face_roi.resize((112, 112), Image.Resampling.LANCZOS)

                    img_filename = image_path.name
                    if img_filename.lower().endswith('.png'):
                        img_filename = img_filename[:-4] + '.jpg'
                    if not img_filename.lower().endswith('.jpg'):
                        img_filename += '.jpg'

                    face_roi_resized.save(os.path.join(output_root, img_filename), 'JPEG', quality=95)
                    success_count += 1

                    if keypoints_csv_path:
                        result_row = {'img': img_filename}

                        if face_bbox is not None:
                            x1, y1, x2, y2 = face_bbox
                            result_row['bbox_x1'] = float(x1)
                            result_row['bbox_y1'] = float(y1)
                            result_row['bbox_x2'] = float(x2)
                            result_row['bbox_y2'] = float(y2)
                        else:
                            result_row['bbox_x1'] = None
                            result_row['bbox_y1'] = None
                            result_row['bbox_x2'] = None
                            result_row['bbox_y2'] = None

                        if keypoints is not None and len(keypoints) > 0:
                            for j in range(max_keypoints):
                                if j < len(keypoints):
                                    result_row[f'keypoint_{j}_x'] = float(keypoints[j][0])
                                    result_row[f'keypoint_{j}_y'] = float(keypoints[j][1])
                                else:
                                    result_row[f'keypoint_{j}_x'] = None
                                    result_row[f'keypoint_{j}_y'] = None
                        else:
                            for j in range(max_keypoints):
                                result_row[f'keypoint_{j}_x'] = None
                                result_row[f'keypoint_{j}_y'] = None

                        keypoints_results.append(result_row)
                except Exception:
                    fail_count += 1
                    if keypoints_csv_path:
                        img_filename = image_path.name
                        if img_filename.lower().endswith('.png'):
                            img_filename = img_filename[:-4] + '.jpg'
                        if not img_filename.lower().endswith('.jpg'):
                            img_filename += '.jpg'

                        result_row = {'img': img_filename}
                        result_row['bbox_x1'] = None
                        result_row['bbox_y1'] = None
                        result_row['bbox_x2'] = None
                        result_row['bbox_y2'] = None
                        for j in range(max_keypoints):
                            result_row[f'keypoint_{j}_x'] = None
                            result_row[f'keypoint_{j}_y'] = None
                        keypoints_results.append(result_row)

    print(f"테스트 얼굴 추출 완료: {success_count}/{len(image_files)} 성공, {fail_count} 실패")

    if keypoints_csv_path and len(keypoints_results) > 0:
        result_df = pd.DataFrame(keypoints_results)
        result_df.to_csv(keypoints_csv_path, index=False)
        print(f"키포인트 CSV 저장 완료: {keypoints_csv_path}")
        print(f"BBox 추출 성공: {result_df['bbox_x1'].notna().sum()}개")
        print(f"Keypoints 추출 성공: {result_df['keypoint_0_x'].notna().sum()}개")
