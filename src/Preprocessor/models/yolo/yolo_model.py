import torch
from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import os


class YOLOFaceDetector:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        try:
            from ultralytics import YOLO

            if model_path:
                self.model = YOLO(model_path)
            else:
                local_model_path = os.path.join(os.path.dirname(__file__), 'yolov8n-face-keypoints.pt')
                if os.path.exists(local_model_path):
                    self.model = YOLO(local_model_path)
                else:
                    self.model = YOLO('yolov8n.pt')
        except ImportError:
            self.model = None
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.model = None

    def detect_face(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        if self.model is None:
            return None

        try:
            img_array = np.array(image)
            results = self.model(img_array, verbose=False)

            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                w, h = image.size
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                x2_expanded = min(int(x1 + bbox_width * 1.2), w)
                y2_expanded = min(int(y1 + bbox_height * 1.2), h)

                return (x1, y1, x2_expanded, y2_expanded)
        except Exception:
            pass

        return None

    def detect_face_with_keypoints(self, image: Image.Image) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        if self.model is None:
            return None, None

        try:
            img_array = np.array(image)
            results = self.model(img_array, verbose=False)

            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                box = results[0].boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                w, h = image.size
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                x2_expanded = min(int(x1 + bbox_width * 1.2), w)
                y2_expanded = min(int(y1 + bbox_height * 1.2), h)

                bbox = (x1, y1, x2_expanded, y2_expanded)

                keypoints = None
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None and len(results[0].keypoints) > 0:
                    keypoints = results[0].keypoints.xy.cpu().numpy()[0]

                return bbox, keypoints
        except Exception:
            pass

        return None, None

    def detect_face_with_keypoints_batch(self, images: List[Image.Image]) -> List[Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]]:
        if self.model is None:
            return [(None, None)] * len(images)

        try:
            img_arrays = [np.array(img) for img in images]

            results = self.model(img_arrays, verbose=False)

            batch_results = []
            for i, result in enumerate(results):
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    box = result.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                    w, h = images[i].size
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    x2_expanded = min(int(x1 + bbox_width * 1.2), w)
                    y2_expanded = min(int(y1 + bbox_height * 1.2), h)

                    bbox = (x1, y1, x2_expanded, y2_expanded)

                    keypoints = None
                    if hasattr(result, 'keypoints') and result.keypoints is not None and len(result.keypoints) > 0:
                        keypoints = result.keypoints.xy.cpu().numpy()[0]

                    batch_results.append((bbox, keypoints))
                else:
                    batch_results.append((None, None))

            return batch_results
        except Exception:
            return [(None, None)] * len(images)


class YOLOHandDetector:
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        try:
            from ultralytics import YOLO

            if model_path:
                self.model = YOLO(model_path)
            else:
                local_model_path = os.path.join(os.path.dirname(__file__), 'hand_yolov8s.pt')
                if os.path.exists(local_model_path):
                    self.model = YOLO(local_model_path)
                else:
                    print("손 감지 모델을 찾을 수 없습니다.")
                    self.model = None
        except ImportError:
            self.model = None
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.model = None

    def detect_hand(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        if self.model is None:
            return None

        try:
            img_array = np.array(image)
            results = self.model(img_array, verbose=False)

            if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                boxes = results[0].boxes

                if len(boxes) > 0:
                    boxes_xyxy = boxes.xyxy.cpu().numpy()

                    if len(boxes_xyxy) == 1:
                        box = boxes_xyxy[0]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    else:
                        leftmost_idx = np.argmin(boxes_xyxy[:, 0])
                        box = boxes_xyxy[leftmost_idx]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                    w, h = image.size

                    left_threshold = w * 0.2
                    if x1 < left_threshold:
                        return None

                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    x1_expanded = max(0, int(x1 - bbox_width * 0.2))
                    y1_expanded = max(0, int(y1 - bbox_height * 0.2))
                    x2_expanded = min(int(x1 + bbox_width * 1.2), w)
                    y2_expanded = min(int(y1 + bbox_height * 1.2), h)

                    return (x1_expanded, y1_expanded, x2_expanded, y2_expanded)
        except Exception:
            pass

        return None
