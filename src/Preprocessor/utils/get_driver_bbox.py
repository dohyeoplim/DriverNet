import torch
from PIL import Image
from typing import Optional, Any

def get_driver_bbox(
    image: Image.Image,
    detector: Any,
    detector_processor: Any,
    device: str,
    threshold: float = 0.5
) -> Optional[list]:

    inputs_detector = detector_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_detector = detector(**inputs_detector)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = detector_processor.post_process_object_detection(
        outputs_detector, threshold=threshold, target_sizes=target_sizes
    )[0]

    best_box = None
    best_score = -1.0
    person_label_id = 1

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if label == person_label_id and score > best_score:
            best_score = score
            best_box = box.cpu().numpy().tolist()

    return best_box
