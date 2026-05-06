import numpy as np
from ultralytics import YOLO

from src.utils import order_points, perspective_crop


class DocumentDetector:
    def __init__(self, model_path: str = "runs/obb/document_obb/weights/best.pt"):
        self.model = YOLO(model_path, task="obb")

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        """Detect documents and return deskewed crops sorted by confidence."""
        results = self.model.predict(
            source=image,
            device="cpu",
            imgsz=640,
            conf=0.25,
            verbose=False,
        )

        if not results or results[0].obb is None or len(results[0].obb) == 0:
            return []

        r = results[0]
        # Sort detections by confidence (descending)
        confs = r.obb.conf.cpu().numpy()
        indices = np.argsort(-confs)

        crops = []
        for idx in indices:
            quad = r.obb.xyxyxyxy[idx].cpu().numpy().reshape(4, 2)
            crop = perspective_crop(image, quad)
            if crop.size > 0:
                crops.append(crop)

        return crops
