import time
from dataclasses import dataclass

from src.document_detector import DocumentDetector
from src.mrz_parser import MRZResult, parse_mrz
from src.utils import load_image


@dataclass
class TimedResult:
    result: MRZResult | None
    total_s: float
    yolo_s: float
    ocr_s: float


class PassportReaderPipeline:
    def __init__(
        self,
        yolo_model_path: str = "runs/obb/document_obb/weights/best.pt",
        ocr_det_model: str = "models/det",
        ocr_rec_model: str = "models/rec",
        engine: str = "paddleocr",
        max_image_dim: int | None = None,
    ):
        self.engine = engine
        self.max_image_dim = max_image_dim
        self.detector = DocumentDetector(yolo_model_path)

        if engine in ("fastmrz", "hybrid"):
            from src.mrz_fastmrz import FastMRZReader
            self.fastmrz_reader = FastMRZReader()
        if engine in ("paddleocr", "hybrid"):
            from src.mrz_ocr import MRZReader
            self.mrz_reader = MRZReader(ocr_det_model, ocr_rec_model)

    def process(self, image_path: str, max_dim: int | None = None) -> MRZResult | None:
        """Process a single image and return parsed MRZ data.

        If max_dim is provided, it overrides the instance-level max_image_dim
        for this call only (per-request downscale cap).
        """
        effective_max = max_dim if max_dim is not None else self.max_image_dim
        if self.engine == "fastmrz":
            return self._process_fastmrz(image_path, effective_max)
        return self._process_paddleocr(image_path, effective_max)

    def process_timed(self, image_path: str) -> TimedResult:
        """Process with timing breakdown (YOLO vs OCR)."""
        t_start = time.perf_counter()
        image = load_image(image_path, max_dim=self.max_image_dim)

        # YOLO detection
        t_yolo_start = time.perf_counter()
        documents = self.detector.detect(image)
        t_yolo = time.perf_counter() - t_yolo_start

        # OCR + parsing
        t_ocr_start = time.perf_counter()
        result = None

        if self.engine == "fastmrz":
            if documents:
                for doc_crop in documents:
                    result = self.fastmrz_reader.read_mrz(doc_crop)
                    if result:
                        break
            if not result:
                result = self.fastmrz_reader.read_from_path(image_path)
        elif self.engine == "hybrid":
            # Try PaddleOCR first (higher overall accuracy)
            paddle_result = None
            if documents:
                for doc_crop in documents:
                    mrz_lines = self.mrz_reader.read_mrz(doc_crop)
                    if mrz_lines:
                        paddle_result = parse_mrz(mrz_lines)
                        if paddle_result and paddle_result.valid:
                            break

            # Keep paddle if valid; else try FastMRZ and prefer its valid result
            if paddle_result and paddle_result.valid:
                result = paddle_result
            else:
                fastmrz_result = None
                if documents:
                    for doc_crop in documents:
                        fastmrz_result = self.fastmrz_reader.read_mrz(doc_crop)
                        if fastmrz_result:
                            break
                if not fastmrz_result:
                    fastmrz_result = self.fastmrz_reader.read_from_path(image_path)

                # Prefer valid fastmrz over invalid paddle; else keep whichever has result
                if fastmrz_result and fastmrz_result.valid:
                    result = fastmrz_result
                else:
                    result = paddle_result or fastmrz_result
        else:
            sources = list(documents) if documents else []
            sources.append(image)
            result = self._best_from_sources(sources)

        t_ocr = time.perf_counter() - t_ocr_start
        t_total = time.perf_counter() - t_start

        return TimedResult(result=result, total_s=t_total, yolo_s=t_yolo, ocr_s=t_ocr)

    def _process_fastmrz(self, image_path: str, max_dim: int | None = None) -> MRZResult | None:
        """Process using FastMRZ engine."""
        image = load_image(image_path, max_dim=max_dim)

        # Try YOLO detection → crop → FastMRZ on crop
        documents = self.detector.detect(image)
        if documents:
            for doc_crop in documents:
                result = self.fastmrz_reader.read_mrz(doc_crop)
                if result:
                    return result

        # Fallback: let FastMRZ handle the full image (it has its own segmentation)
        return self.fastmrz_reader.read_from_path(image_path)

    def _process_paddleocr(self, image_path: str, max_dim: int | None = None) -> MRZResult | None:
        """Process using PaddleOCR engine."""
        image = load_image(image_path, max_dim=max_dim)
        documents = self.detector.detect(image)
        sources = list(documents) if documents else []
        sources.append(image)  # full-image fallback always tried after crops
        return self._best_from_sources(sources)

    def _best_from_sources(self, sources) -> MRZResult | None:
        """Parse all OCR candidates across sources; prefer the first valid one.

        Sources are images in priority order (e.g. YOLO crops first, full image
        last). Each is fed through the multi-variant OCR. A source's candidates
        are tried in score order — if any parses valid, return immediately.
        Otherwise we fall back to the highest-scoring non-valid result so we
        still have something to report char-accuracy on.
        """
        best: MRZResult | None = None
        best_score = float("-inf")
        for src in sources:
            for lines, score in self.mrz_reader.read_mrz_candidates(src):
                result = parse_mrz(lines)
                if result is None:
                    continue
                if result.valid:
                    return result
                if score > best_score:
                    best = result
                    best_score = score
        return best

    def process_batch(self, image_paths: list[str]) -> list[MRZResult | None]:
        """Process multiple images."""
        return [self.process(p) for p in image_paths]
