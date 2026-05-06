import os
import re
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

from src.utils import crop_bottom_fraction

# MRZ formats: TD1 (3x30), TD2 (2x36), TD3 (2x44)
MRZ_FORMATS = {
    (2, 44): "TD3",
    (3, 30): "TD1",
    (2, 36): "TD2",
}

# Characters that PaddleOCR commonly misreads in MRZ context
_CHAR_CLEANUP = str.maketrans({
    "«": "<",
    "×": "X",
    "Х": "X",  # Cyrillic Ha
    "О": "O",  # Cyrillic O
    "С": "C",  # Cyrillic Es
    "К": "K",  # Cyrillic Ka
    "А": "A",  # Cyrillic A
    "В": "B",  # Cyrillic Ve
    "Р": "P",  # Cyrillic Er
    "Т": "T",  # Cyrillic Te
    "Н": "H",  # Cyrillic En
    "М": "M",  # Cyrillic Em
    "Е": "E",  # Cyrillic Ye
    "(": "<",
    ")": "<",
    "{": "<",
    "}": "<",
    "[": "<",
    "]": "<",
    "|": "<",
    "!": "1",
    "?": "<",
    ".": "<",
    ",": "<",
    ";": "<",
    ":": "<",
    "'": "<",
    '"': "<",
    "人": "<",
    "梦": "<",
})

MRZ_CHAR_PATTERN = re.compile(r"[A-Z0-9<]")
MRZ_LINE_PATTERN = re.compile(r"^[A-Z0-9<]{20,50}$")


def _clean_ocr_text(text: str) -> str:
    """Normalize OCR output to MRZ character set."""
    text = text.upper().replace(" ", "").translate(_CHAR_CLEANUP)
    # Drop any remaining non-MRZ characters
    return "".join(MRZ_CHAR_PATTERN.findall(text))


class MRZReader:
    def __init__(
        self,
        det_model_dir: str = "models/det",
        rec_model_dir: str = "models/rec",
    ):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        ocr_kwargs: dict = dict(
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
            text_detection_model_name="PP-OCRv4_mobile_det",
            text_recognition_model_name="PP-OCRv4_mobile_rec",
        )

        det_path = Path(det_model_dir)
        rec_path = Path(rec_model_dir)

        if (det_path / "inference.yml").exists():
            ocr_kwargs["text_detection_model_dir"] = det_model_dir
        if (rec_path / "inference.yml").exists():
            ocr_kwargs["text_recognition_model_dir"] = rec_model_dir

        self.ocr = PaddleOCR(**ocr_kwargs)

    def _preprocess_mrz(self, image: np.ndarray, strong: bool = False) -> np.ndarray:
        """Enhance MRZ region for better OCR: grayscale, CLAHE, optional threshold."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clip = 4.0 if strong else 2.0
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        if strong:
            enhanced = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 25, 10,
            )
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _score_mrz(self, lines: list[str], conf: float = 0.0) -> float:
        """Rough quality score: MRZ char density + filler ratio + OCR confidence.

        Higher is better. Used to rank OCR variants before validation. The
        final tie-break on python-mrz validity lives in the pipeline.
        """
        if not lines:
            return 0.0
        text = "".join(lines)
        if not text:
            return 0.0
        mrz_chars = sum(1 for c in text if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
        fillers = text.count("<")
        return mrz_chars / len(text) + 0.1 * (fillers / len(text)) + 0.2 * conf

    def read_mrz_candidates(
        self, document_crop: np.ndarray
    ) -> list[tuple[list[str], float]]:
        """Produce all OCR candidates (lines, score) sorted best-first.

        Emits up to three variants. Score combines MRZ-char density, filler
        ratio, and PaddleOCR's line-level confidence. The caller is expected
        to try each in order and let python-mrz validity break ties.
        """
        bottom = crop_bottom_fraction(document_crop, fraction=0.45)

        raw: list[tuple[list[str], float]] = []

        r1 = self._try_ocr(self._preprocess_mrz(bottom))
        if r1:
            raw.append(r1)

        r2 = self._try_ocr(self._preprocess_mrz(bottom, strong=True))
        if r2:
            raw.append(r2)

        if len(bottom.shape) == 3:
            gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        else:
            gray = bottom
        r3 = self._try_ocr(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if r3:
            raw.append(r3)

        scored = [(lines, self._score_mrz(lines, conf)) for lines, conf in raw]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def read_mrz(self, document_crop: np.ndarray) -> list[str] | None:
        """Back-compat wrapper: returns the top-scoring candidate."""
        cands = self.read_mrz_candidates(document_crop)
        return cands[0][0] if cands else None

    def _try_ocr(self, image: np.ndarray) -> tuple[list[str], float] | None:
        """Run OCR and return (mrz_lines, avg_line_confidence) or None."""
        results = self.ocr.predict(image)
        if not results:
            return None

        result = results[0]
        texts = result.get("rec_texts", [])
        polys = result.get("dt_polys", [])
        scores = result.get("rec_scores", []) or [1.0] * len(texts)

        if not texts:
            return None

        # Extract text fragments with their Y positions, X positions, and score
        fragments = []
        for text, poly, score in zip(texts, polys, scores):
            y_pos = float(np.mean(poly[:, 1]))
            x_pos = float(np.min(poly[:, 0]))
            cleaned = _clean_ocr_text(text)
            if len(cleaned) >= 3:
                fragments.append((y_pos, x_pos, cleaned, float(score)))

        if not fragments:
            return None

        merged = self._merge_by_y(fragments, image.shape[0])

        # Filter to MRZ-like lines
        mrz_candidates = [
            (y, text, conf) for y, text, conf in merged
            if MRZ_LINE_PATTERN.match(text)
        ]

        if not mrz_candidates:
            return None

        mrz_candidates.sort(key=lambda x: x[0])
        lines = [text for _, text, _ in mrz_candidates]
        confs = [conf for _, _, conf in mrz_candidates]
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        assembled = self._assemble_mrz(lines)
        if assembled is None:
            return None
        return assembled, avg_conf

    def _merge_by_y(
        self, fragments: list[tuple[float, float, str, float]], img_height: float
    ) -> list[tuple[float, str, float]]:
        """Merge text fragments that sit on the same horizontal line.

        Fragments are (y, x, text, conf). Returns (avg_y, combined_text, min_conf)
        since a line's weakness is dominated by its weakest recognized span.
        """
        if not fragments:
            return []

        y_threshold = img_height * 0.05
        fragments.sort(key=lambda x: (x[0], x[1]))

        lines: list[list[tuple[float, float, str, float]]] = []
        current_line: list[tuple[float, float, str, float]] = [fragments[0]]

        for frag in fragments[1:]:
            if abs(frag[0] - current_line[0][0]) <= y_threshold:
                current_line.append(frag)
            else:
                lines.append(current_line)
                current_line = [frag]
        lines.append(current_line)

        merged = []
        for line_frags in lines:
            line_frags.sort(key=lambda x: x[1])
            avg_y = sum(f[0] for f in line_frags) / len(line_frags)
            combined = "".join(f[2] for f in line_frags)
            min_conf = min(f[3] for f in line_frags)
            merged.append((avg_y, combined, min_conf))

        return merged

    def _assemble_mrz(self, lines: list[str]) -> list[str] | None:
        """Try to match lines to a known MRZ format."""
        # Try exact match first
        for (num_lines, line_len), fmt in MRZ_FORMATS.items():
            matching = [l for l in lines if len(l) == line_len]
            if len(matching) == num_lines:
                return matching

        # Try padding/trimming lines to known lengths (tight tolerance)
        for (num_lines, line_len), fmt in MRZ_FORMATS.items():
            adjusted = []
            for line in lines:
                if abs(len(line) - line_len) <= 6:
                    if len(line) < line_len:
                        adjusted.append(line + "<" * (line_len - len(line)))
                    else:
                        adjusted.append(line[:line_len])
            if len(adjusted) >= num_lines:
                # Pick the num_lines longest (closest to target length)
                adjusted.sort(key=len, reverse=True)
                return adjusted[:num_lines]

        return None
