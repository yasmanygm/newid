"""FastMRZ-based MRZ reader (uses Tesseract with MRZ-trained model)."""

from pathlib import Path

import numpy as np
from fastmrz import FastMRZ

from src.mrz_parser import MRZResult

DEFAULT_TESSDATA = str(Path(__file__).resolve().parent.parent / "models" / "tessdata")


class FastMRZReader:
    def __init__(self, tessdata_path: str = DEFAULT_TESSDATA):
        self.fast_mrz = FastMRZ(tessdata_path=tessdata_path)

    def read_mrz(self, document_crop: np.ndarray) -> MRZResult | None:
        """Read MRZ from a document crop using FastMRZ."""
        return self._process(document_crop, input_type="numpy")

    def read_from_path(self, image_path: str) -> MRZResult | None:
        """Read MRZ directly from an image path (FastMRZ handles everything)."""
        return self._process(image_path, input_type="imagepath")

    def _process(self, input_data, input_type: str) -> MRZResult | None:
        # First try full parsing
        try:
            result = self.fast_mrz.get_details(input_data, input_type=input_type)
            if result and result.get("status") == "SUCCESS":
                return self._result_from_parsed(result)
        except (ValueError, KeyError, IndexError, TypeError):
            pass

        # Fallback: get raw MRZ text and parse with our own parser
        try:
            raw_mrz = self.fast_mrz.get_details(
                input_data, input_type=input_type, ignore_parse=True
            )
            if raw_mrz and isinstance(raw_mrz, str):
                mrz_lines = [l for l in raw_mrz.strip().split("\n") if l.strip()]
                if mrz_lines:
                    from src.mrz_parser import parse_mrz
                    return parse_mrz(mrz_lines)
        except (ValueError, KeyError, IndexError, TypeError):
            pass

        return None

    def _result_from_parsed(self, result: dict) -> MRZResult | None:
        raw_mrz = result.get("mrz", "")
        mrz_lines = [l for l in raw_mrz.split("\n") if l.strip()]
        if not mrz_lines:
            return None

        doc_type = _detect_doc_type(mrz_lines)
        if not doc_type:
            return None

        fields = {}
        for key in ["surname", "given_name", "document_code", "issuer_code",
                     "nationality", "date_of_birth", "date_of_expiry",
                     "gender", "document_number", "optional_data_1",
                     "optional_data_2"]:
            value = result.get(key)
            if value:
                fields[key] = str(value).strip()

        return MRZResult(
            raw_mrz=mrz_lines,
            corrected_mrz=mrz_lines,
            valid=result.get("status") == "SUCCESS",
            document_type=doc_type,
            fields=fields,
        )


def _detect_doc_type(mrz_lines: list[str]) -> str | None:
    n = len(mrz_lines)
    length = len(mrz_lines[0]) if mrz_lines else 0
    if n == 2 and length == 44:
        return "TD3"
    if n == 3 and length == 30:
        return "TD1"
    if n == 2 and length == 36:
        return "TD2"
    return None
