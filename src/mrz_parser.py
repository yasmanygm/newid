from dataclasses import dataclass, field
from itertools import product

from mrz.base.countries import english as _COUNTRY_NAMES
from mrz.checker.td1 import TD1CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td3 import TD3CodeChecker

# ISO 3166 alpha-3 codes accepted by MRZ readers, plus ICAO special codes
# (stateless persons, UN agencies, etc.). Built from python-mrz's own table.
_COUNTRY_CODES = set(_COUNTRY_NAMES.values())
_COUNTRY_CODES.update({
    "UNO", "UNA", "UNK",      # UN agencies / unknown
    "XXA", "XXB", "XXC", "XXD",  # stateless / refugee
    "XBA", "XIM", "XCC", "XCE", "XCO", "XEC", "XPO", "XES", "XMP", "XOM",
    "GBD", "GBN", "GBO", "GBP", "GBS",  # British sub-nationality codes
    "D",  # legacy Germany (padded as D<<)
})


@dataclass
class MRZResult:
    raw_mrz: list[str]
    corrected_mrz: list[str]
    valid: bool
    document_type: str
    fields: dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [f"Document Type: {self.document_type}", f"Valid: {self.valid}"]
        for key, value in self.fields.items():
            lines.append(f"  {key}: {value}")
        lines.append("MRZ:")
        for line in self.corrected_mrz:
            lines.append(f"  {line}")
        return "\n".join(lines)


# ICAO 9303 field position maps: 'A' = alpha/filler, 'N' = numeric, 'X' = either
TD3_MAP = [
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "XXXXXXXXXNAAANNNNNNNANNNNNNNXXXXXXXXXXXXXXNN",
]
TD1_MAP = [
    "AAAAAXXXXXXXXXNXXXXXXXXXXXXXXX",
    "NNNNNNNANNNNNNNAAAXXXXXXXXXXXN",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
]
TD2_MAP = [
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "XXXXXXXXXNAAANNNNNNNANNNNNNNXXXXXXXN",
]

ALPHA_TO_DIGIT = {"O": "0", "I": "1", "B": "8", "S": "5", "G": "6", "Z": "2",
                   "D": "0", "Q": "0", "T": "7"}
DIGIT_TO_ALPHA = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "2": "Z", "7": "T"}

# Bidirectional confusions for checksum-guided search (tried in rank order)
CONFUSIONS = {
    "0": ["O", "Q", "D"],
    "O": ["0", "Q", "D"],
    "1": ["I", "L", "T"],
    "I": ["1", "L", "T"],
    "L": ["1", "I"],
    "5": ["S"],
    "S": ["5"],
    "8": ["B"],
    "B": ["8"],
    "2": ["Z"],
    "Z": ["2"],
    "6": ["G"],
    "G": ["6"],
    "7": ["T"],
    "T": ["7"],
    "0O": ["Q", "D"],
}

CHECKERS = {"TD1": TD1CodeChecker, "TD2": TD2CodeChecker, "TD3": TD3CodeChecker}
FIELD_MAPS = {"TD1": TD1_MAP, "TD2": TD2_MAP, "TD3": TD3_MAP}

# (line_idx, slice_start, slice_end) for issuing-country + nationality fields.
COUNTRY_POSITIONS = {
    "TD3": [(0, 2, 5), (1, 10, 13)],
    "TD2": [(0, 2, 5), (1, 10, 13)],
    "TD1": [(0, 2, 5), (1, 15, 18)],
}

# ICAO 9303 check-digit weights: positions cycle through (7,3,1)
_CD_WEIGHTS = (7, 3, 1)


def _icao_char_value(ch: str) -> int:
    if ch.isdigit():
        return int(ch)
    if "A" <= ch <= "Z":
        return ord(ch) - ord("A") + 10
    return 0  # '<' and anything else


def _icao_check_digit(s: str) -> int:
    total = 0
    for i, ch in enumerate(s):
        total += _icao_char_value(ch) * _CD_WEIGHTS[i % 3]
    return total % 10


# Check-digit positions by document type: (line_idx, check_pos, field_start, field_end_exclusive)
# Both indices are 0-based. field slice is [field_start:field_end].
CD_SPECS = {
    "TD3": [
        (1, 9,  0,  9),    # document number check
        (1, 19, 13, 19),   # birth date check
        (1, 27, 21, 27),   # expiry date check
        (1, 42, 28, 42),   # optional data check
        (1, 43, None, None),  # overall check (composite, handled separately)
    ],
    "TD2": [
        (1, 9,  0,  9),
        (1, 19, 13, 19),
        (1, 27, 21, 27),
        (1, 35, None, None),  # overall
    ],
    "TD1": [
        (0, 14, 5,  14),   # document number
        (1, 6,  0,  6),    # birth date
        (1, 14, 8,  14),   # expiry date
        (1, 29, None, None),  # overall
    ],
}


def _detect_type(mrz_lines: list[str]) -> str | None:
    n = len(mrz_lines)
    length = len(mrz_lines[0]) if mrz_lines else 0
    if n == 2 and length == 44:
        return "TD3"
    if n == 3 and length == 30:
        return "TD1"
    if n == 2 and length == 36:
        return "TD2"
    return None


def _collapse_trailing_fillers(line: str, min_run: int = 2) -> str:
    """After the last solid alpha block, collapse hallucinated letters to '<'.

    Finds the position of the last run of `min_run` consecutive `<` and, from that
    point to the right, rewrites any letter that is surrounded by `<` into `<`.
    This fixes the common OCR pattern where trailing filler runs get garbled as
    `<<<<S<K<<<<K<<CC<` instead of `<<<<<<<<<<<<<<<<<<`.
    """
    if "<" not in line:
        return line
    # Find last occurrence of min_run consecutive '<'
    pattern = "<" * min_run
    anchor = line.rfind(pattern)
    if anchor < 0:
        return line
    # Sweep rightwards from the anchor: any non-'<' char preceded or followed
    # by '<' in its immediate neighborhood becomes '<'.
    chars = list(line)
    for i in range(anchor + min_run, len(chars)):
        ch = chars[i]
        if ch == "<":
            continue
        left = chars[i - 1] if i > 0 else ""
        right = chars[i + 1] if i + 1 < len(chars) else ""
        # Isolated letter in filler region → filler
        if left == "<" and (right == "<" or right == ""):
            chars[i] = "<"
        elif left == "<" and right != "<" and i + 2 < len(chars) and chars[i + 2] == "<":
            # Two-letter island: also filler ("...xxYZ<..." → "...xx<<<...")
            chars[i] = "<"
            chars[i + 1] = "<"
    return "".join(chars)


def _collapse_filler_islands(line: str) -> str:
    """Replace 1-2 letter islands inside filler runs with '<'.

    More aggressive than trailing-only: scans the whole line and converts any
    letter/digit that is enclosed by '<' on both sides within a 3-char window.
    """
    chars = list(line)
    n = len(chars)
    # Single-char islands: <X< → <<<
    for i in range(1, n - 1):
        if chars[i] != "<" and chars[i - 1] == "<" and chars[i + 1] == "<":
            chars[i] = "<"
    # Two-char islands: <XY< → <<<<
    for i in range(1, n - 2):
        if chars[i] != "<" and chars[i + 1] != "<" and chars[i - 1] == "<" and chars[i + 2] == "<":
            chars[i] = "<"
            chars[i + 1] = "<"
    return "".join(chars)


def _closest_country(code: str, max_distance: int = 1) -> str | None:
    """Find the ISO code with the smallest Hamming distance from `code`.

    Returns the match only when unambiguous — i.e. exactly one code sits at
    the minimum distance and that distance is <= max_distance. If multiple
    codes tie at the minimum distance we decline to guess. `<` is treated as
    a wildcard that matches anything for free (padded short codes like D<<).
    """
    if len(code) != 3:
        return None
    best = None
    best_d = max_distance + 1
    tie = False
    for cand in _COUNTRY_CODES:
        if len(cand) != 3:
            # Pad short codes (D → D<<) to compare positionally.
            cand_padded = cand + "<" * (3 - len(cand))
        else:
            cand_padded = cand
        d = sum(
            1 for a, b in zip(code, cand_padded)
            if a != b and a != "<" and b != "<"
        )
        if d < best_d:
            best_d = d
            best = cand
            tie = False
        elif d == best_d:
            tie = True
    if tie or best is None or best_d > max_distance:
        return None
    return best


def _country_correct(mrz_lines: list[str], doc_type: str) -> list[str]:
    """Snap issuing-country and nationality fields to the ISO 3166 whitelist.

    Alpha positions don't have MRZ check digits, so OCR errors like HON→HUN
    or URV→URY survive every other correction pass. Here we force the field
    to a valid code when (and only when) there is a single unambiguous match
    within Hamming distance 1.
    """
    positions = COUNTRY_POSITIONS.get(doc_type)
    if not positions:
        return mrz_lines
    lines = list(mrz_lines)
    for line_idx, start, end in positions:
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        if end > len(line):
            continue
        code = line[start:end]
        if code in _COUNTRY_CODES:
            continue
        # Already valid padded-short form (D<<)? leave it.
        if code[0].isalpha() and all(c == "<" for c in code[1:]):
            if code[0] in _COUNTRY_CODES:
                continue
        match = _closest_country(code, max_distance=1)
        if match:
            padded = match + "<" * (end - start - len(match))
            lines[line_idx] = line[:start] + padded + line[end:]
    return lines


def _positional_correct(mrz_lines: list[str], doc_type: str) -> list[str]:
    field_map = FIELD_MAPS.get(doc_type)
    if not field_map or len(field_map) != len(mrz_lines):
        return mrz_lines
    corrected = []
    for line, positions in zip(mrz_lines, field_map):
        chars = list(line)
        for i, ch in enumerate(chars):
            if i >= len(positions):
                break
            pos_type = positions[i]
            if pos_type == "N" and ch in ALPHA_TO_DIGIT:
                chars[i] = ALPHA_TO_DIGIT[ch]
            elif pos_type == "A" and ch in DIGIT_TO_ALPHA:
                chars[i] = DIGIT_TO_ALPHA[ch]
        corrected.append("".join(chars))
    return corrected


# Backwards-compatible alias
correct_mrz_characters = _positional_correct


def _fix_field_with_check(line: str, field_start: int, field_end: int,
                           check_pos: int, max_flips: int = 2) -> str | None:
    """Try to fix a field so that its check digit matches.

    Brute-force search: flip up to `max_flips` chars in the field using the
    CONFUSIONS map. Returns the first candidate that validates, or None.
    """
    field = line[field_start:field_end]
    expected_check = line[check_pos]
    if expected_check not in "0123456789":
        # Check digit itself might be wrong; try correcting via ALPHA_TO_DIGIT
        if expected_check in ALPHA_TO_DIGIT:
            expected_check = ALPHA_TO_DIGIT[expected_check]
        else:
            return None

    # Pass 0: does it already validate?
    if _icao_check_digit(field) == int(expected_check):
        # Ensure the check digit char itself is set correctly
        if line[check_pos] != expected_check:
            return line[:check_pos] + expected_check + line[check_pos + 1:]
        return line

    # Pass 1+: flip chars in the field
    candidates = []
    for i, ch in enumerate(field):
        opts = CONFUSIONS.get(ch, [])
        if opts:
            candidates.append((i, opts))

    if not candidates:
        return None

    # Try single-char flips first, then multi-char up to max_flips
    for k in range(1, min(max_flips, len(candidates)) + 1):
        from itertools import combinations
        for combo in combinations(range(len(candidates)), k):
            pos_opts = [(candidates[j][0], candidates[j][1]) for j in combo]
            for choice in product(*[o for _, o in pos_opts]):
                new_field = list(field)
                for (pos, _), new_ch in zip(pos_opts, choice):
                    new_field[pos] = new_ch
                candidate = "".join(new_field)
                if _icao_check_digit(candidate) == int(expected_check):
                    # Patch the line
                    return line[:field_start] + candidate + line[field_end:]
    return None


def _checksum_correct(mrz_lines: list[str], doc_type: str) -> list[str]:
    """Iteratively fix each field so its check digit validates."""
    specs = CD_SPECS.get(doc_type)
    if not specs:
        return mrz_lines

    lines = list(mrz_lines)
    for line_idx, check_pos, f_start, f_end in specs:
        if f_start is None:  # skip composite check
            continue
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        if check_pos >= len(line) or f_end > len(line):
            continue
        fixed = _fix_field_with_check(line, f_start, f_end, check_pos)
        if fixed:
            lines[line_idx] = fixed
    return lines


def _extract_fields(checker) -> dict:
    f = checker.fields()
    fields = {}
    for attr in ["surname", "name", "country", "nationality", "birth_date",
                  "expiry_date", "sex", "document_type", "document_number",
                  "optional_data", "optional_data_2"]:
        value = getattr(f, attr, None)
        if value is not None:
            fields[attr] = str(value).strip("<").strip()
    return fields


def _try_validate(mrz_lines: list[str], doc_type: str) -> tuple[bool, dict]:
    checker_cls = CHECKERS[doc_type]
    try:
        checker = checker_cls("\n".join(mrz_lines))
        return bool(checker), _extract_fields(checker)
    except Exception:
        return False, {}


def parse_mrz(mrz_lines: list[str]) -> MRZResult | None:
    """Parse and validate MRZ lines with multi-stage correction."""
    doc_type = _detect_type(mrz_lines)
    if not doc_type:
        return None

    # Stage 1: raw
    ok, fields = _try_validate(mrz_lines, doc_type)
    if ok:
        return MRZResult(mrz_lines, mrz_lines, True, doc_type, fields)

    # Stage 2: positional + country-code whitelist correction
    pos_corrected = _positional_correct(mrz_lines, doc_type)
    pos_corrected = _country_correct(pos_corrected, doc_type)
    ok, fields = _try_validate(pos_corrected, doc_type)
    if ok:
        return MRZResult(mrz_lines, pos_corrected, True, doc_type, fields)

    # Stage 3: collapse hallucinated letters inside trailing filler runs only
    # (full-line island collapse is too aggressive — destroys single-letter names)
    filler_fixed = [_collapse_trailing_fillers(l, min_run=4) for l in pos_corrected]
    ok, fields = _try_validate(filler_fixed, doc_type)
    if ok:
        return MRZResult(mrz_lines, filler_fixed, True, doc_type, fields)

    # Stage 4: positional + filler-fix + per-field checksum correction
    cs_corrected = _checksum_correct(filler_fixed, doc_type)
    ok, fields = _try_validate(cs_corrected, doc_type)
    if ok:
        return MRZResult(mrz_lines, cs_corrected, True, doc_type, fields)

    # Stage 5: checksum on the raw lines (no positional correction)
    cs_only = _checksum_correct(mrz_lines, doc_type)
    ok, fields = _try_validate(cs_only, doc_type)
    if ok:
        return MRZResult(mrz_lines, cs_only, True, doc_type, fields)

    # Nothing valid — return best-effort corrected output for char_acc measurement
    return MRZResult(
        raw_mrz=mrz_lines,
        corrected_mrz=cs_corrected,
        valid=False,
        document_type=doc_type,
        fields=fields,
    )
