import cv2
import numpy as np


def load_image(path: str, max_dim: int | None = None) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if max_dim is not None and max_dim > 0:
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / longest
            image = cv2.resize(
                image,
                (int(round(w * scale)), int(round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
    return image


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = pts.reshape(4, 2).astype(np.float32)
    # Sort by sum (x+y): smallest = TL, largest = BR
    s = pts.sum(axis=1)
    # Sort by diff (y-x): smallest = TR, largest = BL
    d = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left
    return ordered


def perspective_crop(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Deskew a quadrilateral region from the image into a flat rectangle."""
    ordered = order_points(quad)
    tl, tr, br, bl = ordered

    # Compute destination width and height from edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    if width <= 0 or height <= 0:
        return image

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def crop_bottom_fraction(image: np.ndarray, fraction: float = 0.4) -> np.ndarray:
    """Return the bottom fraction of the image."""
    h = image.shape[0]
    start = int(h * (1 - fraction))
    return image[start:]
