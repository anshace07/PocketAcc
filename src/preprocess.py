"""
preprocess.py - STEP 3: classical image processing before the model reads a page.

    render_pages()   PDF -> page images (Poppler)
    crop_paper()     find the sheet of paper on the desk (thresholding + largest contour)
    deskew()         estimate text-line angle with a Hough transform and rotate it straight
    resize_for_vlm() scale so width*height <= max_pixels, sides multiples of 28 (Qwen2-VL patch grid)
    prepare()        the full pipeline used for both training and inference

Usage:
    python src/preprocess.py photo.jpg out.png
"""
import argparse
import math

import cv2
import numpy as np


def render_pages(pdf_path: str, dpi: int = 150) -> list[np.ndarray]:
    from pdf2image import convert_from_path
    return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in convert_from_path(pdf_path, dpi=dpi)]


def crop_paper(img: np.ndarray, min_area: float = 0.35) -> np.ndarray:
    """Cut the bright paper out of a darker background (phone photo on a desk)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    h, w = gray.shape
    if cv2.contourArea(c) < min_area * h * w:
        return img                                   # no clear paper edge -> leave as is
    x, y, bw, bh = cv2.boundingRect(c)
    return img[y:y + bh, x:x + bw]


def estimate_skew(img: np.ndarray, max_angle: float = 10.0) -> float:
    """Median angle of near-horizontal lines found by the probabilistic Hough transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 1600 / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    # smear characters horizontally so each text line becomes one long blob
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)))
    edges = cv2.Canny(th, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=120, minLineLength=gray.shape[1] // 6, maxLineGap=15)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) <= max_angle:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(255, 255, 255))


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(img)
    return (rotate(img, angle), angle) if abs(angle) > 0.15 else (img, angle)


def resize_for_vlm(img: np.ndarray, max_pixels: int = 1_400_000, patch: int = 28) -> np.ndarray:
    """Qwen2-VL turns every 28x28 pixel block into one visual token, so pixels = GPU memory."""
    h, w = img.shape[:2]
    s = min(1.0, math.sqrt(max_pixels / (h * w)))
    nh, nw = max(patch, int(h * s) // patch * patch), max(patch, int(w * s) // patch * patch)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def prepare(img: np.ndarray, max_pixels: int = 1_400_000) -> np.ndarray:
    img = crop_paper(img)
    img, _ = deskew(img)
    return resize_for_vlm(img, max_pixels)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-pixels", type=int, default=1_400_000)
    a = ap.parse_args()
    out = prepare(cv2.imread(a.src), a.max_pixels)
    cv2.imwrite(a.dst, out)
    print("saved", a.dst, out.shape)
