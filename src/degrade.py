"""
degrade.py - STEP 2: domain-randomised degradation ("make it look like a phone photo").

Takes a clean rendered statement page and applies random, realistic damage:
paper placed on a desk background, rotation + camera perspective, blur, uneven
lighting/shadow, a bank stamp, sensor noise and JPEG compression.

Only the PIXELS change - the ground truth stays exactly the same - so every damaged
image still has a perfect answer key. This is data augmentation / domain
randomisation: the model sees so many variations during training that a real phone
photo looks like "just one more variation".

Usage:
    python src/degrade.py page.png out.jpg --level medium --seed 3
"""
import argparse
import random

import cv2
import numpy as np

LEVELS = {  # severity presets used for the accuracy-vs-damage curve
    "clean":  dict(rot=0.0, persp=0.00, blur=0, noise=0,  jpeg=95, shadow=0.00, stamp=0.0, desk=0.00),
    "light":  dict(rot=1.0, persp=0.01, blur=0, noise=4,  jpeg=85, shadow=0.10, stamp=0.2, desk=0.03),
    "medium": dict(rot=2.5, persp=0.02, blur=1, noise=8,  jpeg=65, shadow=0.25, stamp=0.5, desk=0.05),
    "heavy":  dict(rot=4.0, persp=0.035, blur=2, noise=13, jpeg=40, shadow=0.40, stamp=0.8, desk=0.07),
}
DESK_COLORS = [(92, 120, 160), (60, 70, 85), (150, 160, 170), (35, 45, 60), (120, 150, 185)]  # BGR


def place_on_desk(img, frac, rng):
    """Add a desk-coloured border around the page (what a phone photo looks like)."""
    if frac <= 0:
        return img
    h, w = img.shape[:2]
    p = int(frac * max(h, w))
    color = rng.choice(DESK_COLORS)
    out = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=color)
    noise = np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, out.shape)
    mask = np.ones(out.shape[:2], bool)
    mask[p:p + h, p:p + w] = False
    out = out.astype(float)
    out[mask] += noise[mask]
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate_and_perspective(img, max_deg, persp, rng, border):
    h, w = img.shape[:2]
    angle = rng.uniform(-max_deg, max_deg)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=border)
    if persp > 0:
        d = persp * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[rng.uniform(0, d), rng.uniform(0, d)], [w - rng.uniform(0, d), rng.uniform(0, d)],
                          [w - rng.uniform(0, d), h - rng.uniform(0, d)], [rng.uniform(0, d), h - rng.uniform(0, d)]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=border)
    return img, angle


def shadow(img, strength, rng):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    cx, cy = rng.random(), rng.random()
    mask = 1 - strength * np.clip(1 - np.sqrt((x - cx) ** 2 + (y - cy) ** 2), 0, 1)
    return (img * mask[..., None]).astype(np.uint8)


def stamp(img, rng):
    h, w = img.shape[:2]
    over = img.copy()
    c = (int(rng.uniform(.25, .8) * w), int(rng.uniform(.35, .85) * h))
    r = int(.07 * w)
    col = rng.choice([(170, 60, 60), (60, 60, 170), (90, 40, 140)])
    cv2.circle(over, c, r, col, max(2, w // 400))
    cv2.circle(over, c, int(r * .8), col, max(1, w // 600))
    cv2.putText(over, rng.choice(["VERIFIED", "RECEIVED", "BRANCH"]), (c[0] - int(.05 * w), c[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX, w / 1500, col, max(1, w // 500))
    return cv2.addWeighted(over, 0.55, img, 0.45, 0)


def degrade(img: np.ndarray, level: str = "medium", seed: int = 0) -> tuple[np.ndarray, dict]:
    """Apply one random degradation. Returns (image, info) where info records what was done."""
    p, rng = LEVELS[level], random.Random(seed)
    info = {"level": level, "seed": seed}
    if level == "clean":
        return img.copy(), info | {"angle": 0.0}
    if rng.random() < p["stamp"]:
        img = stamp(img, rng)
    img = place_on_desk(img, p["desk"], rng)
    border = tuple(int(v) for v in img[0, 0]) if p["desk"] > 0 else (235, 235, 230)
    img, angle = rotate_and_perspective(img, p["rot"], p["persp"], rng, border)
    if p["blur"]:
        k = 2 * p["blur"] + 1
        img = cv2.GaussianBlur(img, (k, k), 0)
    img = shadow(img, p["shadow"], rng)
    noise = np.random.default_rng(seed).normal(0, p["noise"], img.shape)
    img = np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR), info | {"angle": round(angle, 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--level", default="medium", choices=list(LEVELS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out, info = degrade(cv2.imread(a.src), a.level, a.seed)
    cv2.imwrite(a.dst, out)
    print(info)
