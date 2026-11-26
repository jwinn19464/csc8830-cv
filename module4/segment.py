"""Module 4 - exact human boundary with classical CV only (no deep learning / ML).

Q1 RGB:     Lab colour -> background model (border band, or outside a box you draw)
            -> per-pixel colour novelty (Mahalanobis-like distance) -> hysteresis threshold
            -> morphology -> largest component -> watershed refinement -> contour
Q2 Thermal: CLAHE -> polarity (people are usually brighter / hotter) -> Otsu -> morphology
            -> largest component -> watershed refinement -> contour

    python -m module4.segment --image person.jpg --mode rgb [--box X Y W H]
    python -m module4.segment --image thermal.png --mode thermal
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold


def largest_component(mask: np.ndarray, seed=None) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    if seed is not None and lab[seed[1], seed[0]] > 0:
        k = lab[seed[1], seed[0]]
    else:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return np.uint8(lab == k) * 255


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = mask.copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m, (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def watershed_refine(img_bgr, mask, band=None):
    """Grow/shrink the coarse mask to the strongest nearby edges with marker watershed."""
    h, w = mask.shape
    band = band or max(5, int(0.015 * max(h, w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    sure_fg = cv2.erode(mask, k)
    sure_bg = cv2.dilate(mask, k) == 0
    markers = np.zeros((h, w), np.int32)
    markers[sure_bg] = 1
    markers[sure_fg > 0] = 2
    img = img_bgr if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    cv2.watershed(cv2.GaussianBlur(img, (3, 3), 0), markers)
    out = np.uint8(markers == 2) * 255
    return fill_holes(cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))


def boundary(mask):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2) if cs else np.zeros((0, 2), int)


def segment_rgb(img_bgr, box=None, lo=2.5, hi=5.0):
    """Return (mask 0/255, steps dict). box = (x, y, w, h) around the person, optional."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float64)
    bgm = np.zeros((h, w), bool)
    if box is not None:
        x, y, bw, bh = [int(v) for v in box]
        bgm[:] = True
        bgm[max(0, y):y + bh, max(0, x):x + bw] = False
    else:
        b = max(4, int(0.06 * min(h, w)))
        bgm[:b], bgm[-b:], bgm[:, :b], bgm[:, -b:] = True, True, True, True
        bgm[int(0.6 * h):, :] &= False  # people usually touch the bottom edge; don't learn them as background
        bgm[-b:, :b] = bgm[-b:, -b:] = True
    samples = lab[bgm]
    # background as a few colour clusters so busy backgrounds still work
    K = 4
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples.astype(np.float32), K, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    d = np.full((h, w), np.inf)
    flat = lab.reshape(-1, 3)
    for k in range(K):
        s = samples[labels.ravel() == k]
        if len(s) < 10:
            continue
        cov = np.cov(s.T) + np.eye(3) * 4.0
        inv = np.linalg.inv(cov)
        diff = flat - s.mean(0)
        dk = np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff)).reshape(h, w)
        d = np.minimum(d, dk)
    score = np.minimum(d, 4 * hi)
    raw = apply_hysteresis_threshold(score, lo, hi)
    if box is not None:
        raw[bgm] = False
    m = np.uint8(raw) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    seed = (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2)) if box is not None else None
    coarse = fill_holes(largest_component(m, seed))
    final = largest_component(watershed_refine(img_bgr, coarse), seed)
    vis = np.uint8(255 * score / score.max())
    return final, {"novelty score": vis, "hysteresis": np.uint8(raw) * 255, "coarse mask": coarse, "final mask": final}


def segment_thermal(img, hot_is_bright=True):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if not hot_is_bright:
        g = 255 - g
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(g, (5, 5), 0))
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    coarse = fill_holes(largest_component(m))
    final = largest_component(watershed_refine(cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR), coarse))
    return final, {"CLAHE": eq, "Otsu": th, "coarse mask": coarse, "final mask": final}


def overlay(img_bgr, mask, color=(0, 255, 0), other=None, other_color=(255, 0, 255)):
    out = img_bgr.copy() if img_bgr.ndim == 3 else cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    tint = out.copy()
    tint[mask > 0] = (0.55 * tint[mask > 0] + 0.45 * np.array(color)).astype(np.uint8)
    t = max(2, out.shape[1] // 300)
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tint, cs, -1, color, t)
    if other is not None:
        cs2, _ = cv2.findContours(other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(tint, cs2, -1, other_color, t)
    return tint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--mode", choices=["rgb", "thermal"], default="rgb")
    ap.add_argument("--box", nargs=4, type=int)
    ap.add_argument("--out", default="mask.png")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    mask, _ = segment_rgb(img, a.box) if a.mode == "rgb" else segment_thermal(img)
    cv2.imwrite(a.out, mask)
    print(f"wrote {a.out}: {int((mask > 0).sum())} person pixels")


if __name__ == "__main__":
    main()
