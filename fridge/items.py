"""Fridge Oracle step 2 - find items and count each one once (Modules 4 + 6).

Per keyframe (classical, Module 4 ideas):
  colour novelty against the dominant shelf colours (k-means in Lab) OR'd with strong
  Fourier-gradient edges, closed, holes filled, split by distance-transform watershed,
  then filtered by size and shape.
Across keyframes (Module 6): ORB matches + RANSAC homography chain every keyframe to the
first, boxes are mapped into that common frame, and overlapping boxes (IoU) are merged,
so an item seen in three keyframes is one inventory entry.
"""
from __future__ import annotations

import cv2
import numpy as np

from module4.fourier import gradient_fft, norm8
from module4.segment import fill_holes


def item_mask(img_bgr, k=3, novelty=3.0):
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, lbl, cen = cv2.kmeans(lab[::7], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    bg = [i for i in np.argsort(-counts)[:max(1, k - 1)] if counts[i] > 0.25 * len(lbl)] or [int(np.argmax(counts))]
    d = np.full(len(lab), np.inf)
    sub = lab[::7]
    for i in bg:
        s = sub[lbl.ravel() == i]
        sd = s.std(0) + 4
        d = np.minimum(d, np.sqrt((((lab - s.mean(0)) / sd) ** 2).sum(1)))
    col = (d.reshape(h, w) > novelty).astype(np.uint8) * 255
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = norm8(gradient_fft(g, 1.5))
    _, edges = cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.bitwise_or(col, cv2.dilate(edges, np.ones((3, 3), np.uint8)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = fill_holes(m)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def split_items(mask, img_bgr):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dist, 0.45 * dist.max(), 255, 0)
    n, markers = cv2.connectedComponents(np.uint8(peaks))
    markers = markers + 1
    markers[cv2.dilate(mask, np.ones((3, 3), np.uint8)) == 0] = 0
    unknown = (mask > 0) & (peaks == 0)
    markers[unknown] = 0
    markers[mask == 0] = 1
    cv2.watershed(img_bgr, markers)
    return markers  # 1 = background, 2.. = items, -1 = boundaries


def detect_items(img_bgr, min_area_frac=0.004, max_area_frac=0.5):
    h, w = img_bgr.shape[:2]
    mask = item_mask(img_bgr)
    markers = split_items(mask, img_bgr)
    boxes = []
    for lab in range(2, markers.max() + 1):
        m = np.uint8(markers == lab)
        a = int(m.sum())
        if not (min_area_frac * h * w <= a <= max_area_frac * h * w):
            continue
        x, y, bw, bh = cv2.boundingRect(m)
        if bw < 12 or bh < 12 or a / (bw * bh) < 0.35:
            continue
        boxes.append((x, y, bw, bh))
    return boxes, mask


def homography_to_first(frames):
    """H_k mapping keyframe k into keyframe 0 coordinates (identity for k = 0)."""
    orb = cv2.ORB_create(3000)
    kd = [orb.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None) for f in frames]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    Hs = [np.eye(3)]
    for k in range(1, len(frames)):
        (ka, da), (kb, db) = kd[k - 1], kd[k]
        H = None
        if da is not None and db is not None and len(da) > 8 and len(db) > 8:
            good = [p[0] for p in bf.knnMatch(db, da, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) >= 10:
                src = np.float32([kb[m.queryIdx].pt for m in good])
                dst = np.float32([ka[m.trainIdx].pt for m in good])
                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:  # lost track: assume a new, non-overlapping view far to the right
            H = np.array([[1, 0, frames[0].shape[1] * 1.5 * k], [0, 1, 0], [0, 0, 1]], float)
            Hs.append(H)
        else:
            Hs.append(Hs[-1] @ H)
    return Hs


def map_box(box, H):
    x, y, w, h = box
    c = cv2.perspectiveTransform(np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2), H)
    c = c.reshape(-1, 2)
    x0, y0 = c.min(0)
    x1, y1 = c.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def box_iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u else 0.0


def dedupe(frames, boxes_per_frame, iou_thr=0.3):
    """Returns a list of unique items: {'views': [(frame k, box)], 'global_box'}."""
    Hs = homography_to_first(frames)
    items = []
    for k, boxes in enumerate(boxes_per_frame):
        for b in boxes:
            g = map_box(b, Hs[k])
            hit = None
            for it in items:
                if box_iou(g, it["global_box"]) > iou_thr and all(v[0] != k for v in it["views"]):
                    hit = it
                    break
            if hit:
                hit["views"].append((k, b))
            else:
                items.append({"views": [(k, b)], "global_box": g})
    return items


def best_crop(frames, item, pad=6):
    """Largest view of the item, cropped."""
    k, (x, y, w, h) = max(item["views"], key=lambda v: v[1][2] * v[1][3])
    f = frames[k]
    H, W = f.shape[:2]
    return f[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)].copy(), k
