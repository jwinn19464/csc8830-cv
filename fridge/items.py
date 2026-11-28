"""Fridge Oracle step 2 - find items and count each one once (Modules 4 + 6).

Two classical finders:
* Shelf mode (plain background): colour novelty against the background + edges, watershed split.
* Clutter mode (piles, bags, dim light): a "printed packaging" score. Labels are colourful, have
  many colours close together and lots of fine edges and text; bags, walls and shadows don't.
  The highest-scoring spots seed a marker watershed that grows each seed to the item's edges,
  and boxes that mostly overlap are merged.

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


def item_mask(img_bgr, k=4, novelty=3.0):
    """Pixels that don't look like the background. The background colours are learned from a band
    around the frame edge (k-means in Lab); clusters covering under 12% of that band are ignored,
    since they are usually items that touch the edge."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, int(0.05 * min(h, w)))
    band = np.zeros((h, w), bool)
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True
    samples = lab[band]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, lbl, _ = cv2.kmeans(samples, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    flat = lab.reshape(-1, 3)
    d = np.full(len(flat), np.inf, np.float32)
    for i in range(k):
        s_ = samples[lbl.ravel() == i]
        if len(s_) < 0.12 * len(samples):
            continue
        sd = s_.std(0) + 4
        d = np.minimum(d, np.sqrt((((flat - s_.mean(0)) / sd) ** 2).sum(1)))
    col = (d.reshape(h, w) > novelty).astype(np.uint8) * 255
    # strong edges help close outlines, but only near colour-novel pixels (busy backgrounds have edges too)
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    e = norm8(gradient_fft(g, 1.5))
    _, edges = cv2.threshold(e, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    near = cv2.dilate(col, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = cv2.bitwise_or(col, cv2.bitwise_and(edges, near))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = fill_holes(m)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))


def single_item(img_bgr, margin=0.06):
    """Close-up of one item: GrabCut (graph cut with colour mixture models, no learning) started from a
    rectangle just inside the frame. Returns the item's box."""
    h, w = img_bgr.shape[:2]
    s = min(1.0, 500 / max(h, w))
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    sh, sw = small.shape[:2]
    rect = (int(margin * sw), int(margin * sh), int((1 - 2 * margin) * sw), int((1 - 2 * margin) * sh))
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
        fg = np.uint8((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
        if n > 1:
            i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh = st[i, :4]
            if bw * bh > 0.05 * sw * sh:
                return (int(x / s), int(y / s), int(bw / s), int(bh / s))
    except cv2.error:
        pass
    return (int(margin * w), int(margin * h), int((1 - 2 * margin) * w), int((1 - 2 * margin) * h))


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


def packaging_score(img_bgr):
    """0..1 map of how much each neighbourhood looks like printed packaging."""
    h, w = img_bgr.shape[:2]
    k = max(15, int(0.06 * min(h, w))) | 1
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    edges = cv2.blur((gm > np.percentile(gm, 75)).astype(np.float32), (k, k))
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    var = sum(cv2.blur(lab[:, :, c] ** 2, (k, k)) - cv2.blur(lab[:, :, c], (k, k)) ** 2 for c in (1, 2))
    variety = np.sqrt(np.maximum(var, 0))
    variety = np.clip(variety / (np.percentile(variety, 99) + 1e-6), 0, 1)
    colour = cv2.blur(hsv[:, :, 1] * hsv[:, :, 2], (k, k))
    sc = 0.4 * edges / (edges.max() + 1e-6) + 0.3 * variety + 0.3 * colour / (colour.max() + 1e-6)
    return sc / (sc.max() + 1e-6)


def merge_boxes(boxes, frac=0.35):
    """Merge boxes whose overlap covers at least `frac` of the smaller one (one item split in two)."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if iw > 0 and ih > 0 and iw * ih >= frac * min(a[2] * a[3], b[2] * b[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(int(v) for v in b) for b in boxes]


def detect_cluttered(img_bgr, seed_pct=92):
    h, w = img_bgr.shape[:2]
    s8 = np.uint8(255 * packaging_score(img_bgr))
    lo, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    core = np.uint8(s8 >= np.percentile(s8, seed_pct)) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    markers = np.zeros((h, w), np.int32)
    markers[s8 < 0.8 * lo] = 1
    k = 2
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.002 * h * w:
            markers[lab == i] = k
            k += 1
    cv2.watershed(cv2.GaussianBlur(img_bgr, (5, 5), 0), markers)
    boxes = []
    for i in range(2, k):
        m = np.uint8(markers == i)
        a = int(m.sum())
        if 0.01 * h * w <= a <= 0.6 * h * w:
            boxes.append(cv2.boundingRect(m))
    return merge_boxes(boxes), s8


def detect_items(img_bgr, min_area_frac=0.004, max_area_frac=0.92, one_item=False, mode="auto"):
    """Item boxes in one image.
    mode: "shelf" (plain background), "clutter" (piles, bags, dim light) or "auto" (shelf first;
    clutter when the shelf finder sees almost everything as an item or finds nothing).
    Close-ups (one_item=True) use GrabCut on the whole photo."""
    h, w = img_bgr.shape[:2]
    if one_item:
        return [single_item(img_bgr)], None
    boxes, mask = [], None
    if mode in ("auto", "shelf"):
        mask = item_mask(img_bgr)
        if (mask > 0).mean() < (0.4 if mode == "auto" else 0.85):
            markers = split_items(mask, img_bgr)
            for lab in range(2, markers.max() + 1):
                m = np.uint8(markers == lab)
                a = int(m.sum())
                if not (min_area_frac * h * w <= a <= max_area_frac * h * w):
                    continue
                x, y, bw, bh = cv2.boundingRect(m)
                if bw < 12 or bh < 12 or a / (bw * bh) < 0.35:
                    continue
                boxes.append((x, y, bw, bh))
    if not boxes and mode in ("auto", "clutter"):
        boxes, mask = detect_cluttered(img_bgr)
    if not boxes:
        boxes = [single_item(img_bgr)]
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
