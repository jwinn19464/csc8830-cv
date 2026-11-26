"""Module 6 Part A - dense optical flow on a 30 s clip, visualised, with per-frame inferences.

For each frame pair (Farneback flow):
  * HSV visualisation: hue = direction, brightness = speed; and sparse arrows
  * camera motion: a robust affine fit to the whole flow field (pan = translation,
    zoom = scale change, roll = rotation)
  * moving objects: pixels whose flow disagrees with the camera model
  * focus of expansion: least-squares point where all flow lines meet (forward motion)

    python -m module6.flow --video walk.mp4 --seconds 30 --out flow.mp4
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def farneback(g0, g1):
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 4, 21, 3, 5, 1.1, 0)


def flow_hsv(flow, max_mag=None):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = np.uint8(ang * 90 / np.pi)
    hsv[..., 1] = 255
    m = max_mag or (np.percentile(mag, 99) + 1e-6)
    hsv[..., 2] = np.uint8(np.clip(255 * mag / m, 0, 255))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_arrows(img, flow, step=24, scale=3.0):
    out = img.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if dx * dx + dy * dy > 0.25:
                cv2.arrowedLine(out, (x, y), (int(x + scale * dx), int(y + scale * dy)), (0, 255, 255), 1, tipLength=0.3)
    return out


def camera_motion(flow, step=8):
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    p0 = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    p1 = p0 + flow[ys.ravel(), xs.ravel()]
    A, inl = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if A is None:
        return None
    s = float(np.hypot(A[0, 0], A[1, 0]))
    return {"tx": float(A[0, 2]), "ty": float(A[1, 2]), "scale": s,
            "rot_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "A": A,
            "inlier_frac": float(inl.mean()) if inl is not None else 0.0}


def moving_mask(flow, A, thr=1.5):
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    px = A[0, 0] * xs + A[0, 1] * ys + A[0, 2] - xs
    py = A[1, 0] * xs + A[1, 1] * ys + A[1, 2] - ys
    res = np.hypot(flow[..., 0] - px, flow[..., 1] - py)
    m = np.uint8(res > thr) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def focus_of_expansion(flow, step=8, min_mag=0.5):
    """Point minimising the distance to every flow line: each vector gives n.(p - x) = 0."""
    h, w = flow.shape[:2]
    ys, xs = np.mgrid[step // 2:h:step, step // 2:w:step]
    f = flow[ys, xs].reshape(-1, 2)
    p = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    mag = np.linalg.norm(f, axis=1)
    keep = mag > min_mag
    if keep.sum() < 20:
        return None
    n = np.stack([-f[keep, 1], f[keep, 0]], 1) / mag[keep, None]
    A, b = n, (n * p[keep]).sum(1)
    foe, *_ = np.linalg.lstsq(A, b, rcond=None)
    # expansion check: flow should point away from the FOE
    away = ((p[keep] - foe) * f[keep]).sum(1)
    return foe, float((away > 0).mean())


def describe(cm, moving_frac, foe):
    words = []
    if cm:
        if abs(cm["tx"]) > 1 or abs(cm["ty"]) > 1:
            dx = "right" if cm["tx"] < 0 else "left"
            dy = "down" if cm["ty"] < 0 else "up"
            words.append(f"camera pans {dx if abs(cm['tx']) >= abs(cm['ty']) else dy}")
        if abs(cm["scale"] - 1) > 0.004:
            words.append("moving forward / zooming in" if cm["scale"] > 1 else "moving back / zooming out")
        if abs(cm["rot_deg"]) > 0.3:
            words.append("camera rolls")
    if moving_frac > 0.02:
        words.append(f"independent motion in {100 * moving_frac:.0f}% of the frame")
    if foe is not None and foe[1] > 0.75:
        words.append("flow expands from a single point (forward motion)")
    return "; ".join(words) or "almost static"


def analyse(frames, fps, every=1):
    rows, vis = [], []
    g_prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, len(frames)):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        if i % every:
            g_prev = g
            continue
        fl = farneback(g_prev, g)
        mag = np.linalg.norm(fl, axis=2)
        cm = camera_motion(fl)
        mm = moving_mask(fl, cm["A"]) if cm else np.zeros(g.shape, np.uint8)
        foe = focus_of_expansion(fl)
        mf = float((mm > 0).mean())
        rows.append({"frame": i, "t (s)": round(i / fps, 2), "mean speed (px/frame)": float(mag.mean()),
                     "max speed": float(np.percentile(mag, 99)),
                     "pan x": cm["tx"] if cm else 0, "pan y": cm["ty"] if cm else 0,
                     "zoom": cm["scale"] if cm else 1, "roll (deg)": cm["rot_deg"] if cm else 0,
                     "moving fraction": mf,
                     "FOE x": float(foe[0][0]) if foe else np.nan, "FOE y": float(foe[0][1]) if foe else np.nan,
                     "inference": describe(cm, mf, foe)})
        hsv = flow_hsv(fl)
        arrows = flow_arrows(frames[i], fl)
        if foe and 0 <= foe[0][0] < g.shape[1] and 0 <= foe[0][1] < g.shape[0] and foe[1] > 0.75:
            cv2.drawMarker(arrows, tuple(int(v) for v in foe[0]), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        vis.append(np.hstack([arrows, hsv]))
        g_prev = g
    return rows, vis


def main():
    from cvlib.io import read_video, write_video
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--out", default="flow.mp4")
    a = ap.parse_args()
    frames, fps = read_video(a.video, a.seconds)
    rows, vis = analyse(frames, fps)
    open(a.out, "wb").write(write_video(vis, fps))
    pd.DataFrame(rows).to_csv(a.out.replace(".mp4", ".csv"), index=False)
    print(f"{len(rows)} frame pairs -> {a.out}")


if __name__ == "__main__":
    main()
