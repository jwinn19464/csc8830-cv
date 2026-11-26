"""Module 7 - room dimensions from stereo video.

1. Sync the two videos: flick the light or fire a flash at the start; the largest jump in
   mean brightness marks the same instant in both clips.
2. For a few synced frame pairs: rectify, semi-global block matching (SGBM) disparity,
   reproject to 3-D with Q.
3. RANSAC planes: repeatedly fit the dominant plane and remove its points (floor, ceiling,
   walls). Group planes with parallel normals; the gap between two opposite parallel planes
   is a room dimension (length, width, height).
4. Fallback: tap opposite corners of the room in the left frame; they are matched on the
   same row and triangulated (module7.stereo.points_3d).

A phone rig with a 10-30 cm baseline is accurate to a few percent at 3-5 m. Stand in the
middle of the room; blank walls give no disparity, so posters and furniture help.
"""
from __future__ import annotations

import cv2
import numpy as np

from .stereo import rectify_pair


def flash_frame(frames) -> int:
    m = np.array([f.mean() for f in frames])
    return int(np.argmax(np.diff(m))) + 1 if len(m) > 1 else 0


def sync_offset(left_frames, right_frames) -> int:
    """right_index = left_index + offset."""
    return flash_frame(right_frames) - flash_frame(left_frames)


def disparity(rL, rR, num_disp=128, block=5):
    gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY) if rL.ndim == 3 else rL
    gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY) if rR.ndim == 3 else rR
    num_disp = int(np.ceil(num_disp / 16) * 16)
    sgbm = cv2.StereoSGBM_create(0, num_disp, block, 8 * block * block, 32 * block * block, 1, 63, 10, 100, 2,
                                 cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    return sgbm.compute(gL, gR).astype(np.float32) / 16.0


def cloud(disp, Q, step=2, max_depth=None):
    pts = cv2.reprojectImageTo3D(disp, Q)[::step, ::step].reshape(-1, 3)
    d = disp[::step, ::step].ravel()
    ok = (d > 1) & np.isfinite(pts).all(1)
    pts = pts[ok]
    if max_depth:
        pts = pts[pts[:, 2] < max_depth]
    return pts


def ransac_plane(P, thr, iters=400, rng=None):
    rng = rng or np.random.default_rng(0)
    best, best_n = None, 0
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((P - s[0]) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    Pi = P[best]
    c = Pi.mean(0)
    n = np.linalg.svd(Pi - c)[2][2]
    return n, float(n @ c), best


def find_planes(P, thr=None, max_planes=6, min_frac=0.04):
    thr = thr or 0.01 * np.median(np.linalg.norm(P, axis=1))
    planes, rest = [], P
    for _ in range(max_planes):
        if len(rest) < max(50, min_frac * len(P)):
            break
        n, d, inl = ransac_plane(rest, thr)
        if inl.sum() < min_frac * len(P):
            break
        if d < 0:
            n, d = -n, -d
        planes.append({"n": n, "d": d, "count": int(inl.sum())})
        rest = rest[~inl]
    return planes


def opposite_pairs(planes, max_angle_deg=12):
    pairs = []
    cosmax = np.cos(np.deg2rad(max_angle_deg))
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            a, b = planes[i], planes[j]
            c = a["n"] @ b["n"]
            if abs(c) > cosmax:
                # planes n.X = d; opposite sides of the camera have normals pointing opposite ways
                gap = abs(a["d"] - np.sign(c) * b["d"])
                pairs.append({"planes": (i, j), "gap": float(gap), "normal": a["n"]})
    return pairs


def room_from_pair(rig, imgL, imgR, num_disp=128):
    rL, rR, geo = rectify_pair(rig, imgL, imgR)
    disp = disparity(rL, rR, num_disp)
    P = cloud(disp, geo["Q"])
    planes = find_planes(P) if len(P) > 200 else []
    return {"rectL": rL, "rectR": rR, "disparity": disp, "points": P, "planes": planes,
            "pairs": opposite_pairs(planes), "geo": geo}
