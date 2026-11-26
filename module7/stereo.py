"""Module 7 - object size from CALIBRATED stereo: estimate depth Z first, then real dimensions.

Rig options
  A. Two phones (or one phone in two positions) calibrated together with a checkerboard:
     cv2.stereoCalibrate -> K1, d1, K2, d2, R, T.  Baseline B = |T|.
  B. One phone with K from Module 2, moved sideways by a measured baseline B:
     essential matrix from matched features -> R and the direction of T; |T| = B fixes scale.

Rectified stereo: same row in both images, disparity d = uL - uR,
    Z = f B / d,  X = (uL - cx) Z / f,  Y = (v - cy) Z / f
General case: linear triangulation of x1 ~ P1 X, x2 ~ P2 X (cv2.triangulatePoints).
Shapes: rectangle (4 corners -> sides, area), circle (centre + rim point(s) -> diameter),
polygon (any number of vertices -> side lengths, perimeter, area).
"""
from __future__ import annotations

import io

import cv2
import numpy as np


# ---------- rig ----------------------------------------------------------------------------
def stereo_calibrate(left_imgs, right_imgs, pattern=(9, 6), square=25.0):
    from module2.calibrate import board_points, find_corners
    obj = board_points(pattern, square)
    O, L, R = [], [], []
    size = left_imgs[0].shape[1], left_imgs[0].shape[0]
    for a, b in zip(left_imgs, right_imgs):
        ca, cb = find_corners(a, pattern), find_corners(b, pattern)
        if ca is not None and cb is not None:
            O.append(obj); L.append(ca); R.append(cb)
    if len(O) < 3:
        raise ValueError(f"Board found in both images of {len(O)} pair(s); need at least 3.")
    _, K1, d1, _, _ = cv2.calibrateCamera(O, L, size, None, None)
    _, K2, d2, _, _ = cv2.calibrateCamera(O, R, size, None, None)
    rms, K1, d1, K2, d2, Rm, T, E, F = cv2.stereoCalibrate(
        O, L, R, K1, d1, K2, d2, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return {"K1": K1, "d1": d1.ravel(), "K2": K2, "d2": d2.ravel(), "R": Rm, "T": T.ravel(),
            "size": size, "rms": float(rms), "pairs": len(O)}


def rig_from_motion(imgL, imgR, K, dist, baseline):
    """Option B: one calibrated phone, two positions, measured baseline (same unit as output)."""
    from module6.sfm import match_tracks
    t = match_tracks([imgL, imgR], n=4000)
    if len(t) < 15:
        raise ValueError("Too few matching features; photograph a textured scene.")
    p1 = cv2.undistortPoints(t[:, 0].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    p2 = cv2.undistortPoints(t[:, 1].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, T, _ = cv2.recoverPose(E, p1, p2, K, mask=inl)
    T = T.ravel() / np.linalg.norm(T) * baseline
    size = imgL.shape[1], imgL.shape[0]
    return {"K1": K, "d1": np.ravel(dist), "K2": K, "d2": np.ravel(dist), "R": R, "T": T,
            "size": size, "rms": float("nan"), "pairs": 0, "matches": int(inl.sum())}


def save_rig(rig) -> bytes:
    b = io.BytesIO()
    np.savez(b, **{k: np.asarray(v) for k, v in rig.items()})
    return b.getvalue()


def load_rig(data) -> dict:
    d = np.load(io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data)
    r = {k: d[k] for k in d.files}
    r["size"] = tuple(int(x) for x in r["size"])
    return r


def baseline(rig):
    return float(np.linalg.norm(rig["T"]))


# ---------- geometry -------------------------------------------------------------------------
def rectify_pair(rig, imgL, imgR):
    size = imgL.shape[1], imgL.shape[0]
    f = lambda k: np.asarray(rig[k], np.float64)  # noqa: E731
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(f("K1"), f("d1").reshape(-1, 1), f("K2"), f("d2").reshape(-1, 1),
                                                size, f("R"), f("T").reshape(3, 1), alpha=0)
    m1 = cv2.initUndistortRectifyMap(f("K1"), f("d1"), R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(f("K2"), f("d2"), R2, P2, size, cv2.CV_32FC1)
    rL = cv2.remap(imgL, *m1, cv2.INTER_LINEAR)
    rR = cv2.remap(imgR, *m2, cv2.INTER_LINEAR)
    return rL, rR, {"P1": P1, "P2": P2, "Q": Q, "R1": R1}


def depth_from_disparity(d, f, B):
    return f * B / np.asarray(d, float)


def rectified_to_3d(uL, v, disp, P1, P2):
    f, cx, cy = P1[0, 0], P1[0, 2], P1[1, 2]
    B = -P2[0, 3] / P2[0, 0]
    Z = depth_from_disparity(disp, f, B)
    return np.stack([(np.asarray(uL) - cx) * Z / f, (np.asarray(v) - cy) * Z / f, Z], -1)


def match_along_row(rL, rR, pt, half=12, max_disp=None):
    """Find pt (in rectified left) in the rectified right image on the same row (NCC)."""
    gL = cv2.cvtColor(rL, cv2.COLOR_BGR2GRAY) if rL.ndim == 3 else rL
    gR = cv2.cvtColor(rR, cv2.COLOR_BGR2GRAY) if rR.ndim == 3 else rR
    h, w = gL.shape
    u, v = int(round(pt[0])), int(round(pt[1]))
    u, v = np.clip(u, half, w - half - 1), np.clip(v, half, h - half - 1)
    tpl = gL[v - half:v + half + 1, u - half:u + half + 1]
    lo = max(0, u - (max_disp or w) - half)
    strip = gR[v - half:v + half + 1, lo:u + half + 1]
    if strip.shape[1] <= tpl.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(strip, tpl, cv2.TM_CCOEFF_NORMED)[0]
    k = int(np.argmax(res))
    # sub-pixel parabola
    off = 0.0
    if 0 < k < len(res) - 1:
        a, b, c = res[k - 1], res[k], res[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den else 0.0
    uR = lo + k + off + half
    return (uR, v), float(res[k])


def points_3d(rL, rR, ptsL, geo):
    out, scores, ptsR = [], [], []
    for p in ptsL:
        q, s = match_along_row(rL, rR, p)
        if q is None or p[0] - q[0] <= 0:
            out.append([np.nan] * 3); scores.append(0.0); ptsR.append((np.nan, np.nan))
            continue
        out.append(rectified_to_3d(p[0], p[1], p[0] - q[0], geo["P1"], geo["P2"]))
        scores.append(s); ptsR.append(q)
    return np.array(out), np.array(scores), ptsR


def measure_shape(P, shape):
    P = np.asarray(P, float)
    if shape == "circle":
        c = P[0]
        r = np.linalg.norm(P[1:] - c, axis=1).mean()
        return {"diameter": 2 * r, "circumference": 2 * np.pi * r, "area": np.pi * r * r, "depth Z": c[2]}
    sides = np.linalg.norm(np.roll(P, -1, 0) - P, axis=0 if P.ndim == 1 else 1)
    # polygon area in 3-D: half the norm of the summed cross products
    area = 0.5 * np.linalg.norm(np.sum(np.cross(P, np.roll(P, -1, 0)), axis=0))
    out = {f"side {i + 1}": s for i, s in enumerate(sides)}
    out.update({"perimeter": sides.sum(), "area": area, "mean depth Z": P[:, 2].mean()})
    if shape == "rectangle" and len(P) == 4:
        out["width (avg of opposite sides)"] = (sides[0] + sides[2]) / 2
        out["height (avg of opposite sides)"] = (sides[1] + sides[3]) / 2
    return out
