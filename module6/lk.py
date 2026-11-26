"""Module 6 Part A - Lucas-Kanade tracking and bilinear interpolation, from scratch.

Brightness constancy:  I(x + u, y + v, t + 1) = I(x, y, t)
First-order Taylor:    Ix u + Iy v + It = 0          (one equation, two unknowns)
Assume (u, v) is constant in a window W and solve the least squares problem
    [ sum Ix^2   sum IxIy ] [u]     [ sum IxIt ]
    [ sum IxIy   sum Iy^2 ] [v] = - [ sum IyIt ]
Iterate (Newton-Raphson): warp the second image by the current estimate with bilinear
interpolation, recompute It, solve for an update. Pyramids handle large motions.

Bilinear interpolation at (x, y) with x0 = floor(x), a = x - x0, b = y - y0:
    I = (1-a)(1-b) I[y0, x0] + a(1-b) I[y0, x0+1] + (1-a) b I[y0+1, x0] + a b I[y0+1, x0+1]
"""
from __future__ import annotations

import cv2
import numpy as np


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    a, b = x - x0, y - y0
    return ((1 - a) * (1 - b) * img[y0, x0] + a * (1 - b) * img[y0, x0 + 1]
            + (1 - a) * b * img[y0 + 1, x0] + a * b * img[y0 + 1, x0 + 1])


def gradients(img):
    k = np.array([[-1, 0, 1]], np.float64) / 2
    return cv2.filter2D(img, -1, k), cv2.filter2D(img, -1, k.T)


def lk_point(I0, I1, Ix, Iy, p, guess, half=7, iters=20, eps=0.01):
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    xs, ys = p[0] + dx.ravel(), p[1] + dy.ravel()
    T = bilinear(I0, xs, ys)
    gx, gy = bilinear(Ix, xs, ys), bilinear(Iy, xs, ys)
    G = np.array([[gx @ gx, gx @ gy], [gx @ gy, gy @ gy]])
    if np.linalg.eigvalsh(G)[0] < 1e-3 * len(xs):
        return guess, False
    Ginv = np.linalg.inv(G)
    d = np.array(guess, float)
    for _ in range(iters):
        It = bilinear(I1, xs + d[0], ys + d[1]) - T
        step = -Ginv @ np.array([gx @ It, gy @ It])
        d += step
        if np.hypot(*step) < eps:
            break
    return d, True


def track(img0, img1, pts, levels=3, half=7):
    """Pyramidal LK. pts: (N, 2) in img0. Returns (pts in img1, ok mask)."""
    I0 = img0.astype(np.float64) if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float64)
    I1 = img1.astype(np.float64) if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    pyr0, pyr1 = [I0], [I1]
    for _ in range(levels - 1):
        pyr0.append(cv2.pyrDown(pyr0[-1]))
        pyr1.append(cv2.pyrDown(pyr1[-1]))
    grads = [gradients(p) for p in pyr0]
    pts = np.asarray(pts, float)
    out, ok = np.zeros_like(pts), np.ones(len(pts), bool)
    for i, p in enumerate(pts):
        d = np.zeros(2)
        good = True
        for L in range(levels - 1, -1, -1):
            s = 2 ** L
            d, g = lk_point(pyr0[L], pyr1[L], *grads[L], p / s, d, half)
            good &= g if L == 0 else True
            if L:
                d = d * 2
        out[i] = p + d
        ok[i] = good
    return out, ok


def good_features(img, n=60):
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p = cv2.goodFeaturesToTrack(g, n, 0.01, 12)
    return np.zeros((0, 2)) if p is None else p.reshape(-1, 2)


def validate(img0, img1, pts, levels=3, half=7):
    """Compare our LK with OpenCV's, and check photometric agreement of predicted locations."""
    ours, ok = track(img0, img1, pts, levels, half)
    g0 = img0 if img0.ndim == 2 else cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    g1 = img1 if img1.ndim == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cv_pts, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, np.float32(pts).reshape(-1, 1, 2), None,
                                             winSize=(2 * half + 1, 2 * half + 1), maxLevel=levels - 1)
    cv_pts = cv_pts.reshape(-1, 2)
    both = ok & (st.ravel() == 1)
    diff = np.linalg.norm(ours - cv_pts, axis=1)
    # patch error at predicted location (lower = the patch really moved there)
    G0, G1 = g0.astype(float), g1.astype(float)
    dy, dx = np.mgrid[-half:half + 1, -half:half + 1]
    zero_err, pred_err = [], []
    for p, q in zip(pts, ours):
        a = bilinear(G0, p[0] + dx.ravel(), p[1] + dy.ravel())
        zero_err.append(np.abs(bilinear(G1, p[0] + dx.ravel(), p[1] + dy.ravel()) - a).mean())
        pred_err.append(np.abs(bilinear(G1, q[0] + dx.ravel(), q[1] + dy.ravel()) - a).mean())
    return {"ours": ours, "opencv": cv_pts, "ok": both, "diff_px": diff,
            "patch_err_no_motion": np.array(zero_err), "patch_err_predicted": np.array(pred_err)}
