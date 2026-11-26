"""Fridge Oracle step 1 - keyframes from a slow sweep video (Modules 3 + 6).

* Sharpness (Module 3): fraction of spectral energy above a radial frequency cut-off.
  Motion blur is a low-pass filter, so blurred frames lose high-frequency energy.
* Redundancy (Modules 3 + 6): accumulate the frame-to-frame shift (phase correlation, with
  Farneback flow as the fallback); start a new segment once the
  view has moved by `move_frac` of the frame width. Keep the sharpest frame per segment.
"""
from __future__ import annotations

import cv2
import numpy as np


def fft_sharpness(gray: np.ndarray, cutoff: float = 0.15) -> float:
    g = cv2.resize(gray, (256, 256)).astype(np.float64)
    g -= g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g * np.outer(np.hanning(256), np.hanning(256))))) ** 2
    yy, xx = np.mgrid[-128:128, -128:128] / 256.0
    r = np.hypot(xx, yy)
    return float(P[r > cutoff].sum() / (P.sum() + 1e-12))


def select_keyframes(frames, move_frac=0.35):
    if not frames:
        return [], []
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    sharp = np.array([fft_sharpness(g) for g in grays])
    w = grays[0].shape[1]
    segments, cur, moved = [], [0], 0.0
    hh, ww = grays[0].shape
    win = cv2.createHanningWindow((ww // 2, hh // 2), cv2.CV_32F)
    for i in range(1, len(grays)):
        a = cv2.resize(grays[i - 1], None, fx=0.5, fy=0.5)
        b = cv2.resize(grays[i], None, fx=0.5, fy=0.5)
        # global image motion by phase correlation (Fourier shift theorem): a shift is a phase ramp
        (dx, dy), resp = cv2.phaseCorrelate(np.float32(a), np.float32(b), win)
        if resp < 0.05:  # weak peak: fall back to dense optical flow on textured pixels
            fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.2, 0)
            gm = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
            tex = gm > np.percentile(gm, 85)
            dx, dy = np.median(fl[tex], axis=0) if tex.sum() > 50 else (0.0, 0.0)
        moved += 2 * float(np.hypot(dx, dy))
        cur.append(i)
        if moved > move_frac * w:
            segments.append(cur)
            cur, moved = [], 0.0
    if cur:
        segments.append(cur)
    # the sharpest frame of each segment; a segment is never dropped, so the sweep stays covered
    keys = [max(seg, key=lambda k: sharp[k]) for seg in segments]
    return keys, sharp.tolist()
