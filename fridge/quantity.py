"""Fridge Oracle step 4 - size and how much is left (Modules 2 + 7).

* Metric scale (Module 2): a reference of known width in the shot (a credit-card-sized
  card is 85.6 mm) gives mm per pixel at that depth, so item heights come out in mm.
  With a calibrated phone and a measured distance, Module 2's projection formula is used.
* Fill level: for a bottle or jar seen from the side, the liquid line is the strongest
  horizontal change in the row-mean intensity profile (a 1-D edge).
* Hidden items (Module 7): with a stereo pair, items whose depth is well behind the front
  row are flagged as partly hidden.
"""
from __future__ import annotations

import cv2
import numpy as np


def mm_per_px_from_reference(ref_width_px: float, ref_width_mm: float = 85.6) -> float:
    return ref_width_mm / ref_width_px


def fill_level(crop_bgr) -> tuple[float, np.ndarray]:
    """Fraction of the container that is full (0..1) and the smoothed row profile."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = g.shape
    core = g[:, int(0.3 * w):max(int(0.3 * w) + 1, int(0.7 * w))]
    prof = cv2.GaussianBlur(core.mean(1).reshape(-1, 1), (1, 9), 0).ravel()
    lo, hi = int(0.08 * h), int(0.95 * h)
    d = np.abs(np.diff(prof))[lo:hi]
    if len(d) == 0:
        return 1.0, prof
    line = lo + int(np.argmax(d))
    return float(np.clip(1 - line / h, 0, 1)), prof


def depth_layers(depths, gap_frac=0.15):
    """Flag items noticeably behind the nearest item (partly hidden in the back row)."""
    z = np.asarray(depths, float)
    front = np.nanmin(z)
    return z > front * (1 + gap_frac)
