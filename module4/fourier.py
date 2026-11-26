"""Module 4 Q3 - edge detection and region segmentation in the Fourier domain.

Derivative theorem:  F{df/dx}(u, v) = j 2 pi u F(u, v)
  -> gradient by multiplying the spectrum by j2(pi)u and j2(pi)v (with a Gaussian to tame noise)
Laplacian:           F{lap f} = -4 pi^2 (u^2 + v^2) F
  -> LoG edges are zero crossings of IFFT( -4pi^2 |w|^2 G_sigma(w) F )
High-pass:           1 - Gaussian low-pass keeps edges and fine texture
Region segmentation: low-pass (keep coarse regions) then Otsu threshold
"""
from __future__ import annotations

import cv2
import numpy as np


def _freq(shape):
    h, w = shape
    v = np.fft.fftfreq(h)[:, None]
    u = np.fft.fftfreq(w)[None, :]
    return u, v


def gaussian_lp(shape, sigma_px):
    """Frequency response of a spatial Gaussian of std sigma_px."""
    u, v = _freq(shape)
    return np.exp(-2 * (np.pi ** 2) * (sigma_px ** 2) * (u ** 2 + v ** 2))


def gradient_fft(g, sigma=1.5):
    F = np.fft.fft2(g.astype(np.float64))
    u, v = _freq(g.shape)
    G = gaussian_lp(g.shape, sigma)
    gx = np.real(np.fft.ifft2(1j * 2 * np.pi * u * G * F))
    gy = np.real(np.fft.ifft2(1j * 2 * np.pi * v * G * F))
    return np.hypot(gx, gy)


def log_fft(g, sigma=2.0):
    F = np.fft.fft2(g.astype(np.float64))
    u, v = _freq(g.shape)
    L = -4 * np.pi ** 2 * (u ** 2 + v ** 2) * gaussian_lp(g.shape, sigma)
    r = np.real(np.fft.ifft2(L * F))
    zc = np.zeros(g.shape, bool)
    thr = 0.02 * np.abs(r).max()
    for dy, dx in ((0, 1), (1, 0)):
        s = np.roll(r, (-dy, -dx), (0, 1))
        zc |= (np.sign(r) != np.sign(s)) & (np.abs(r - s) > thr)
    return r, np.uint8(zc) * 255


def highpass_fft(g, sigma=3.0):
    F = np.fft.fft2(g.astype(np.float64))
    return np.abs(np.real(np.fft.ifft2((1 - gaussian_lp(g.shape, sigma)) * F)))


def lowpass_regions(g, sigma=6.0):
    F = np.fft.fft2(g.astype(np.float64))
    smooth = np.real(np.fft.ifft2(gaussian_lp(g.shape, sigma) * F))
    s8 = norm8(smooth)
    _, m = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return s8, m


def norm8(x):
    x = np.asarray(x, np.float64)
    return np.uint8(255 * (x - x.min()) / (np.ptp(x) + 1e-12))


def run_all(g, sigma=2.0):
    grad = gradient_fft(g, sigma)
    _, zc = log_fft(g, sigma)
    _, edges = cv2.threshold(norm8(grad), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth, regions = lowpass_regions(g, 3 * sigma)
    return {"gradient magnitude (j2πu·F)": norm8(grad), "gradient edges (Otsu)": edges,
            "LoG zero crossings": zc, "high-pass": norm8(highpass_fft(g, sigma)),
            "low-pass": smooth, "low-pass regions (Otsu)": regions}
