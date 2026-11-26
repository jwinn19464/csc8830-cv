"""Module 3 - blurring by spatial convolution and in the Fourier domain, both from scratch.

Convolution theorem:  f * h  =  IFFT( FFT(f) . FFT(h) )
The DFT product is a CIRCULAR convolution, so the spatial version here wraps around the
image edges too; with identical boundary handling the two agree to floating-point
precision (MSE ~ 1e-25).

Rule of thumb: a Gaussian needs size ~ 6 * sigma. With sigma much larger than size/6 the
kernel is cut off, behaves like a box filter, and its spectrum shows sinc ringing.

    python -m module3.blur --image photo.jpg --size 15 --sigma 2.5 --kind gaussian
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size) - (size - 1) / 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def box_kernel(size: int) -> np.ndarray:
    return np.full((size, size), 1.0 / size ** 2)


def motion_kernel(size: int, angle_deg: float = 0.0) -> np.ndarray:
    k = np.zeros((size, size))
    c = (size - 1) / 2
    t = np.deg2rad(angle_deg)
    for s in np.linspace(-c, c, 4 * size):
        x, y = int(round(c + s * np.cos(t))), int(round(c - s * np.sin(t)))
        k[y, x] = 1
    return k / k.sum()


def make_kernel(kind: str, size: int, sigma: float = 1.0, angle: float = 0.0) -> np.ndarray:
    return {"gaussian": lambda: gaussian_kernel(size, sigma), "box": lambda: box_kernel(size),
            "motion": lambda: motion_kernel(size, angle)}[kind]()


def conv2d_spatial(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """True convolution (kernel flipped) with circular boundary, written with shifted sums.

    g[y, x] = sum_{i, j} h[i, j] * f[y - (i - ci), x - (j - cj)]
    Loops over the kernel taps (not the pixels) so it stays fast in numpy.
    """
    f = f.astype(np.float64)
    kh, kw = h.shape
    ci, cj = kh // 2, kw // 2
    out = np.zeros_like(f)
    for i in range(kh):
        for j in range(kw):
            w = h[i, j]
            if w != 0:
                out += w * np.roll(f, shift=(i - ci, j - cj), axis=(0, 1))
    return out


def pad_kernel(h: np.ndarray, shape) -> np.ndarray:
    """Zero-pad the kernel to the image size with its centre moved to (0, 0)."""
    H = np.zeros(shape[:2])
    kh, kw = h.shape
    H[:kh, :kw] = h
    return np.roll(H, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))


def conv2d_fourier(f: np.ndarray, h: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(f.astype(np.float64), axes=(0, 1))
    Hf = np.fft.fft2(pad_kernel(h, f.shape))
    if f.ndim == 3:
        Hf = Hf[:, :, None]
    return np.real(np.fft.ifft2(F * Hf, axes=(0, 1)))


def apply_per_channel(fn, img, h):
    if img.ndim == 2:
        return fn(img, h)
    return np.stack([fn(img[:, :, c], h) for c in range(img.shape[2])], axis=2)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    mse = float(np.mean((a - b) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))
    return {"MSE": mse, "max |diff|": float(np.max(np.abs(a - b))), "PSNR (dB)": psnr}


def log_spectrum(x: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum scaled to 0-255 for display."""
    m = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(x))))
    return np.uint8(255 * (m - m.min()) / (np.ptp(m) + 1e-12))


def to_u8(x):
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--kind", default="gaussian", choices=["gaussian", "box", "motion"])
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--sigma", type=float, default=2.5)
    a = ap.parse_args()
    img = cv2.imread(a.image)
    h = make_kernel(a.kind, a.size, a.sigma)
    s = apply_per_channel(conv2d_spatial, img, h)
    f = apply_per_channel(conv2d_fourier, img, h)
    print(compare(s, f))
    cv2.imwrite("blur_spatial.png", to_u8(s))
    cv2.imwrite("blur_fourier.png", to_u8(f))


if __name__ == "__main__":
    main()
