"""Fridge Oracle - cleaning up dim, noisy, blurry frames before anything else (Modules 3 + 6).

Low light makes the camera use a longer shutter (motion blur) and more gain (noise).

* Multi-frame stacking (Module 6): align the neighbouring frames to the current one with an
  ORB + RANSAC homography and average them. Noise is random from frame to frame, so averaging
  N aligned frames cuts it by about sqrt(N); the labels stay put because they were aligned.
* Denoise + contrast: non-local means denoising, then CLAHE on the lightness channel only (so
  colours don't shift) and a gamma lift when the frame is dark.
* Motion blur (Modules 3 + 6): the optical flow between neighbouring frames gives the blur's
  direction and length (the camera moved about that far while the shutter was open), which is
  turned into a line-shaped blur kernel and undone with Wiener deconvolution.
"""
from __future__ import annotations

import cv2
import numpy as np

from module3.blur import motion_kernel


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer's fast noise estimate: a Laplacian-like mask cancels image structure, leaving noise."""
    g = gray.astype(np.float64)
    M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    h, w = g.shape
    conv = cv2.filter2D(g, -1, M)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2)))


def homography(src, dst, orb=None):
    """H mapping src onto dst, or None if they can't be matched."""
    orb = orb or cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 12:
        return None
    H, inl = cv2.findHomography(np.float32([ka[m.queryIdx].pt for m in good]),
                                np.float32([kb[m.trainIdx].pt for m in good]), cv2.RANSAC, 3.0)
    return H if H is not None and inl.sum() >= 10 else None


def ecc_homography(src, dst):
    """Fallback alignment for small moves: ECC maximises the correlation of the two frames'
    brightness directly, so it works where feature matching finds too few keypoints."""
    a = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = min(1.0, 320 / max(a.shape))
    a_s, b_s = cv2.resize(a, None, fx=f, fy=f), cv2.resize(b, None, fx=f, fy=f)
    W = np.eye(3, dtype=np.float32)
    try:
        cc, W = cv2.findTransformECC(b_s, a_s, W, cv2.MOTION_HOMOGRAPHY,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5), None, 5)
    except cv2.error:
        return None
    if cc < 0.6:
        return None
    S = np.diag([f, f, 1.0])
    # W maps dst coords to src coords at the small scale; invert and rescale to get src -> dst
    return np.linalg.inv(S) @ np.linalg.inv(W.astype(np.float64)) @ S


def stack(frames, i, radius=2):
    """Average frame i with up to `radius` aligned neighbours on each side."""
    ref = frames[i]
    h, w = ref.shape[:2]
    acc = ref.astype(np.float32)
    cnt = np.ones((h, w, 1), np.float32)
    orb = cv2.ORB_create(1500)
    used = 1
    for j in range(max(0, i - radius), min(len(frames), i + radius + 1)):
        if j == i:
            continue
        H = homography(frames[j], ref, orb)
        if H is None:
            H = ecc_homography(frames[j], ref)
        if H is None:
            continue
        warped = cv2.warpPerspective(frames[j], H, (w, h))
        valid = cv2.warpPerspective(np.ones((h, w), np.uint8), H, (w, h))[..., None].astype(np.float32)
        # skip a neighbour that disagrees a lot (bad alignment or something moved)
        diff = np.abs(warped.astype(np.float32) - ref.astype(np.float32)).mean(2, keepdims=True)
        valid *= (diff < 40).astype(np.float32)
        acc += warped.astype(np.float32) * valid
        cnt += valid
        used += 1
    return np.clip(acc / cnt, 0, 255).astype(np.uint8), used


def lift(img):
    """Gamma on the lightness channel so a dark frame's mean lands near 0.45 (colours untouched)."""
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    mean = L.mean()
    if mean < 110:
        gamma = np.log(0.45) / np.log(max(mean, 5) / 255)
        lut = np.clip(255 * (np.arange(256) / 255) ** gamma, 0, 255).astype(np.uint8)
        L = cv2.LUT(L, lut)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def brighten(img, denoise=True):
    """Brighten first (so the noise is at its final size), then denoise with a strength set from the
    measured noise, then a gentle CLAHE on lightness."""
    out = lift(img)
    if denoise:
        s = noise_sigma(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY))
        if s > 1.5:
            hh = float(np.clip(0.8 * s, 2, 12))  # gentle: strong settings smear small print
            out = cv2.fastNlMeansDenoisingColored(out, None, hh, hh, 5, 15)
    L, a, b = cv2.split(cv2.cvtColor(out, cv2.COLOR_BGR2LAB))
    L = cv2.createCLAHE(1.5, (8, 8)).apply(L)
    return cv2.cvtColor(cv2.merge([L, a, b]), cv2.COLOR_LAB2BGR)


def motion_psf(frames, i, exposure=0.5):
    """Blur kernel for frame i from the median optical flow to its neighbour.
    exposure = fraction of the frame interval the shutter is open (about half in dim light)."""
    j = i + 1 if i + 1 < len(frames) else i - 1
    if j < 0:
        return None
    a = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(frames[j], cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    dx, dy = np.median(fl.reshape(-1, 2), axis=0)
    length = exposure * float(np.hypot(dx, dy))
    if length < 2:
        return None
    size = int(min(31, round(length))) | 1
    return motion_kernel(max(3, size), float(np.degrees(np.arctan2(-dy, dx))))


def clean_keyframes(frames, keys, stack_radius=2):
    """Stacked + brightened keyframes and their motion-blur kernels."""
    out, psfs, info = [], [], []
    for k in keys:
        st, used = stack(frames, k, stack_radius) if stack_radius else (frames[k], 1)
        out.append(brighten(st))
        psfs.append(motion_psf(frames, k))
        info.append(used)
    return out, psfs, info
