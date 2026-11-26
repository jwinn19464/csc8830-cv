"""Module 2 - camera calibration with a checkerboard (Zhang's method, OpenCV).

    python -m module2.calibrate --images "data/calib/*.jpg" --pattern 9x6 --square 25 --out module2/calibration.npz

--pattern is the number of INNER corners (a 10 x 7 square board has 9 x 6).
Good calibrations have an RMS reprojection error below about 0.5 px.
"""
from __future__ import annotations

import argparse
import glob
import io
import json

import cv2
import numpy as np


def find_corners(img_bgr, pattern=(9, 6)):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(g, pattern, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)


def board_points(pattern=(9, 6), square=25.0):
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    return obj


def calibrate_images(images, pattern=(9, 6), square=25.0):
    """images: list of BGR arrays, all the same size. Returns a dict with K, dist and errors."""
    obj = board_points(pattern, square)
    objpts, imgpts, used, overlays = [], [], [], []
    size = None
    for i, im in enumerate(images):
        if size is None:
            size = im.shape[1], im.shape[0]
        if (im.shape[1], im.shape[0]) != size:
            continue
        c = find_corners(im, pattern)
        if c is None:
            continue
        objpts.append(obj)
        imgpts.append(c)
        used.append(i)
        ov = im.copy()
        cv2.drawChessboardCorners(ov, pattern, c, True)
        overlays.append(ov)
    if len(used) < 3:
        raise ValueError(f"Board found in {len(used)} image(s); need at least 3 (10+ is better).")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    per = []
    for o, p, r, t in zip(objpts, imgpts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((proj - p) ** 2, axis=2)))))
    return {"K": K, "dist": dist.ravel(), "rms": float(rms), "size": size, "used": used,
            "per_image_err": per, "overlays": overlays, "rvecs": rvecs, "tvecs": tvecs}


def scale_K(K, from_size, to_size):
    """Intrinsics for the same camera at a different image resolution (same aspect ratio)."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    K2 = K.copy().astype(float)
    K2[0] *= sx
    K2[1] *= sy
    K2[2] = [0, 0, 1]
    return K2


def to_npz_bytes(c) -> bytes:
    b = io.BytesIO()
    np.savez(b, K=c["K"], dist=c["dist"], size=np.array(c["size"]), rms=c["rms"])
    return b.getvalue()


def load_calibration(src):
    d = np.load(src if isinstance(src, str) else io.BytesIO(src))
    return {"K": d["K"], "dist": d["dist"], "size": tuple(int(x) for x in d["size"]), "rms": float(d["rms"])}


def to_json(c) -> str:
    return json.dumps({"K": np.asarray(c["K"]).tolist(), "dist": np.asarray(c["dist"]).tolist(),
                       "size": list(c["size"]), "rms": c["rms"]}, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--square", type=float, default=25.0)
    ap.add_argument("--out", default="module2/calibration.npz")
    a = ap.parse_args()
    pat = tuple(int(x) for x in a.pattern.lower().split("x"))
    files = sorted(glob.glob(a.images))
    c = calibrate_images([cv2.imread(f) for f in files], pat, a.square)
    open(a.out, "wb").write(to_npz_bytes(c))
    print(f"used {len(c['used'])}/{len(files)} images, RMS {c['rms']:.3f} px")
    print("K =\n", np.round(c["K"], 2), "\ndist =", np.round(c["dist"], 4))


if __name__ == "__main__":
    main()
