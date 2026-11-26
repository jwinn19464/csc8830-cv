"""Module 2 - real-world 2-D dimensions from one photo with perspective projection.

Pinhole model (after undistortion):  u = fx * X / Z + cx,   v = fy * Y / Z + cy
Inverting for a planar object facing the camera at distance Z:
    X = (u - cx) * Z / fx,   Y = (v - cy) * Z / fy
So the real distance between two image points is
    D = Z * sqrt( ((u1 - u2) / fx)^2 + ((v1 - v2) / fy)^2 )

    python -m module2.measure --calib module2/calibration.npz --image book.jpg --z 2500 --points 812,400 1490,410
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from .calibrate import load_calibration, scale_K


def undistort(img, K, dist):
    return cv2.undistort(img, K, dist)


def undistort_points(pts, K, dist):
    """Distorted pixel coords -> undistorted pixel coords (same K)."""
    p = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, dist, P=K).reshape(-1, 2)


def pixel_to_world(u, v, Z, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * Z / fx, (v - cy) * Z / fy, Z])


def dimension(p1, p2, Z, K):
    """Distance between two (undistorted) pixel points on a plane at depth Z (same unit as Z)."""
    return float(np.linalg.norm(pixel_to_world(*p1, Z, K) - pixel_to_world(*p2, Z, K)))


def polyline_lengths(pts, Z, K, closed=False):
    pts = list(pts)
    pairs = list(zip(pts, pts[1:] + (pts[:1] if closed else [])))
    return [dimension(a, b, Z, K) for a, b in pairs]


def K_for_image(calib, img_shape):
    return scale_K(calib["K"], calib["size"], (img_shape[1], img_shape[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--z", type=float, required=True, help="distance lens -> object plane (mm)")
    ap.add_argument("--points", nargs=2, required=True, help="u1,v1 u2,v2 in the original photo")
    a = ap.parse_args()
    c = load_calibration(a.calib)
    img = cv2.imread(a.image)
    K = K_for_image(c, img.shape)
    pts = undistort_points([tuple(map(float, p.split(","))) for p in a.points], K, c["dist"])
    print(f"dimension = {dimension(pts[0], pts[1], a.z, K):.1f} mm")


if __name__ == "__main__":
    main()
