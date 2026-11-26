"""Module 6 Part B - structure from motion from 4 photos of a PLANAR object, with K from Module 2.

1. Points: ORB features in view 1 matched into views 2-4 (or taps), kept if seen in all 4.
2. Homographies H_k (view 1 -> view k), decomposed with K:  H ~ K (R + t n^T / d) K^-1.
   Of the up-to-4 solutions, keep those with the plane in front of view 1 and pick the
   combination whose plane normals n agree best across views (the plane is the same one).
3. Initial 3-D points: ray-plane intersection in view 1, X = d m / (n . m), m = K^-1 [u v 1].
4. Bundle adjustment: refine all poses and points by minimising reprojection error in all
   4 views (scipy least_squares, Huber loss). View 1 is fixed at [I | 0]; d = 1 fixes scale.
5. Metric scale from one known length; the boundary is the convex hull of the points on
   the fitted plane (or 4 corners you tap in view 1).
"""
from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares


def match_tracks(imgs, n=3000, ratio=0.75):
    orb = cv2.ORB_create(n)
    gs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) if i.ndim == 3 else i for i in imgs]
    kd = [orb.detectAndCompute(g, None) for g in gs]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    k0, d0 = kd[0]
    maps = []
    for k, d in kd[1:]:
        m = {}
        for pair in bf.knnMatch(d0, d, k=2):
            if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                m[pair[0].queryIdx] = k[pair[0].trainIdx].pt
        maps.append(m)
    common = set(maps[0]).intersection(*maps[1:])
    tracks = [[k0[i].pt] + [mp[i] for mp in maps] for i in sorted(common)]
    t = np.array(tracks, float).reshape(-1, len(imgs), 2)
    # keep only matches consistent with a homography in every view
    keep = np.ones(len(t), bool)
    for v in range(1, len(imgs)):
        if len(t) >= 4:
            _, inl = cv2.findHomography(t[:, 0], t[:, v], cv2.RANSAC, 4.0)
            keep &= inl.ravel().astype(bool)
    return t[keep]


def decompose(H, K, m0):
    n_sol, Rs, ts, ns = cv2.decomposeHomographyMat(H, K)
    sols = []
    for R, t, n in zip(Rs, ts, ns):
        n = n.ravel()
        if np.all(m0 @ n > 0):  # plane in front of camera 1
            sols.append((R, t.ravel(), n))
    return sols


def project(K, R, t, X):
    x = (K @ (R @ X.T + t[:, None])).T
    return x[:, :2] / x[:, 2:3]


def run_sfm(tracks, K, dist=None):
    tracks = np.asarray(tracks, float)
    N, V, _ = tracks.shape
    if dist is not None:
        tracks = np.stack([cv2.undistortPoints(tracks[:, v].reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
                           for v in range(V)], 1)
    Kinv = np.linalg.inv(K)
    m0 = (Kinv @ np.c_[tracks[:, 0], np.ones(N)].T).T
    cands = []
    for v in range(1, V):
        H, _ = cv2.findHomography(tracks[:, 0], tracks[:, v], cv2.RANSAC, 3.0)
        cands.append(decompose(H, K, m0))
    best, best_cost = None, np.inf
    for combo in itertools.product(*cands):
        ns = np.array([c[2] for c in combo])
        cost = np.linalg.norm(ns - ns.mean(0), axis=1).sum()
        if cost < best_cost:
            best, best_cost = combo, cost
    n = np.mean([c[2] for c in best], 0)
    n /= np.linalg.norm(n)
    X0 = m0 / (m0 @ n)[:, None]  # d = 1
    poses = [(np.eye(3), np.zeros(3))] + [(R, t) for R, t, _ in best]

    def pack(poses, X):
        p = [np.r_[cv2.Rodrigues(R)[0].ravel(), t] for R, t in poses[1:]]
        return np.r_[np.concatenate(p), X.ravel()]

    def unpack(x):
        ps = [(np.eye(3), np.zeros(3))]
        for k in range(V - 1):
            r = x[6 * k:6 * k + 6]
            ps.append((cv2.Rodrigues(r[:3])[0], r[3:]))
        return ps, x[6 * (V - 1):].reshape(N, 3)

    def resid(x):
        ps, X = unpack(x)
        return np.concatenate([(project(K, R, t, X) - tracks[:, v]).ravel() for v, (R, t) in enumerate(ps)])

    x0 = pack(poses, X0)
    init_err = float(np.sqrt(np.mean(resid(x0) ** 2)))
    sol = least_squares(resid, x0, loss="huber", f_scale=2.0, max_nfev=200)
    poses, X = unpack(sol.x)
    # rescale so the plane is at distance 1 from camera 1 (BA can drift the gauge slightly)
    c = X.mean(0)
    _, _, Vt = np.linalg.svd(X - c)
    nrm = Vt[2] if Vt[2] @ c > 0 else -Vt[2]
    d = float(nrm @ c)
    X, poses = X / d, [(R, t / d) for R, t in poses]
    planarity = float(np.abs((X - X.mean(0)) @ nrm).std() / np.ptp(X @ Vt[0]))
    uv = (X - X.mean(0)) @ Vt[:2].T
    hull = cv2.convexHull(np.float32(uv)).reshape(-1, 2)
    centers = [-R.T @ t for R, t in poses]
    r = resid(sol.x) / 1.0
    return {"points3d": X, "poses": poses, "camera_centers": centers, "normal": nrm,
            "reproj_err_init_px": init_err, "reproj_err_px": float(np.sqrt(np.mean(r ** 2))),
            "planarity": planarity, "plane_axes": Vt[:2], "plane_origin": X.mean(0), "boundary_uv": hull}


def corners_on_plane(result, K, uv_pixels):
    """Tapped corners in view 1 -> 3-D points on the reconstructed plane (ray-plane intersection)."""
    n, c = result["normal"], result["plane_origin"]
    m = (np.linalg.inv(K) @ np.c_[np.asarray(uv_pixels, float), np.ones(len(uv_pixels))].T).T
    return m * ((n @ c) / (m @ n))[:, None]


def scale_from_length(P, Q, true_length):
    return true_length / float(np.linalg.norm(np.asarray(P) - np.asarray(Q)))
