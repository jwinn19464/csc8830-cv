import cv2
import numpy as np

from module2.measure import dimension
from module3.blur import apply_per_channel, compare, conv2d_fourier, conv2d_spatial, gaussian_kernel
from module4.metrics import compare as mcompare
from module4.segment import segment_rgb, segment_thermal
from module6.lk import bilinear, track
from module6.sfm import project, run_sfm

K = np.array([[1000., 0, 640], [0, 1000, 360], [0, 0, 1]])


def test_measure_exact():
    Z = 2500.0
    P1, P2 = np.array([-300, 100, Z]), np.array([200, -150, Z])
    p1 = (K @ P1)[:2] / Z
    p2 = (K @ P2)[:2] / Z
    assert abs(dimension(p1, p2, Z, K) - np.linalg.norm(P1 - P2)) < 1e-6


def test_blur_match():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 80, 3)).astype(float)
    h = gaussian_kernel(9, 2.0)
    s, f = apply_per_channel(conv2d_spatial, img, h), apply_per_channel(conv2d_fourier, img, h)
    assert compare(s, f)["MSE"] < 1e-20
    a = np.zeros((40, 40)); a[20, 20] = 1
    assert np.allclose(conv2d_spatial(a, h)[16:25, 16:25], h)  # impulse response = kernel


def _person_scene():
    rng = np.random.default_rng(1)
    img = np.full((240, 200, 3), (180, 200, 210), np.uint8) + rng.integers(0, 8, (240, 200, 3)).astype(np.uint8)
    m = np.zeros((240, 200), np.uint8)
    cv2.ellipse(m, (100, 70), (22, 28), 0, 0, 360, 255, -1)
    cv2.rectangle(m, (70, 98), (130, 240), 255, -1)
    img[m > 0] = (60, 40, 150)
    return img, m


def test_segment_rgb():
    img, gt = _person_scene()
    mask, _ = segment_rgb(img)
    assert mcompare(mask, gt)["IoU"] > 0.9


def test_segment_thermal():
    _, gt = _person_scene()
    th = np.full(gt.shape, 70, np.uint8); th[gt > 0] = 200
    th = cv2.GaussianBlur(th, (5, 5), 0)
    mask, _ = segment_thermal(th)
    assert mcompare(mask, gt)["IoU"] > 0.9


def test_bilinear_and_lk():
    yy, xx = np.mgrid[0:120, 0:160].astype(float)
    I0 = 128 + 60 * np.sin(xx / 7) * np.cos(yy / 9)
    assert abs(bilinear(I0, np.array([10.5]), np.array([20.25]))[0]
               - (128 + 60 * np.sin(10.5 / 7) * np.cos(20.25 / 9))) < 0.5
    I1 = 128 + 60 * np.sin((xx - 2.3) / 7) * np.cos((yy - 1.7) / 9)
    out, ok = track(I0, I1, np.array([[60., 50], [90, 70]]))
    assert np.allclose(out - [[60, 50], [90, 70]], [2.3, 1.7], atol=0.05)


def test_sfm_planar():
    rng = np.random.default_rng(2)
    X = np.c_[rng.uniform(-0.4, 0.4, (60, 2)), np.zeros(60)]  # plane z=0 in world
    poses = []
    for ang, t in [(0, [0, 0, 2.0]), (12, [0.3, 0, 2.1]), (-10, [-0.25, 0.1, 1.9]), (8, [0.1, -0.3, 2.2])]:
        R = cv2.Rodrigues(np.deg2rad([ang * 0.6, ang, 3]))[0]
        poses.append((R, np.array(t)))
    tracks = np.stack([project(K, R, t, X) for R, t in poses], 1)
    r = run_sfm(tracks, K)
    assert r["reproj_err_px"] < 0.05
    # distances between reconstructed points proportional to true ones
    i, j, k = 0, 1, 2
    P = r["points3d"]
    ratio = np.linalg.norm(P[i] - P[j]) / np.linalg.norm(P[i] - P[k])
    true = np.linalg.norm(X[i] - X[j]) / np.linalg.norm(X[i] - X[k])
    assert abs(ratio - true) < 1e-3


def test_stereo_depth():
    from module7.stereo import measure_shape, points_3d, rectify_pair
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (360, 480)).astype(np.uint8), (3, 3), 0)
    L = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    f, B, Z = 500.0, 10.0, 100.0          # disparity = fB/Z = 50 px
    R = np.roll(L, -50, axis=1)
    Kc = np.array([[f, 0, 240], [0, f, 180], [0, 0, 1]])
    rig = {"K1": Kc, "d1": np.zeros(5), "K2": Kc, "d2": np.zeros(5), "R": np.eye(3), "T": np.array([-B, 0, 0]), "size": (480, 360)}
    rL, rR, geo = rectify_pair(rig, L, R)
    pts = [(150, 100), (350, 100), (350, 250), (150, 250)]
    P, s, _ = points_3d(rL, rR, pts, geo)
    assert np.allclose(P[:, 2], Z, rtol=0.02)
    m = measure_shape(P, "rectangle")
    assert abs(m["width (avg of opposite sides)"] - 200 * Z / f) < 1.0


def test_room_planes():
    from module7.room import find_planes, opposite_pairs
    rng = np.random.default_rng(4)
    a = np.c_[np.full(800, -150.0), rng.uniform(-100, 100, (800, 2)) + [0, 300]]   # wall x=-150
    b = np.c_[np.full(800, 250.0), rng.uniform(-100, 100, (800, 2)) + [0, 300]]    # wall x=+250
    pr = opposite_pairs(find_planes(np.r_[a, b] + rng.normal(0, 0.5, (1600, 3)), thr=2.0))
    assert any(abs(p["gap"] - 400) < 3 for p in pr)


def test_dates_and_plan():
    import datetime as dt
    from fridge.dates import parse_dates
    from fridge.plan import rank
    T = dt.date(2026, 11, 26)
    assert parse_dates("BEST BY 12/04", T)[0] == dt.date(2026, 12, 4)
    assert parse_dates("EXP 04DEC26", T)[0] == dt.date(2026, 12, 4)
    assert parse_dates("USE BY 2027-01-15", T)[0] == dt.date(2027, 1, 15)
    inv = [{"name": "eggs", "expiry": T + dt.timedelta(1)}, {"name": "milk", "expiry": T - dt.timedelta(2)},
           {"name": "tomatoes", "expiry": T + dt.timedelta(2)}]
    r = rank(inv, 3, T)
    assert r[0]["recipe"] == "Tomato egg stir-fry"
    assert all("milk" not in " ".join(x["uses"]) for x in r)


def test_fridge_detect_and_dedupe():
    from fridge.items import dedupe, detect_items
    rng = np.random.default_rng(5)
    shelf = np.full((300, 900, 3), (205, 210, 215), np.uint8)
    shelf = (shelf + rng.integers(0, 6, shelf.shape)).astype(np.uint8)
    for x, col in [(60, (30, 30, 200)), (260, (40, 160, 40)), (470, (200, 120, 20)), (680, (20, 200, 220))]:
        cv2.rectangle(shelf, (x, 90), (x + 120, 260), col, -1)
        cv2.putText(shelf, "EXP", (x + 10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    v1, v2 = shelf[:, :600].copy(), shelf[:, 300:].copy()   # two overlapping keyframes
    b1, b2 = detect_items(v1)[0], detect_items(v2)[0]
    items = dedupe([v1, v2], [b1, b2])
    assert len(items) == 4, (b1, b2, len(items))


def test_keyframe_sharpness():
    from fridge.keyframes import fft_sharpness
    rng = np.random.default_rng(6)
    g = rng.integers(0, 255, (200, 200)).astype(np.uint8)
    assert fft_sharpness(g) > fft_sharpness(cv2.GaussianBlur(g, (15, 15), 4))


def test_calibration_synthetic():
    from module2.calibrate import calibrate_images
    sq = 50
    board = np.full((7 * sq + 100, 10 * sq + 100), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                board[50 + r * sq:50 + (r + 1) * sq, 50 + c * sq:50 + (c + 1) * sq] = 0
    Kt = np.array([[900., 0, 640], [0, 900, 480], [0, 0, 1]])
    h, w = board.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    rng = np.random.default_rng(7)
    imgs = []
    for _ in range(8):
        R = cv2.Rodrigues(np.deg2rad(rng.uniform(-25, 25, 3) * [1, 1, 0.3]))[0]
        t = np.array([rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(1300, 1700)])
        P = np.c_[src - [w / 2, h / 2], np.zeros(4)]
        x = (Kt @ (R @ P.T + t[:, None])).T
        H = cv2.getPerspectiveTransform(src, np.float32(x[:, :2] / x[:, 2:]))
        imgs.append(cv2.cvtColor(cv2.warpPerspective(board, H, (1280, 960), borderValue=150), cv2.COLOR_GRAY2BGR))
    c = calibrate_images(imgs, (9, 6), 25.0)
    assert c["rms"] < 0.5 and max(c["per_image_err"]) < 0.5
    assert abs(c["K"][0, 0] - 900) < 20
