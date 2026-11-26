"""Module 4 - compare a classical mask with a SAM2 mask.

SAM2 runs separately (GPU Colab). Give it the same box you gave the classical method,
then export its mask as a PNG:

    # Colab:  !pip install -q sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import numpy as np, cv2
    p = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    img = cv2.imread("person.jpg")[:, :, ::-1]
    p.set_image(img)
    masks, scores, _ = p.predict(box=np.array([x0, y0, x1, y1]), multimask_output=False)
    cv2.imwrite("sam2_mask.png", (masks[0] > 0).astype("uint8") * 255)

    python -m module4.metrics --ours mask.png --sam2 sam2_mask.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def _b(m):
    return np.asarray(m) > 0


def iou(a, b):
    a, b = _b(a), _b(b)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def dice(a, b):
    a, b = _b(a), _b(b)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


def edge_map(m):
    m = _b(m).astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_scores(a, b, tol=None):
    """Boundary precision/recall/F within tol pixels, mean and Hausdorff boundary distance."""
    ea, eb = edge_map(a), edge_map(b)
    if not ea.any() or not eb.any():
        return {"boundary F": 0.0, "mean boundary dist (px)": float("inf"), "Hausdorff (px)": float("inf")}
    tol = tol or max(2.0, 0.0075 * np.hypot(*a.shape))
    da, db = distance_transform_edt(~ea), distance_transform_edt(~eb)
    d_ab, d_ba = db[ea], da[eb]  # each boundary pixel of A -> nearest of B, and back
    p, r = float((d_ab <= tol).mean()), float((d_ba <= tol).mean())
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"boundary precision": p, "boundary recall": r, "boundary F": f, "tolerance (px)": float(tol),
            "mean boundary dist (px)": float(np.concatenate([d_ab, d_ba]).mean()),
            "Hausdorff (px)": float(max(d_ab.max(), d_ba.max()))}


def compare(ours, ref):
    if ours.shape != ref.shape:
        ref = cv2.resize(ref, (ours.shape[1], ours.shape[0]), interpolation=cv2.INTER_NEAREST)
    return {"IoU": iou(ours, ref), "Dice": dice(ours, ref), **boundary_scores(ours, ref)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--sam2", required=True)
    a = ap.parse_args()
    r = compare(cv2.imread(a.ours, 0), cv2.imread(a.sam2, 0))
    for k, v in r.items():
        print(f"{k:>26}: {v:.4f}")


if __name__ == "__main__":
    main()
