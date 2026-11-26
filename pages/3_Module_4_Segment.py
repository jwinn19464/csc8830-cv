import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import cv2
import numpy as np
import streamlit as st

from cvlib.io import decode_upload, gray, limit_size, png_bytes
from cvlib.ui import page, photo, show, tap_points, theory
from module4.fourier import run_all
from module4.metrics import compare
from module4.segment import overlay, segment_rgb, segment_thermal

page("Module 4: Find the Human", "Classical segmentation only (no deep learning or machine learning). SAM2 runs separately.")
t_rgb, t_th, t_sam, t_fft, t_theory = st.tabs(["RGB", "Thermal", "vs SAM2", "Fourier", "Theory"])
ss = st.session_state


def steps(d):
    with st.expander("Pipeline steps"):
        for k, v in d.items():
            show(v, k)


with t_rgb:
    img = photo("Photo of a person", "m4rgb")
    if img is not None:
        img = limit_size(img, 900)
        use_box = st.toggle("Draw a box around the person (tap two opposite corners)")
        box = None
        if use_box:
            p = tap_points(img, "m4box", n=2)
            if len(p) == 2:
                (x0, y0), (x1, y1) = p
                box = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
        lo = st.slider("Low threshold", 1.0, 6.0, 2.5, 0.1)
        hi = st.slider("High threshold", 2.0, 10.0, 5.0, 0.1)
        with st.spinner("Segmenting..."):
            mask, d = segment_rgb(img, box, lo, hi)
        ss["m4_img"], ss["m4_mask"], ss["m4_box"] = img, mask, box
        show(overlay(img, mask), "boundary")
        steps(d)
        st.download_button("Download mask.png", png_bytes(mask), "rgb_mask.png")

with t_th:
    up = st.file_uploader("Thermal image", type=["jpg", "jpeg", "png"], key="m4th")
    if up is not None:
        th = limit_size(decode_upload(up), 900)
        hot = st.toggle("Hot is bright", value=True)
        mask, d = segment_thermal(th, hot)
        ss["m4_th_img"], ss["m4_th_mask"] = th, mask
        show(overlay(th, mask), "boundary")
        steps(d)
        st.download_button("Download mask.png", png_bytes(mask), "thermal_mask.png")
    st.caption("Public thermal people images: FLIR ADAS dataset, or the OSU thermal pedestrian database.")

with t_sam:
    which = st.radio("Compare", ["RGB result", "Thermal result"], horizontal=True)
    img, mask = (ss.get("m4_img"), ss.get("m4_mask")) if which == "RGB result" else (ss.get("m4_th_img"), ss.get("m4_th_mask"))
    if mask is None:
        st.info("Run the segmentation tab first.")
    else:
        box = ss.get("m4_box")
        if box is not None and which == "RGB result":
            x, y, w, h = [int(v) for v in box]
            st.code(f"box = np.array([{x}, {y}, {x + w}, {y + h}])  # give SAM2 the same box", language="python")
        st.markdown("Run SAM2 in Colab with the snippet in `module4/metrics.py`, then upload its mask "
                    "(on the same image, at the size shown).")
        st.caption(f"Image size here: {mask.shape[1]}×{mask.shape[0]}")
        up = st.file_uploader("SAM2 mask (PNG)", type=["png"], key="m4sam")
        if up is not None:
            ref = gray(decode_upload(up, None))
            r = compare(mask, ref)
            c1, c2, c3 = st.columns(3)
            c1.metric("IoU", f"{r['IoU']:.3f}")
            c2.metric("Dice", f"{r['Dice']:.3f}")
            c3.metric("Boundary F", f"{r['boundary F']:.3f}")
            st.json({k: round(v, 3) for k, v in r.items()})
            ref_r = cv2.resize(ref, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            show(overlay(img, mask, other=np.uint8(ref_r > 127) * 255), "green = classical, magenta = SAM2")

with t_fft:
    src = ss.get("m4_img")
    if src is None:
        st.info("Add a photo on the RGB tab first.")
    else:
        sigma = st.slider("Gaussian sigma (px)", 0.5, 6.0, 2.0, 0.5, key="m4sig")
        for k, v in run_all(gray(limit_size(src, 600)), sigma).items():
            show(v, k)

with t_theory:
    theory("module4")
