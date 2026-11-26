import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import time

import cv2
import numpy as np
import streamlit as st

from cvlib.io import limit_size
from cvlib.ui import page, photo, show, theory
from module3.blur import apply_per_channel, compare, conv2d_fourier, conv2d_spatial, log_spectrum, make_kernel, to_u8

page("Module 3: Blur", "Spatial convolution and Fourier-domain multiplication, both from scratch, should give the same image.")
t_run, t_th = st.tabs(["Blur two ways", "Theory"])

with t_run:
    img = photo("Photo", "m3img")
    if img is None:
        yy, xx = np.mgrid[0:240, 0:320]
        img = np.uint8(np.dstack([(xx // 20 + yy // 20) % 2 * 200 + 30] * 3))
        st.caption("Using a test checkerboard until you add a photo.")
    img = limit_size(img, 480)
    kind = st.segmented_control("Kernel", ["gaussian", "box", "motion"], default="gaussian")
    size = st.slider("Kernel size (px)", 3, 51, 15, step=2)
    sigma = st.slider("Gaussian sigma", 0.5, 20.0, 2.5) if kind == "gaussian" else 1.0
    angle = st.slider("Motion angle (deg)", 0, 180, 0) if kind == "motion" else 0
    if kind == "gaussian" and size < 6 * sigma - 1:
        st.warning(f"size {size} < 6σ ≈ {6 * sigma:.0f}: the Gaussian is cut off and acts like a box filter "
                   "(look for ringing in its spectrum).")
    h = make_kernel(kind, size, sigma, angle)
    t0 = time.perf_counter(); s = apply_per_channel(conv2d_spatial, img, h); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); f = apply_per_channel(conv2d_fourier, img, h); tf = time.perf_counter() - t0
    r = compare(s, f)
    c1, c2 = st.columns(2)
    with c1:
        show(to_u8(s), f"Spatial convolution ({ts * 1000:.0f} ms)")
    with c2:
        show(to_u8(f), f"Fourier domain ({tf * 1000:.0f} ms)")
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{r['MSE']:.1e}")
    c2.metric("max |diff|", f"{r['max |diff|']:.1e}")
    c3.metric("PSNR", "∞" if r["PSNR (dB)"] == float("inf") else f"{r['PSNR (dB)']:.0f} dB")
    st.caption("Both use circular (wrap-around) boundaries, so they agree to floating-point precision.")
    with st.expander("Spectra (log magnitude)"):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c1, c2, c3 = st.columns(3)
        c1.image(log_spectrum(g), caption="image")
        kp = np.zeros((128, 128)); kp[:h.shape[0], :h.shape[1]] = h
        c2.image(log_spectrum(kp), caption="kernel")
        c3.image(log_spectrum(cv2.cvtColor(to_u8(f), cv2.COLOR_BGR2GRAY)), caption="blurred")
        st.image(np.uint8(255 * h / h.max()), caption="kernel", width=160)

with t_th:
    theory("module3")
