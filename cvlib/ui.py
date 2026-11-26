"""Streamlit widgets that work on a phone: photo pickers and tap-to-place points."""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from .io import decode_upload, rgb

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAVE_TAP = True
except Exception:  # pragma: no cover
    HAVE_TAP = False


def page(title: str, caption: str = ""):
    st.set_page_config(page_title=title, page_icon="🔭", layout="centered")
    st.title(title)
    if caption:
        st.caption(caption)


def photo(label: str, key: str, camera: bool = True):
    """File upload (on a phone this offers Camera / Photo Library) plus an optional in-page camera."""
    up = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=key + "_file")
    if up is None and camera and st.toggle("Use the in-page camera instead", key=key + "_camtoggle"):
        up = st.camera_input(label, key=key + "_cam", label_visibility="collapsed")
    return decode_upload(up) if up is not None else None


def tap_points(img_bgr: np.ndarray, key: str, n: int | None = None, width: int = 360) -> list[tuple[float, float]]:
    """Tap on the image to add points (full-resolution pixel coordinates). Works with touch."""
    pts_key = key + "_pts"
    st.session_state.setdefault(pts_key, [])
    pts = st.session_state[pts_key]
    h, w = img_bgr.shape[:2]
    s = width / w
    view = img_bgr.copy()
    r = max(3, int(round(4 / s)))
    for i, (u, v) in enumerate(pts):
        cv2.circle(view, (int(u), int(v)), r * 2, (0, 255, 255), -1)
        cv2.putText(view, str(i + 1), (int(u) + 2 * r, int(v)), cv2.FONT_HERSHEY_SIMPLEX, r / 3, (0, 255, 255), max(1, r // 2))
    if len(pts) >= 2:
        cv2.polylines(view, [np.int32(pts)], len(pts) > 2, (0, 255, 255), max(1, r // 2))
    small = cv2.resize(view, (width, int(h * s)))
    if HAVE_TAP:
        c = streamlit_image_coordinates(rgb(small), key=key + "_tap")
        if c and c != st.session_state.get(key + "_last"):
            st.session_state[key + "_last"] = c
            if n is None or len(pts) < n:
                pts.append((c["x"] / s, c["y"] / s))
                st.rerun()
    else:
        st.image(rgb(small))
    c1, c2 = st.columns(2)
    if c1.button("Undo last point", key=key + "_undo", disabled=not pts):
        pts.pop()
        st.rerun()
    if c2.button("Clear points", key=key + "_clear", disabled=not pts):
        pts.clear()
        st.rerun()
    with st.expander("Type coordinates instead"):
        txt = st.text_input("u,v pairs separated by spaces", key=key + "_txt", placeholder="410,300 820,310")
        if st.button("Use these", key=key + "_use") and txt.strip():
            pts[:] = [tuple(map(float, p.split(","))) for p in txt.split()]
            st.rerun()
    return list(pts)


def show(img, caption=None):
    st.image(rgb(img) if img.ndim == 3 else img, caption=caption, width="stretch")


def calibration_widget(key: str):
    """Use the calibration made on the Module 2 page this session, or upload calibration.npz."""
    from module2.calibrate import load_calibration
    c = st.session_state.get("calib")
    up = st.file_uploader("Calibration file (calibration.npz from Module 2)", type=["npz"], key=key + "_npz")
    if up is not None:
        c = load_calibration(up.getvalue())
        st.session_state["calib"] = c
    if c is None:
        st.info("No calibration yet. Make one on the Module 2 page (it carries over), or upload calibration.npz.")
    else:
        st.caption(f"Calibration: fx = {c['K'][0, 0]:.0f} px at {c['size'][0]}×{c['size'][1]}, RMS {c['rms']:.2f} px")
    return c


def theory(name: str):
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "docs" / f"{name}.md"
    st.markdown(p.read_text() if p.exists() else "_Theory notes not found._")
