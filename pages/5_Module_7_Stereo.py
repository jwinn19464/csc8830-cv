import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cvlib.io import decode_upload, read_video
from cvlib.ui import calibration_widget, page, show, tap_points, theory
from module2.measure import K_for_image
from module7.room import room_from_pair, sync_offset
from module7.stereo import (baseline, load_rig, measure_shape, points_3d, rectify_pair, rig_from_motion,
                            save_rig, stereo_calibrate)

page("Module 7: Stereo", "Estimate depth Z from two calibrated views first, then real-world sizes.")
t_rig, t_obj, t_room, t_th = st.tabs(["1 · Stereo rig", "2 · Measure objects", "3 · Room size", "Theory"])
ss = st.session_state


def draw_rows(rL, rR, n=12):
    v = np.hstack([rL, rR])
    for y in np.linspace(0, v.shape[0] - 1, n).astype(int):
        cv2.line(v, (0, y), (v.shape[1], y), (0, 255, 0), 1)
    return v


with t_rig:
    mode = st.radio("How are the two views made?",
                    ["Two phones + checkerboard (stereo calibration)",
                     "One phone moved sideways by a measured distance (uses Module 2 calibration)",
                     "Upload stereo_rig.npz"])
    if mode.startswith("Two"):
        st.markdown("Fix both phones side by side (tape them to a ruler), then take **10+ simultaneous pairs** of "
                    "the checkerboard. Upload all left photos, then all right photos, in the same order.")
        c1, c2, c3 = st.columns(3)
        pc = c1.number_input("Inner corners across", 3, 20, 9, key="m7c")
        pr = c2.number_input("Inner corners down", 3, 20, 6, key="m7r")
        sq = c3.number_input("Square (cm)", 0.1, 20.0, 2.5, key="m7s")
        L = st.file_uploader("Left photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="m7L")
        R = st.file_uploader("Right photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="m7R")
        if L and R and len(L) == len(R) and st.button("Stereo calibrate", type="primary"):
            try:
                with st.spinner("Calibrating..."):
                    ss["rig"] = stereo_calibrate([decode_upload(f) for f in L], [decode_upload(f) for f in R],
                                                 (int(pc), int(pr)), sq)
            except ValueError as e:
                st.error(str(e))
    elif mode.startswith("One"):
        calib = calibration_widget("m7k")
        st.markdown("Take the left photo, slide the phone **straight sideways** (along a ruler or table edge) by a "
                    "measured distance, keep it pointing the same way, and take the right photo.")
        B = st.number_input("Baseline: how far you moved the phone (cm)", 1.0, 200.0, 20.0)
        a = st.file_uploader("Left photo", type=["jpg", "jpeg", "png"], key="m7a")
        b = st.file_uploader("Right photo", type=["jpg", "jpeg", "png"], key="m7b")
        if calib is not None and a and b and st.button("Build rig", type="primary"):
            iL, iR = decode_upload(a), decode_upload(b)
            K = K_for_image(calib, iL.shape)
            try:
                ss["rig"] = rig_from_motion(iL, iR, K, calib["dist"], B)
                ss["m7_pair"] = (iL, iR)
            except ValueError as e:
                st.error(str(e))
    else:
        up = st.file_uploader("stereo_rig.npz", type=["npz"], key="m7npz")
        if up is not None:
            ss["rig"] = load_rig(up.getvalue())
    rig = ss.get("rig")
    if rig is not None:
        st.success(f"Rig ready: baseline **{baseline(rig):.2f} cm**, f = {rig['K1'][0, 0]:.0f} px"
                   + (f", stereo RMS {rig['rms']:.2f} px" if np.isfinite(rig["rms"]) else ""))
        st.download_button("Download stereo_rig.npz", save_rig(rig), "stereo_rig.npz")

with t_obj:
    rig = ss.get("rig")
    if rig is None:
        st.info("Set up the stereo rig in tab 1 first.")
    else:
        st.markdown("A left/right pair of the object (or reuse the pair from tab 1). Tap points **in the left image**; "
                    "each is found on the same row of the right image, then Z = fB/d.")
        a = st.file_uploader("Left photo", type=["jpg", "jpeg", "png"], key="m7oa")
        b = st.file_uploader("Right photo", type=["jpg", "jpeg", "png"], key="m7ob")
        pair = (decode_upload(a), decode_upload(b)) if a and b else ss.get("m7_pair")
        if pair:
            iL, iR = pair
            sx = rig["size"][0] / iL.shape[1]
            if abs(sx - 1) > 0.01:
                iL = cv2.resize(iL, rig["size"]); iR = cv2.resize(iR, rig["size"])
            rL, rR, geo = rectify_pair(rig, iL, iR)
            with st.expander("Rectified pair (rows should line up)"):
                show(draw_rows(rL, rR))
            shape = st.segmented_control("Shape", ["rectangle", "circle", "polygon"], default="rectangle")
            st.caption({"rectangle": "Tap the 4 corners in order.", "circle": "Tap the centre, then 1–4 points on the rim.",
                        "polygon": "Tap the vertices in order."}[shape])
            pts = tap_points(rL, "m7pts", n=4 if shape == "rectangle" else None)
            need = 4 if shape == "rectangle" else 2 if shape == "circle" else 3
            if len(pts) >= need:
                P, scores, ptsR = points_3d(rL, rR, pts, geo)
                if np.isnan(P).any():
                    st.error("Some points couldn't be matched in the right image. Tap on textured spots near the edge.")
                else:
                    st.dataframe(pd.DataFrame({"u": [p[0] for p in pts], "disparity": [p[0] - q[0] for p, q in zip(pts, ptsR)],
                                               "match score": scores, "X": P[:, 0], "Y": P[:, 1], "Z": P[:, 2]}).round(2),
                                 hide_index=True)
                    for k, v in measure_shape(P, shape).items():
                        st.metric(k, f"{v:.2f} {'cm²' if k == 'area' else 'cm'}")

with t_room:
    rig = ss.get("rig")
    if rig is None:
        st.info("Set up the stereo rig in tab 1 first.")
    else:
        st.markdown("Record the room with both phones at once, from the middle of the room. Flick the light (or use a "
                    "flash) at the start so the videos can be synced. A single stereo photo pair works too.")
        src = st.radio("Input", ["Stereo videos", "One stereo photo pair"], horizontal=True)
        pair = None
        if src == "Stereo videos":
            a = st.file_uploader("Left video", type=["mp4", "mov", "m4v"], key="m7va")
            b = st.file_uploader("Right video", type=["mp4", "mov", "m4v"], key="m7vb")
            if a and b:
                fL, _ = read_video(a, 20, max(rig["size"]))
                fR, _ = read_video(b, 20, max(rig["size"]))
                off = sync_offset(fL, fR)
                st.caption(f"Sync: right frame = left frame {off:+d}")
                i = st.slider("Left frame to use", 0, max(0, len(fL) - 1 - max(0, off)), min(len(fL) - 1, 30))
                j = int(np.clip(i + off, 0, len(fR) - 1))
                pair = fL[i], fR[j]
        else:
            a = st.file_uploader("Left photo", type=["jpg", "jpeg", "png"], key="m7ra")
            b = st.file_uploader("Right photo", type=["jpg", "jpeg", "png"], key="m7rb")
            if a and b:
                pair = decode_upload(a), decode_upload(b)
        if pair:
            iL, iR = [cv2.resize(x, rig["size"]) for x in pair]
            nd = st.select_slider("Max disparity", [64, 96, 128, 192, 256], value=128)
            with st.spinner("Disparity, 3-D points and planes..."):
                r = room_from_pair(rig, iL, iR, nd)
            d = r["disparity"]
            show(cv2.applyColorMap(np.uint8(255 * np.clip(d, 0, None) / (d.max() + 1e-6)), cv2.COLORMAP_TURBO), "disparity")
            if r["pairs"]:
                st.subheader("Distances between opposite parallel planes")
                for k, p in enumerate(sorted(r["pairs"], key=lambda p: -p["gap"])):
                    ax = ["left-right", "up-down", "front-back"][int(np.argmax(np.abs(p["normal"])))]
                    st.metric(f"{ax} (planes {p['planes'][0] + 1} & {p['planes'][1] + 1})", f"{p['gap']:.0f} cm")
            else:
                st.warning(f"Found {len(r['planes'])} plane(s) but no opposite pair. Use the corner method below.")
            st.markdown("**Corner method:** tap two points on opposite walls (or floor and ceiling).")
            pts = tap_points(r["rectL"], "m7room", n=2)
            if len(pts) == 2:
                P, _, _ = points_3d(r["rectL"], r["rectR"], pts, r["geo"])
                if not np.isnan(P).any():
                    st.metric("Distance", f"{np.linalg.norm(P[0] - P[1]):.0f} cm")

with t_th:
    theory("module7")
