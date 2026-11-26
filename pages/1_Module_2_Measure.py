import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import numpy as np
import pandas as pd
import streamlit as st

from cvlib.io import decode_upload
from cvlib.ui import calibration_widget, page, photo, show, tap_points, theory
from module2.calibrate import calibrate_images, to_json, to_npz_bytes
from module2.measure import K_for_image, polyline_lengths, undistort, undistort_points
from module2.validate import add_errors, stats

page("Module 2: Measure", "Calibrate your phone camera, then measure real-world sizes from a single photo.")
t_cal, t_meas, t_val, t_th = st.tabs(["Calibrate", "Measure", "Validate", "Theory"])

with t_cal:
    st.markdown("Print a checkerboard, tape it flat, and take **15–25 photos** from different angles with the "
                "1× camera. Use the same camera and orientation you'll use for measuring.")
    c1, c2, c3 = st.columns(3)
    cols = c1.number_input("Inner corners across", 3, 20, 9)
    rows = c2.number_input("Inner corners down", 3, 20, 6)
    sq = c3.number_input("Square size (mm)", 1.0, 200.0, 25.0)
    files = st.file_uploader("Checkerboard photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if files and st.button("Calibrate", type="primary"):
        imgs = [decode_upload(f) for f in files]
        try:
            with st.spinner("Finding corners and calibrating..."):
                c = calibrate_images(imgs, (int(cols), int(rows)), sq)
            st.session_state["calib"] = {k: c[k] for k in ("K", "dist", "size", "rms")}
            st.session_state["calib_full"] = c
        except ValueError as e:
            st.error(str(e))
    c = st.session_state.get("calib_full")
    if c:
        st.success(f"Board found in {len(c['used'])} photos. RMS reprojection error **{c['rms']:.3f} px** "
                   f"({'good' if c['rms'] < 0.5 else 'usable' if c['rms'] < 1 else 'retake blurry photos'}).")
        st.write("Camera matrix K (pixels, at the resized working resolution):")
        st.dataframe(pd.DataFrame(np.round(c["K"], 2)), hide_index=True)
        st.write("Distortion k1 k2 p1 p2 k3:", np.round(c["dist"], 4).tolist())
        st.bar_chart(pd.DataFrame({"error (px)": c["per_image_err"]}, index=[f"#{i + 1}" for i in c["used"]]))
        st.download_button("Download calibration.npz", to_npz_bytes(c), "calibration.npz")
        st.download_button("Download calibration.json", to_json(c), "calibration.json")
        with st.expander("Detected corners"):
            for ov in c["overlays"][:6]:
                show(ov)

with t_meas:
    calib = calibration_widget("m2meas")
    st.markdown("Photograph a flat object **facing the camera squarely**, measure the distance from the lens "
                "to the object with a tape, then tap the object's corners or ends.")
    img = photo("Object photo", "m2obj")
    if img is not None and calib is not None:
        K = K_for_image(calib, img.shape)
        und = undistort(img, K, calib["dist"])
        Z = st.number_input("Distance from lens to object (cm)", 10.0, 2000.0, 250.0)
        closed = st.toggle("Closed shape (join the last point to the first)")
        pts = tap_points(und, "m2pts")
        if len(pts) >= 2:
            L = polyline_lengths(pts, Z, K, closed)
            st.subheader("Result")
            for i, d in enumerate(L):
                st.metric(f"Segment {i + 1}", f"{d:.1f} cm")
            st.caption("Z is the distance to the lens, not the phone's back; a 1 cm error in Z changes sizes by 1/Z.")

with t_val:
    st.markdown("Log **20 measurements** (for example 5 objects × 4 edges) next to the tape-measured truth.")
    st.session_state.setdefault("m2_df", pd.DataFrame({"label": [f"object {i // 4 + 1} edge {i % 4 + 1}" for i in range(20)],
                                                       "true": [np.nan] * 20, "measured": [np.nan] * 20}))
    df = st.data_editor(st.session_state["m2_df"], num_rows="dynamic", width="stretch", key="m2_editor")
    good = df.dropna(subset=["true", "measured"])
    good = good[good["true"] > 0]
    if len(good) >= 2:
        e = add_errors(good)
        s = stats(e)
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean % error", f"{s['mean % error']:.2f}%")
        c2.metric("MAE", f"{s['MAE']:.2f}")
        c3.metric("RMSE", f"{s['RMSE']:.2f}")
        st.json({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
        st.scatter_chart(e, x="true", y="measured")
        st.download_button("Download errors.csv", e.to_csv(index=False), "errors.csv")
    st.caption("Saved only while this page is open; download the CSV to keep it.")

with t_th:
    theory("module2")
