import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cvlib.io import decode_upload, read_video, write_video
from cvlib.ui import calibration_widget, page, show, tap_points, theory
from module2.measure import K_for_image, undistort_points
from module6.flow import analyse
from module6.lk import good_features, validate
from module6.sfm import corners_on_plane, match_tracks, run_sfm, scale_from_length

page("Module 6: Motion", "Optical flow with inferences, Lucas-Kanade from scratch, and structure from motion.")
t_flow, t_lk, t_sfm, t_th = st.tabs(["Optical flow", "Lucas-Kanade", "Structure from motion", "Theory"])
ss = st.session_state

with t_flow:
    st.markdown("Upload a **30 s clip** (do two: one with the camera moving, one with things moving in front "
                "of a still camera).")
    up = st.file_uploader("Video", type=["mp4", "mov", "m4v", "avi"], key="m6vid")
    secs = st.slider("Seconds to analyse", 2, 30, 30)
    step = st.select_slider("Use every Nth frame (faster on a phone)", [1, 2, 3, 5], value=3)
    if up is not None and st.button("Run optical flow", type="primary"):
        with st.spinner("Reading video..."):
            frames, fps = read_video(up, secs, 480, step)
        bar = st.progress(0.0, "Computing flow")
        rows, vis = [], []
        chunk = 20
        for i in range(0, len(frames) - 1, chunk):
            r, v = analyse(frames[i:i + chunk + 1], fps)
            for row in r:
                row["frame"] += i
                row["t (s)"] = round(row["frame"] / fps, 2)
            rows += r; vis += v
            bar.progress(min(1.0, (i + chunk) / max(1, len(frames) - 1)))
        ss["m6_rows"], ss["m6_vid"] = rows, write_video(vis, fps)
        ss["m6_frames"] = frames[:2]
    if ss.get("m6_rows"):
        df = pd.DataFrame(ss["m6_rows"])
        st.video(ss["m6_vid"])
        st.caption("Left: arrows (red cross = focus of expansion). Right: hue = direction, brightness = speed.")
        st.line_chart(df, x="t (s)", y=["mean speed (px/frame)", "moving fraction"])
        st.line_chart(df, x="t (s)", y=["pan x", "pan y"])
        st.subheader("What the flow says")
        summ = df["inference"].value_counts()
        for txt, n in summ.head(5).items():
            st.write(f"- **{txt}** in {100 * n / len(df):.0f}% of frames")
        st.dataframe(df.round(3), hide_index=True)
        st.download_button("Download per-frame stats (CSV)", df.to_csv(index=False), "flow_stats.csv")
        st.download_button("Download flow video", ss["m6_vid"], "flow.mp4")

with t_lk:
    st.markdown("Two **consecutive frames** (photos or the first two frames of your video). Our tracker is "
                "compared with OpenCV's, and the patch at each predicted location is checked against the original.")
    a = st.file_uploader("Frame t", type=["jpg", "jpeg", "png"], key="m6a")
    b = st.file_uploader("Frame t+1", type=["jpg", "jpeg", "png"], key="m6b")
    pair = None
    if a and b:
        pair = decode_upload(a, 800), decode_upload(b, 800)
    elif ss.get("m6_frames"):
        pair = ss["m6_frames"]
        st.caption("Using the first two frames of the flow video.")
    if pair:
        f0, f1 = pair
        n = st.slider("Corners to track", 10, 150, 50)
        pts = good_features(f0, n)
        r = validate(f0, f1, pts)
        vis = f1.copy()
        for p, q, c in zip(pts, r["ours"], r["opencv"]):
            cv2.arrowedLine(vis, tuple(np.int32(p)), tuple(np.int32(q)), (0, 255, 0), 2, tipLength=0.3)
            cv2.circle(vis, tuple(np.int32(c)), 4, (255, 0, 255), 1)
        show(vis, "green = our LK, magenta circles = OpenCV LK")
        ok = r["ok"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Median |ours − OpenCV|", f"{np.median(r['diff_px'][ok]):.3f} px")
        c2.metric("Patch error, no motion", f"{np.mean(r['patch_err_no_motion']):.1f}")
        c3.metric("Patch error, predicted", f"{np.mean(r['patch_err_predicted']):.1f}")
        st.dataframe(pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "ours x": r["ours"][:, 0], "ours y": r["ours"][:, 1],
                                   "OpenCV x": r["opencv"][:, 0], "OpenCV y": r["opencv"][:, 1],
                                   "diff (px)": r["diff_px"]}).round(2), hide_index=True)

with t_sfm:
    st.markdown("Four photos of a **flat object** (a poster, a book cover, a box face) from different angles, "
                "with the camera calibrated in Module 2.")
    calib = calibration_widget("m6sfm")
    ups = st.file_uploader("Four photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="m6sfm4")
    if calib is not None and ups and len(ups) == 4:
        imgs = [decode_upload(u, 1200) for u in ups]
        K = K_for_image(calib, imgs[0].shape)
        if st.button("Reconstruct", type="primary") or "m6_sfm" in ss:
            if "m6_sfm" not in ss:
                with st.spinner("Matching and running bundle adjustment..."):
                    tr = match_tracks(imgs)
                    if len(tr) < 12:
                        st.error(f"Only {len(tr)} points seen in all four photos. Use a textured object and smaller angle changes.")
                        st.stop()
                    ss["m6_sfm"] = run_sfm(tr, K, calib["dist"]), tr
            res, tr = ss["m6_sfm"]
            st.write(f"{len(tr)} points in all four views; reprojection error "
                     f"{res['reproj_err_init_px']:.2f} → **{res['reproj_err_px']:.2f} px** after bundle adjustment; "
                     f"planarity {res['planarity']:.4f}")
            P = res["points3d"]
            st.scatter_chart(pd.DataFrame({"x": P[:, 0], "y": P[:, 1]}), x="x", y="y")
            st.markdown("**Tap the object's corners in photo 1**, in order around the edge.")
            corners = tap_points(imgs[0], "m6corners")
            known = st.number_input("Known length of side 1 (cm, 0 = unknown)", 0.0, 1000.0, 0.0)
            if len(corners) >= 3:
                C = corners_on_plane(res, K, undistort_points(corners, K, calib["dist"]))
                s = scale_from_length(C[0], C[1], known) if known > 0 else 1.0
                sides = np.linalg.norm(np.roll(C, -1, 0) - C, axis=1) * s
                unit = "cm" if known > 0 else "units (add a known length for cm)"
                for i, v in enumerate(sides):
                    st.metric(f"Side {i + 1}", f"{v:.2f} {unit}")
            if st.button("Start over"):
                ss.pop("m6_sfm", None); st.rerun()

with t_th:
    theory("module6")
