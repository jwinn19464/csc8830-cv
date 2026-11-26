import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import datetime as dt

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cvlib.io import decode_upload, read_video
from cvlib.ui import page, show
from fridge.dates import HAVE_OCR, ocr, parse_dates
from fridge.items import best_crop, dedupe, detect_items
from fridge.keyframes import select_keyframes
from fridge.plan import days_left, rank, status
from fridge.quantity import fill_level

page("Fridge Oracle", "Sweep your phone slowly across a fridge shelf or pantry. Get a dated inventory and what to cook first.")
ss = st.session_state
t_scan, t_inv, t_plan, t_how = st.tabs(["1 · Scan", "2 · Inventory", "3 · What to cook", "How it works"])

with t_scan:
    src = st.radio("Input", ["Video sweep", "Photos"], horizontal=True)
    frames = None
    if src == "Video sweep":
        up = st.file_uploader("Video (10–30 s, slow sweep)", type=["mp4", "mov", "m4v"], key="fo_vid")
        if up is not None:
            frames, fps = read_video(up, 30, 720, 3)
    else:
        ups = st.file_uploader("Shelf photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="fo_imgs")
        if ups:
            frames = [decode_upload(u, 1000) for u in ups]
    deblur = st.slider("Deblur strength for date reading (Wiener, Gaussian σ)", 0.0, 3.0, 1.0, 0.25)
    if frames and st.button("Scan", type="primary"):
        with st.spinner("Picking sharp, non-redundant keyframes..."):
            keys, sharp = select_keyframes(frames) if src == "Video sweep" else (list(range(len(frames))), [])
            kf = [frames[k] for k in keys]
        with st.spinner(f"Finding items in {len(kf)} keyframe(s)..."):
            boxes = [detect_items(f)[0] for f in kf]
            items = dedupe(kf, boxes)
        rows = []
        bar = st.progress(0.0, "Reading labels")
        for i, it in enumerate(items):
            crop, k = best_crop(kf, it)
            text = ocr(crop, deblur) if HAVE_OCR else ""
            dates = parse_dates(text)
            rows.append({"keep": True, "name": "", "expiry": dates[0] if dates else None,
                         "seen in": len(it["views"]), "label text": text.replace("\n", " ")[:80]})
            it["crop"] = crop
            bar.progress((i + 1) / max(1, len(items)))
        ss["fo_kf"], ss["fo_items"], ss["fo_sharp"], ss["fo_keys"] = kf, items, sharp, keys
        ss["fo_df"] = pd.DataFrame(rows)
    if ss.get("fo_kf"):
        kf, items = ss["fo_kf"], ss["fo_items"]
        st.success(f"{len(ss['fo_keys'])} keyframe(s), **{len(items)} distinct item(s)**. Open the Inventory tab to name them.")
        if ss["fo_sharp"]:
            st.line_chart(pd.DataFrame({"sharpness (high-frequency energy)": ss["fo_sharp"]}))
        for k, f in enumerate(kf):
            vis = f.copy()
            for n, it in enumerate(items):
                for kk, (x, y, w, h) in it["views"]:
                    if kk == k:
                        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(vis, str(n + 1), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            show(vis, f"keyframe {k + 1}")
    if not HAVE_OCR:
        st.caption("Text reading (Tesseract) isn't installed here, so type the dates in the Inventory tab.")

with t_inv:
    if "fo_df" not in ss:
        ss["fo_df"] = pd.DataFrame(columns=["keep", "name", "expiry", "seen in", "label text"])
    items = ss.get("fo_items", [])
    if items:
        st.markdown("Name each numbered item and fix any date the scan got wrong. Untick anything that isn't food.")
        cols = st.columns(4)
        for n, it in enumerate(items):
            with cols[n % 4]:
                st.image(it["crop"][:, :, ::-1], caption=f"#{n + 1}", width="stretch")
    df = st.data_editor(ss["fo_df"], num_rows="dynamic", width="stretch", key="fo_editor",
                        column_config={"expiry": st.column_config.DateColumn("expiry"),
                                       "keep": st.column_config.CheckboxColumn("keep"),
                                       "seen in": st.column_config.NumberColumn("seen in", disabled=True),
                                       "label text": st.column_config.TextColumn("label text", disabled=True)})
    today = dt.date.today()
    inv = []
    for _, r in df.iterrows():
        if r.get("keep", True) is False or not str(r.get("name") or "").strip():
            continue
        e = r.get("expiry")
        e = None if e is None or (isinstance(e, float) and np.isnan(e)) or pd.isna(e) else pd.Timestamp(e).date()
        d = days_left(e, today)
        inv.append({"name": str(r["name"]).strip(), "expiry": e, "days left": d, "status": status(d)})
    ss["fo_inv"] = inv
    if inv:
        idf = pd.DataFrame(inv).sort_values("days left", na_position="last")
        exp = idf[idf["status"] == "expired"]
        if len(exp):
            st.error("Past its date: " + ", ".join(exp["name"]))
        st.dataframe(idf, hide_index=True)
        st.download_button("Download inventory.csv", idf.to_csv(index=False), "inventory.csv")
    if items:
        with st.expander("How much is left? (bottles and jars seen from the side)"):
            n = st.number_input("Item #", 1, len(items), 1)
            frac, prof = fill_level(items[n - 1]["crop"])
            st.metric("Estimated fill level", f"{100 * frac:.0f}%")
            st.line_chart(pd.DataFrame({"row brightness (top → bottom)": prof}))

with t_plan:
    inv = ss.get("fo_inv", [])
    if not inv:
        st.info("Name at least one item in the Inventory tab.")
    else:
        for r in rank(inv, 6):
            with st.container(border=True):
                st.subheader(f"{r['recipe']} · {r['minutes']} min")
                st.write("Uses: " + ", ".join(r["uses"]))
                if r["buy"]:
                    st.write("Need: " + ", ".join(r["buy"]))
                with st.expander("Steps"):
                    for i, s in enumerate(r["steps"]):
                        st.write(f"{i + 1}. {s}")
        st.caption("Recipes are ranked by how much soon-to-expire food they use, minus what you'd have to buy. "
                   "Expired items are never suggested.")

with t_how:
    st.markdown("""
| Step | Technique | Module |
|---|---|---|
| Keyframes | Fourier-domain sharpness rejects motion blur; optical flow drops redundant frames | 3 + 6 |
| Items | Colour novelty against shelf colours + Fourier-gradient edges, watershed split | 4 |
| Count once | ORB + RANSAC homographies chain keyframes; boxes merged by overlap | 6 |
| Dates | Text regions, Wiener deconvolution, Tesseract OCR, parsing many label formats | 3 |
| How much is left | Liquid line from the row-brightness profile; known-size reference for mm | 2 |
| Hidden items | `fridge/quantity.py` flags items behind the front row using depths from a Module 7 stereo pair | 7 |
| Plan | Urgency from days left; recipes ranked by urgent food used minus purchases | — |
""")
