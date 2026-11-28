import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # noqa: E702

import datetime as dt

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cvlib.io import decode_upload, read_video
from cvlib.ui import page, show
from fridge.dates import HAVE_OCR, ocr, parse_dates, suggest_names
from fridge.enhance import brighten, clean_keyframes
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
        ups = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="fo_imgs")
        one = st.toggle("Each photo shows one item (close-ups)", value=True,
                        help="Turn off for photos of a whole shelf, so it looks for several items in each photo.")
        if ups:
            frames = [decode_upload(u, 1000) for u in ups]
    scene = st.radio("Background", ["Cluttered (bags, piles, a full fridge)", "Plain shelf"], horizontal=False,
                     help="Cluttered looks for printed packaging; plain shelf looks for anything that differs from the shelf.")
    mode = "clutter" if scene.startswith("Cluttered") else "shelf"
    lowlight = st.toggle("Low-light cleanup", value=True,
                         help="Video: average each keyframe with its aligned neighbours, remove noise, brighten, and undo "
                              "motion blur measured from optical flow. Photos: remove noise and brighten.")
    deblur = 0.0
    if src == "Photos":
        deblur = st.slider("Extra deblur for date reading (Wiener, Gaussian σ; 0 = off)", 0.0, 2.0, 0.0, 0.25)
    if frames and st.button("Scan", type="primary"):
        with st.spinner("Picking sharp, non-redundant keyframes..."):
            keys, sharp = select_keyframes(frames) if src == "Video sweep" else (list(range(len(frames))), [])
        psfs = [None] * len(keys)
        if lowlight and src == "Video sweep":
            with st.spinner("Cleaning up dim frames (stacking neighbours, denoising, brightening)..."):
                kf, psfs, used = clean_keyframes(frames, keys)
            st.caption("Frames averaged per keyframe: " + ", ".join(map(str, used)))
            ss["fo_before"] = (frames[keys[0]], kf[0])
        elif lowlight:
            kf = [brighten(frames[k]) for k in keys]
            ss["fo_before"] = (frames[keys[0]], kf[0])
        else:
            kf = [frames[k] for k in keys]
        with st.spinner(f"Finding items in {len(kf)} image(s)..."):
            single = src == "Photos" and one
            boxes = [detect_items(f, one_item=single, mode=mode)[0] for f in kf]
            if src == "Photos":  # separate photos: no shared view to chain, so each box is its own item
                items = [{"views": [(k, b)], "global_box": b} for k, bs in enumerate(boxes) for b in bs]
            else:
                items = dedupe(kf, boxes)
        rows = []
        bar = st.progress(0.0, "Reading labels")
        for i, it in enumerate(items):
            crop, k = best_crop(kf, it)
            text = ocr(crop, deblur, psfs[k]) if HAVE_OCR else ""
            if HAVE_OCR and src == "Photos" and one:  # close-ups: the whole photo often reads better than the crop
                text += "\n" + ocr(kf[k], deblur)
            dates = parse_dates(text)
            names = suggest_names(text)
            rows.append({"keep": True, "name": names[0] if names else "", "expiry": dates[0] if dates else None,
                         "suggested": ", ".join(names), "seen in": len(it["views"]),
                         "label text": " ".join(w for w in text.split() if sum(c.isalnum() for c in w) >= 3)[:80]})
            it["crop"] = crop
            bar.progress((i + 1) / max(1, len(items)))
        ss["fo_kf"], ss["fo_items"], ss["fo_sharp"], ss["fo_keys"] = kf, items, sharp, keys
        ss["fo_df"] = pd.DataFrame(rows)
    if ss.get("fo_kf"):
        kf, items = ss["fo_kf"], ss["fo_items"]
        unit = "photo(s)" if not ss["fo_sharp"] else "keyframe(s)"
        st.success(f"{len(ss['fo_keys'])} {unit}, **{len(items)} item(s)**. Open the Inventory tab to check names and dates.")
        if ss.get("fo_before") and lowlight:
            with st.expander("Low-light cleanup: before and after"):
                b0, a0 = ss["fo_before"]
                c1, c2 = st.columns(2)
                c1.image(b0[:, :, ::-1], caption="as recorded", width="stretch")
                c2.image(a0[:, :, ::-1], caption="cleaned up", width="stretch")
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
        ss["fo_df"] = pd.DataFrame(columns=["keep", "name", "expiry", "suggested", "seen in", "label text"])
    items = ss.get("fo_items", [])
    if items:
        st.markdown("Check each numbered item's name (filled in from its label when it could read one) and its date. "
                    "Untick anything that isn't food. Dates are often printed on the lid or bottom, so photograph that side too.")
        thumbs = []
        for it in items:
            c = it["crop"]
            f = 160 / max(c.shape[:2])
            thumbs.append(cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)[:, :, ::-1])
        st.image(thumbs, caption=[f"#{n + 1}" for n in range(len(items))], width=96)
    df = st.data_editor(ss["fo_df"], num_rows="dynamic", width="stretch", key="fo_editor",
                        column_config={"expiry": st.column_config.DateColumn("expiry"),
                                       "keep": st.column_config.CheckboxColumn("keep"),
                                       "suggested": st.column_config.TextColumn("suggested", disabled=True),
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
| Low light | Neighbouring frames aligned (ORB + RANSAC homography, ECC fallback) and averaged; noise-matched denoising, gamma and CLAHE; motion-blur kernel from optical flow, undone by Wiener deconvolution | 6 + 3 |
| Items | Cluttered: "printed packaging" score (edges, colour variety, saturation) seeds a marker watershed; plain shelf: colour novelty + Fourier-gradient edges; close-ups: GrabCut | 4 |
| Count once | ORB + RANSAC homographies chain keyframes; boxes merged by overlap | 6 |
| Dates | Text regions, Wiener deconvolution, Tesseract OCR, parsing many label formats | 3 |
| How much is left | Liquid line from the row-brightness profile; known-size reference for mm | 2 |
| Hidden items | `fridge/quantity.py` flags items behind the front row using depths from a Module 7 stereo pair | 7 |
| Plan | Urgency from days left; recipes ranked by urgent food used minus purchases | — |
""")
