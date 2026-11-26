"""CSc 8830 Computer Vision - one web app for every module and the final project.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="CSc 8830 Computer Vision", page_icon="🔭", layout="centered")
st.title("Computer Vision, CSc 8830")
st.caption("Every module assignment and the final project run here. Open the menu (top left on a phone) to pick a page.")

PAGES = [
    ("pages/1_Module_2_Measure.py", "Module 2: Measure",
     "Calibrate a phone camera with a checkerboard, measure real sizes from one photo, validate 20 measurements."),
    ("pages/2_Module_3_Blur.py", "Module 3: Blur",
     "Blur by spatial convolution and in the Fourier domain, both from scratch, and show they match."),
    ("pages/3_Module_4_Segment.py", "Module 4: Find the Human",
     "Classical person segmentation in RGB and thermal images, compared with SAM2, plus Fourier edges."),
    ("pages/4_Module_6_Motion.py", "Module 6: Motion",
     "Optical flow with inferences, Lucas-Kanade from scratch, and structure from motion from four photos."),
    ("pages/5_Module_7_Stereo.py", "Module 7: Stereo",
     "Calibrated stereo: depth first, then object sizes, then a room's dimensions."),
    ("pages/6_Fridge_Oracle.py", "Final project: Fridge Oracle",
     "A fridge or pantry video becomes a dated inventory and a ranked cooking plan."),
]
for path, title, blurb in PAGES:
    with st.container(border=True):
        st.page_link(path, label=title, icon="➡️")
        st.write(blurb)
