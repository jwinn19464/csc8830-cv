# CSc 8830 Computer Vision

One Streamlit web app that runs every module assignment and the final project. Each page works
on a phone: upload buttons open the camera or photo library, and you tap on images to place points.

```
app.py                       home page            ->  streamlit run app.py
pages/                       one page per module + Fridge Oracle
cvlib/                       shared image/video I/O and phone-friendly widgets
module2/  calibrate.py  measure.py  validate.py
module3/  blur.py
module4/  segment.py  metrics.py  fourier.py
module6/  flow.py  lk.py  sfm.py
module7/  stereo.py  room.py
fridge/   keyframes.py  items.py  dates.py  quantity.py  plan.py  recipes.json
docs/     theory notes shown in each page's Theory tab
tests/    checks on synthetic data with known answers   ->  python -m pytest -q
requirements.txt  packages.txt  .streamlit/config.toml
```

## What each assignment asks for, and where it lives

| Module | Assignment | Page tabs |
|---|---|---|
| 2 | Calibrate a phone camera with a checkerboard; measure a flat object's real size from one photo taken more than 2 m away using perspective projection; validate with 20 measurements; derive how two cameras relate | Calibrate · Measure · Validate · Theory |
| 3 | Blur an image by spatial convolution and in the Fourier domain, both from scratch, and show they match (convolution theorem) | Blur two ways · Theory |
| 4 | Exact human boundary in an RGB photo (Q1) and a thermal image (Q2) with no deep learning or ML; compare with SAM2; Fourier-domain edge detection and segmentation (Q3) | RGB · Thermal · vs SAM2 · Fourier · Theory |
| 6 | Optical flow on two 30 s clips with inferences; derive Lucas-Kanade and bilinear interpolation and validate on two frames; structure from motion from 4 photos of a flat object | Optical flow · Lucas-Kanade · Structure from motion · Theory |
| 7 | Object sizes from calibrated stereo (depth first); uncalibrated stereo by hand; room dimensions from stereo video | Stereo rig · Measure objects · Room size · Theory |
| Project | Fridge Oracle: a fridge/pantry sweep becomes a dated inventory and a cooking plan, reusing the modules | Scan · Inventory · What to cook · How it works |

Done outside the app: the SAM2 masks (GPU Colab, snippet at the top of `module4/metrics.py`) and the
handwritten Module 7 uncalibrated-stereo derivation.

## Run it on your computer

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                    # 11 tests should pass
streamlit run app.py
```

Date reading in Fridge Oracle needs Tesseract (`brew install tesseract`, or `sudo apt install tesseract-ocr`).
Without it, you type the dates in the Inventory tab.

## Deploy it (free) so it works on your phone

1. **Make a new GitHub repository** called `csc8830-cv` (public; keep it separate from your
   `jwinn19464.github.io` site repository). On a phone: github.com in the browser → **+** → New
   repository. Tick "Add a README file" so it isn't empty.
2. **Upload the files**: unzip `csc8830-cv.zip`, then in the new repository choose
   **Add file → Upload files** and drag in everything *inside* the `csc8830-cv` folder, keeping
   the folders (`pages/`, `module2/`, ...). The hidden `.streamlit` folder is optional; it
   only sets the dark theme. On iPhone, upload from the Files app with the browser in desktop
   mode, or do this step on a computer.
3. **Deploy on Streamlit Community Cloud**: go to share.streamlit.io, sign in with GitHub, choose
   **Create app → Deploy a public app from GitHub**, pick `jwinn19464/csc8830-cv`, branch `main`,
   main file `app.py`, and pick a custom URL such as `jenifer-cv`. Deploying takes a few minutes
   the first time (it installs `requirements.txt` and Tesseract from `packages.txt`).
4. **Open it on your phone**: `https://jenifer-cv.streamlit.app` (or the URL you chose). Add it
   to your home screen from the Share menu so it opens like an app.

Apps on the free tier go to sleep after several days without visitors. The first visit after that
shows a "wake up" button and takes about a minute.

## Link it from Jenifer's World

1. Sign in to your site's editor (click your name 5 times).
2. **Settings → Live app links**:
   - `portal` → `https://jenifer-cv.streamlit.app`
   - `fridge` → `https://jenifer-cv.streamlit.app/Fridge_Oracle`
3. Save. Every Prism module's **Live app** tab now opens the course app, and Fridge Oracle's opens
   the project page. The site adds `?embed=true` to Streamlit links by itself, which hides
   Streamlit's menu so the app fits inside the site.

To give a module its own page: in Settings → Live app links, type a new key such as `module3`,
tap **Add**, and set it to `https://jenifer-cv.streamlit.app/Module_3_Blur`. Then open that
memory's editor and set **Live app key** to `module3`. Page addresses are `Module_2_Measure`,
`Module_3_Blur`, `Module_4_Segment`, `Module_6_Motion`, `Module_7_Stereo` and `Fridge_Oracle`.

Fridge Oracle is unlisted on the site right now. Switch it to Public in the editor when you're
ready to show it.

## Tips for good results

- **Calibration**: 15–25 sharp photos of a flat checkerboard at different angles, 1× camera,
  the same orientation you'll measure with.
- **Measuring**: face the object squarely and measure the distance to the lens.
- **Flow clips**: one where the camera moves (walk forward, pan) and one where the camera is still
  and something moves.
- **Structure from motion**: a textured flat object (poster, cereal box face), angles under ~30°.
- **Stereo**: tape two phones to a ruler, or slide one phone along a table edge by a measured
  distance. Textured scenes match best; blank walls don't.
- **Fridge sweep**: move slowly, 10–30 s, labels facing the camera, good light.
