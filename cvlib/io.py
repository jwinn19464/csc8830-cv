"""Shared image/video I/O and phone-friendly Streamlit helpers."""
from __future__ import annotations

import io
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 1600  # phone photos are 12-48 MP; resize for speed (intrinsics are rescaled to match)


def decode_upload(uploaded, max_side: int | None = MAX_SIDE) -> np.ndarray:
    """Uploaded file / camera_input -> BGR uint8, EXIF orientation applied, optionally downscaled."""
    data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)[:, :, ::-1].copy()
    if max_side:
        arr = limit_size(arr, max_side)
    return arr


def limit_size(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        return img_bgr
    return np.ascontiguousarray(img_bgr[:, :, ::-1])


def gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def read_video(uploaded_or_path, max_seconds: float | None = 30, max_side: int = 640, step: int = 1):
    """Return (frames BGR list, fps). Frames are downscaled for phone-speed processing."""
    path = uploaded_or_path
    if hasattr(uploaded_or_path, "getvalue"):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(uploaded_or_path.getvalue())
        tmp.close()
        path = tmp.name
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(max_seconds * fps) if max_seconds else None
    frames, i = [], 0
    while True:
        ok, f = cap.read()
        if not ok or (limit and i >= limit):
            break
        if i % step == 0:
            frames.append(limit_size(f, max_side))
        i += 1
    cap.release()
    return frames, fps / step


def write_video(frames, fps: float) -> bytes:
    """Encode frames as H.264 MP4 (plays in iPhone and Android browsers)."""
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    try:
        import imageio_ffmpeg
        wr = imageio_ffmpeg.write_frames(tmp.name, (w2, h2), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                                         output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"])
        wr.send(None)
        for f in frames:
            f = f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            wr.send(np.ascontiguousarray(f[:h2, :w2, ::-1]))
        wr.close()
    except ImportError:  # fall back to OpenCV (may not play in Safari)
        vw = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
        for f in frames:
            vw.write((f if f.ndim == 3 else cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))[:h2, :w2])
        vw.release()
    return open(tmp.name, "rb").read()
