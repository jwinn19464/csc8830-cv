"""Fridge Oracle step 3 - read printed dates (Module 3 deblurring + OCR + parsing).

1. Text regions: morphological gradient -> horizontal closing -> wide, short components.
2. Deblur (Module 3): Wiener deconvolution in the Fourier domain with a Gaussian (or motion)
   PSF:  F_hat = conj(H) / (|H|^2 + 1/SNR) * G
3. OCR with Tesseract (pytesseract) when it is installed; otherwise type the date.
4. Parse many label formats: BEST BY 12/04, EXP 04DEC26, USE BY 2026-12-04, 12.04.26, DEC 4 2026...
"""
from __future__ import annotations

import datetime as dt
import re

import cv2
import numpy as np

from module3.blur import pad_kernel

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
MONTHS.update({"SEPT": 9})
KEYWORDS = r"(?:BEST\s*(?:BEFORE|BY|IF\s*USED\s*BY)|BB|USE\s*BY|EXP(?:IRES|IRY)?|SELL\s*BY|BBE)"


def wiener_deblur(gray, psf, snr=100.0):
    g = gray.astype(np.float64)
    H = np.fft.fft2(pad_kernel(psf, g.shape))
    G = np.fft.fft2(g)
    F = np.conj(H) / (np.abs(H) ** 2 + 1.0 / snr) * G
    return np.clip(np.real(np.fft.ifft2(F)), 0, 255).astype(np.uint8)


def text_regions(img_bgr, min_w=30):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    grad = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(9, g.shape[1] // 40)
    con = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3)))
    cs, _ = cv2.findContours(con, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_w and 6 <= h <= 0.3 * g.shape[0] and w / h > 2:
            boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda b: -b[2] * b[3])[:8]


def ocr(img, deblur_sigma=0.0):
    if not HAVE_OCR:
        return ""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if deblur_sigma > 0:
        from module3.blur import gaussian_kernel
        size = int(6 * deblur_sigma) | 1
        g = wiener_deblur(g, gaussian_kernel(size, deblur_sigma))
    texts = [pytesseract.image_to_string(g, config="--psm 6")]
    for x, y, w, h in text_regions(g):
        texts.append(pytesseract.image_to_string(g[y:y + h, x:x + w], config="--psm 7"))
    return "\n".join(t.strip() for t in texts if t.strip())


def _year(y, today):
    y = int(y)
    if y < 100:
        y += 2000
    return y


def _mk(y, m, d):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def parse_dates(text: str, today: dt.date | None = None, day_first: bool = False):
    """All plausible dates in text, most label-like first. US month/day by default."""
    today = today or dt.date.today()
    t = text.upper().replace("|", "1")
    t = re.sub(r"(?<=\d)[O]|[O](?=\d)", "0", t)      # OCR reads 0 as O next to digits
    t = re.sub(r"(?<=\d)[Il]|[Il](?=\d)", "1", t)
    for wrong, right in (("N0V", "NOV"), ("0CT", "OCT"), ("SEPT", "SEP")):
        t = t.replace(wrong, right)
    found = []

    def add(d, pos):
        if d and today - dt.timedelta(days=3 * 365) <= d <= today + dt.timedelta(days=6 * 365):
            near = bool(re.search(KEYWORDS, t[max(0, pos - 25):pos]))
            found.append((not near, pos, d))

    for m in re.finditer(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", t):
        add(_mk(int(m[1]), int(m[2]), int(m[3])), m.start())
    for m in re.finditer(r"(?<!\d)(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})(?!\d)", t):
        a, b, y = int(m[1]), int(m[2]), _year(m[3], today)
        mo, da = (b, a) if day_first else (a, b)
        add(_mk(y, mo, da) or _mk(y, da, mo), m.start())
    mon = "|".join(sorted(MONTHS, key=len, reverse=True))
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}})\s*({mon})[A-Z]*\.?\s*,?\s*(\d{{2,4}})?", t):
        y = _year(m[3], today) if m[3] else today.year
        d = _mk(y, MONTHS[m[2]], int(m[1]))
        if d and not m[3] and d < today - dt.timedelta(days=60):
            d = _mk(y + 1, MONTHS[m[2]], int(m[1]))
        add(d, m.start())
    for m in re.finditer(rf"({mon})[A-Z]*\.?\s*(\d{{1,2}})(?:\s*,?\s*(\d{{4}}|\d{{2}}(?!\d)))?", t):
        y = _year(m[3], today) if m[3] else today.year
        d = _mk(y, MONTHS[m[1]], int(m[2]))
        if d and not m[3] and d < today - dt.timedelta(days=60):
            d = _mk(y + 1, MONTHS[m[1]], int(m[2]))
        add(d, m.start())
    # bare month/day after a keyword, e.g. BEST BY 12/04
    for m in re.finditer(rf"{KEYWORDS}\s*:?\s*(\d{{1,2}})[/.-](\d{{1,2}})(?![/.-]?\d)", t):
        a, b = int(m[1]), int(m[2])
        mo, da = (b, a) if day_first else (a, b)
        d = _mk(today.year, mo, da)
        if d and d < today - dt.timedelta(days=60):
            d = _mk(today.year + 1, mo, da)
        add(d, m.end())
    found.sort(key=lambda x: (x[0], x[1]))
    out = []
    for _, _, d in found:
        if d not in out:
            out.append(d)
    return out
