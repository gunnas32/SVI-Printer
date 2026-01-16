import io
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pypdfium2 as pdfium

# ==========================================================
# Cloud-deployable Streamlit app
# - Works on Streamlit Community Cloud / Linux
# - Uses file uploads (no local folder picker / no Windows printing APIs)
# - Provides a page placement preview for PDFs, TXT, and images
# ==========================================================

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Physical sizes for preview (mm)
PAPER_MM: Dict[str, Tuple[int, int]] = {
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
    "LETTER": (216, 279),
    "LEGAL": (216, 356),
}

SCALE_MODES = [
    "Fit (no crop)",
    "Fill (crop)",
    "Fit width (may crop)",
    "Fit height (may crop)",
]


@dataclass
class LoadedDoc:
    name: str
    ext: str
    data: bytes


# -----------------------------
# Rendering helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def _render_pdf_page_to_pil(pdf_bytes: bytes, page_index: int, target_px: int = 1600) -> Image.Image:
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[page_index]
    w, h = page.get_size()
    scale = target_px / max(w, h)
    bitmap = page.render(scale=scale)
    pil = bitmap.to_pil()
    page.close()
    pdf.close()
    return pil


def _txt_bytes_to_pil(text_bytes: bytes, max_width: int = 1800) -> Image.Image:
    content = text_bytes.decode("utf-8", errors="replace")
    lines = content.splitlines() or [""]

    try:
        # Works on Windows only; will fall back on Linux
        font = ImageFont.truetype("consola.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    dummy = Image.new("RGB", (10, 10), "white")
    draw = ImageDraw.Draw(dummy)

    line_heights: List[int] = []
    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_line_w = max(max_line_w, w)
        line_heights.append(max(h, 30))

    padding = 60
    img_w = min(max_width, max_line_w + padding * 2)
    img_h = sum(line_heights) + padding * 2

    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    y = padding
    for line, lh in zip(lines, line_heights):
        draw.text((padding, y), line, fill="black", font=font)
        y += lh

    return img


def _load_any_file_as_pil(doc: LoadedDoc, pdf_page: int = 0, preview_px: int = 1600) -> Tuple[Image.Image, str, int]:
    """
    Returns (image, label, pdf_page_count)
    """
    ext = doc.ext

    if ext == ".pdf":
        pdf = pdfium.PdfDocument(doc.data)
        n_pages = len(pdf)
        pdf.close()
        img = _render_pdf_page_to_pil(doc.data, pdf_page, target_px=preview_px)
        return img, f"PDF page {pdf_page + 1}", n_pages

    if ext == ".txt":
        img = _txt_bytes_to_pil(doc.data)
        return img, "Text rendered", 1

    # image
    img = Image.open(io.BytesIO(doc.data))
    return img, "Image", 1


# -----------------------------
# Scaling / placement logic
# -----------------------------
def _compute_draw_rect(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    scale_mode: str,
) -> Tuple[int, int, int, int]:
    """
    Returns (draw_w, draw_h, x, y) where x,y is top-left position in dst.
    draw_w/h can exceed dst for crop modes.
    """
    if scale_mode == "Fill (crop)":
        scale = max(dst_w / src_w, dst_h / src_h)
    elif scale_mode == "Fit width (may crop)":
        scale = dst_w / src_w
    elif scale_mode == "Fit height (may crop)":
        scale = dst_h / src_h
    else:  # "Fit (no crop)"
        scale = min(dst_w / src_w, dst_h / src_h)

    draw_w = int(src_w * scale)
    draw_h = int(src_h * scale)
    x = (dst_w - draw_w) // 2
    y = (dst_h - draw_h) // 2
    return draw_w, draw_h, x, y


def _visible_area(draw_w: int, draw_h: int, x: int, y: int, dst_w: int, dst_h: int) -> int:
    left = max(0, x)
    top = max(0, y)
    right = min(dst_w, x + draw_w)
    bottom = min(dst_h, y + draw_h)
    if right <= left or bottom <= top:
        return 0
    return (right - left) * (bottom - top)


def _maybe_rotate_to_best_fit(
    img: Image.Image,
    dst_w: int,
    dst_h: int,
    scale_mode: str,
    auto_rotate: bool,
) -> Tuple[Image.Image, bool]:
    if not auto_rotate:
        return img, False

    src_w, src_h = img.size

    # Not rotated
    dw0, dh0, x0, y0 = _compute_draw_rect(src_w, src_h, dst_w, dst_h, scale_mode)
    area0 = _visible_area(dw0, dh0, x0, y0, dst_w, dst_h)

    # Rotated 90 degrees
    dw1, dh1, x1, y1 = _compute_draw_rect(src_h, src_w, dst_w, dst_h, scale_mode)
    area1 = _visible_area(dw1, dh1, x1, y1, dst_w, dst_h)

    if area1 > area0:
        return img.transpose(Image.Transpose.ROTATE_90), True

    return img, False


def _paste_with_crop(canvas: Image.Image, img: Image.Image, x: int, y: int) -> None:
    cw, ch = canvas.size
    iw, ih = img.size

    dst_left = max(0, x)
    dst_top = max(0, y)
    dst_right = min(cw, x + iw)
    dst_bottom = min(ch, y + ih)
    if dst_right <= dst_left or dst_bottom <= dst_top:
        return

    src_left = dst_left - x
    src_top = dst_top - y
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)

    crop = img.crop((src_left, src_top, src_right, src_bottom))
    canvas.paste(crop, (dst_left, dst_top))


def build_page_preview(
    content_img: Image.Image,
    paper_key: str,
    landscape: bool,
    scale_mode: str,
    auto_rotate: bool,
    preview_width_px: int = 900,
) -> Tuple[Image.Image, Dict[str, str]]:
    w_mm, h_mm = PAPER_MM[paper_key]
    if landscape:
        w_mm, h_mm = h_mm, w_mm

    page_w = preview_width_px
    page_h = max(200, int(preview_width_px * (h_mm / w_mm)))
    canvas = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((5, 5, page_w - 6, page_h - 6), outline=(0, 0, 0), width=2)

    img, rotated = _maybe_rotate_to_best_fit(content_img, page_w - 20, page_h - 20, scale_mode, auto_rotate)

    inner_w = page_w - 20
    inner_h = page_h - 20
    src_w, src_h = img.size

    draw_w, draw_h, x, y = _compute_draw_rect(src_w, src_h, inner_w, inner_h, scale_mode)
    x += 10
    y += 10

    resized = img.convert("RGB").resize((max(1, draw_w), max(1, draw_h)), Image.Resampling.LANCZOS)
    _paste_with_crop(canvas, resized, x, y)

    info = {
        "paper": f"{paper_key} {'Landscape' if landscape else 'Portrait'}",
        "mode": scale_mode,
        "auto_rotate": "On" if auto_rotate else "Off",
        "rotated": "Yes" if rotated else "No",
    }
    return canvas, info


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    st.set_page_config(page_title="Page Preview (Upload & Fit)", layout="centered")
    st.title("📄 Page Preview (Upload & Fit)")
    st.caption("Upload PDFs / TXT / images, choose paper + scaling, preview placement, and download previews.")

    st.info(
        "Note: Streamlit Community Cloud runs on Linux, so Windows-only printing libraries (pywin32/tkinter file dialogs) "
        "won’t work there. This deployed version focuses on preview + export. "
        "If you want actual printer output, run your original Windows-only version locally."
    )

    uploaded = st.file_uploader(
        "Upload files",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if not uploaded:
        st.stop()

    docs: List[LoadedDoc] = []
    for uf in uploaded:
        name = uf.name
        ext = os.path.splitext(name)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        docs.append(LoadedDoc(name=name, ext=ext, data=uf.getvalue()))

    if not docs:
        st.warning("No supported files uploaded.")
        st.stop()

    # Sidebar settings
    with st.sidebar:
        st.subheader("Page settings")
        paper_key = st.selectbox("Paper size", list(PAPER_MM.keys()), index=1)  # A4 default
        landscape = st.checkbox("Landscape", value=False)
        scale_mode = st.selectbox("Scaling", SCALE_MODES, index=0)
        auto_rotate = st.checkbox("Auto-rotate to best fit", value=True)
        preview_width = st.slider("Preview width (px)", 500, 1200, 900, 50)

    # Select a doc
    sel_name = st.selectbox("Select a file", [d.name for d in docs])
    doc = next(d for d in docs if d.name == sel_name)

    pdf_page = 0
    n_pages = 1
    with st.spinner("Loading…"):
        # First load to get page count for PDFs
        if doc.ext == ".pdf":
            pdf = pdfium.PdfDocument(doc.data)
            n_pages = len(pdf)
            pdf.close()

    if doc.ext == ".pdf" and n_pages > 1:
        pdf_page = st.number_input("PDF page", min_value=1, max_value=n_pages, value=1, step=1) - 1

    with st.spinner("Rendering preview…"):
        content_img, content_label, _ = _load_any_file_as_pil(doc, pdf_page=pdf_page, preview_px=1600)
        preview_img, info = build_page_preview(
            content_img=content_img,
            paper_key=paper_key,
            landscape=landscape,
            scale_mode=scale_mode,
            auto_rotate=auto_rotate,
            preview_width_px=int(preview_width),
        )

    st.image(
        preview_img,
        caption=f"{doc.name} • {content_label} • {info['paper']} • {info['mode']} • Auto-rotate: {info['auto_rotate']} • Rotated: {info['rotated']}",
    )

    # Download preview
    out_name = f"preview_{os.path.splitext(doc.name)[0]}_{paper_key}{'_landscape' if landscape else ''}.png"
    st.download_button(
        "⬇️ Download preview PNG",
        data=_to_png_bytes(preview_img),
        file_name=out_name,
        mime="image/png",
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
