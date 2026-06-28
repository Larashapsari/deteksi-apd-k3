"""
=============================================================
  SISTEM DETEKSI APD PEKERJA PABRIK - K3
  Aplikasi Web (Streamlit + YOLOv8)
  Deteksi: Helm & Rompi (Vest)
  Kelas: helmet, no helmet, vest, no vest, person
  NOTE: Tidak pakai cv2 langsung — pakai PIL + numpy
=============================================================
"""

import streamlit as st
import numpy as np
import time
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "best_helmet_model.pt"
CONF_DEFAULT = 0.40

CLASS_COLORS = {
    'helmet'   : (0, 200, 0),
    'no helmet': (220, 0, 0),
    'vest'     : (0, 210, 210),
    'no vest'  : (220, 100, 0),
    'person'   : (150, 150, 150),
}

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model tidak ditemukan: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

# ============================================================
# FUNGSI DRAW BOUNDING BOX (pakai PIL, bukan cv2)
# ============================================================
def draw_boxes(pil_img, results, model):
    draw = ImageDraw.Draw(pil_img)
    stats = {
        'helmet': 0, 'no_helmet': 0, 'vest': 0, 'no_vest': 0,
        'person': 0, 'alert': False, 'alert_msg': []
    }

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name  = model.names.get(cls_id, str(cls_id))
            cls_lower = cls_name.strip().lower()

            # Statistik
            if cls_lower == 'helmet':
                stats['helmet'] += 1
            elif cls_lower in ('no helmet', 'no_helmet', 'nohelmet'):
                stats['no_helmet'] += 1
                stats['alert'] = True
                if 'Tanpa Helm' not in stats['alert_msg']:
                    stats['alert_msg'].append('Tanpa Helm')
            elif cls_lower == 'vest':
                stats['vest'] += 1
            elif cls_lower in ('no vest', 'no_vest', 'novest'):
                stats['no_vest'] += 1
                stats['alert'] = True
                if 'Tanpa Vest' not in stats['alert_msg']:
                    stats['alert_msg'].append('Tanpa Vest')
            elif cls_lower == 'person':
                stats['person'] += 1

            color = CLASS_COLORS.get(cls_lower, (200, 200, 200))
            # Gambar kotak
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            # Label background
            label = f"{cls_name} {conf_val:.2f}"
            text_bbox = draw.textbbox((x1, y1), label)
            draw.rectangle([x1, max(0, y1-20), x1+(text_bbox[2]-text_bbox[0])+6, y1], fill=color)
            draw.text((x1+3, max(0, y1-18)), label, fill=(255, 255, 255))

    return pil_img, stats

# ============================================================
# FUNGSI DETEKSI
# ============================================================
def detect(model, pil_img, conf):
    t0 = time.time()
    img_rgb = pil_img.convert("RGB")
    results = model.predict(np.array(img_rgb), conf=conf, verbose=False)
    inf_ms = (time.time() - t0) * 1000

    annotated, stats = draw_boxes(img_rgb.copy(), results, model)
    stats['inf_ms'] = inf_ms
    return annotated, stats

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Deteksi APD Pekerja Pabrik - K3",
    page_icon="🪖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f1a; }
    h1, h2, h3 { color: #ffffff !important; }
    .alert-box {
        background: #c0392b;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
        animation: blink 1s linear infinite;
    }
    .safe-box {
        background: #00b894;
        color: white;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    @keyframes blink { 50% { opacity: 0.7; } }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🪖 Sistem Deteksi APD Pekerja Pabrik")
st.markdown("**Deteksi Helm dan Rompi Keselamatan secara Real-Time — K3 Smart Factory**")
st.divider()

with st.spinner("⏳ Memuat model YOLOv8..."):
    model = load_model()

class_names = list(model.names.values())
st.success(f"✅ Model siap! Kelas: {class_names}")

with st.sidebar:
    st.markdown("## ⚙️ Pengaturan")
    conf = st.slider("Confidence Threshold", 0.10, 0.90, CONF_DEFAULT, 0.05)
    st.divider()
    st.markdown("## 🎨 Legenda Warna")
    st.markdown("🟢 **helmet** — Pakai helm")
    st.markdown("🔴 **no helmet** — Tidak pakai helm ⚠️")
    st.markdown("🔵 **vest** — Pakai rompi")
    st.markdown("🟠 **no vest** — Tidak pakai rompi ⚠️")
    st.markdown("⚪ **person** — Pekerja")
    st.divider()
    st.markdown("## 📋 Tentang")
    st.markdown("Sistem ini menggunakan **YOLOv8s** untuk mendeteksi kepatuhan APD pekerja pabrik.")

tab1, tab2, tab3 = st.tabs(["🖼️ Gambar", "🎬 Video", "📷 Kamera"])

# ── TAB 1: GAMBAR ───────────────────────────────────────────
with tab1:
    st.markdown("### Upload Gambar")
    uploaded = st.file_uploader("Pilih file gambar", type=["jpg","jpeg","png","bmp"], key="img_upload")

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gambar Asli**")
            st.image(img, use_column_width=True)

        with st.spinner("🔍 Mendeteksi..."):
            annotated, stats = detect(model, img, conf)

        with col2:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated, use_column_width=True)

        if stats['alert']:
            msg = " & ".join(stats['alert_msg'])
            st.markdown(f'<div class="alert-box">⚠️ PERINGATAN: Ada Pekerja {msg}!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ AMAN: Semua Pekerja Menggunakan APD Lengkap</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📊 Statistik Deteksi")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Helm",     stats['helmet'])
        c2.metric("⚠️ No Helm", stats['no_helmet'])
        c3.metric("✅ Vest",     stats['vest'])
        c4.metric("⚠️ No Vest", stats['no_vest'])
        c5.metric("⏱️ Waktu",   f"{stats['inf_ms']:.1f}ms")

# ── TAB 2: VIDEO ────────────────────────────────────────────
with tab2:
    st.markdown("### Upload Video")
    st.info("⚠️ Untuk video, diperlukan cv2. Fitur ini tidak tersedia di environment ini. Gunakan tab Gambar atau Kamera.")

# ── TAB 3: KAMERA ───────────────────────────────────────────
with tab3:
    st.markdown("### 📷 Kamera")
    st.info("Ambil foto dari kamera untuk deteksi APD.")

    camera_img = st.camera_input("Ambil foto dari kamera")

    if camera_img:
        img = Image.open(camera_img).convert("RGB")

        with st.spinner("🔍 Mendeteksi..."):
            annotated, stats = detect(model, img, conf)

        st.markdown("**Hasil Deteksi**")
        st.image(annotated, use_column_width=True)

        if stats['alert']:
            msg = " & ".join(stats['alert_msg'])
            st.markdown(f'<div class="alert-box">⚠️ PERINGATAN: Ada Pekerja {msg}!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ AMAN: APD Lengkap</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Helm",     stats['helmet'])
        c2.metric("⚠️ No Helm", stats['no_helmet'])
        c3.metric("✅ Vest",     stats['vest'])
        c4.metric("⚠️ No Vest", stats['no_vest'])
        c5.metric("⏱️ Waktu",   f"{stats['inf_ms']:.1f}ms")
