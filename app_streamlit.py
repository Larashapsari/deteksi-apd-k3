"""
=============================================================
  SISTEM DETEKSI APD PEKERJA PABRIK - K3
  Aplikasi Web (Streamlit + YOLOv8)
  Deteksi: Helm & Rompi (Vest)
  Kelas: helmet, no helmet, vest, no vest, person
=============================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# ============================================================
# KONFIGURASI
# ============================================================
MODEL_PATH = "best_helmet_model.pt"
CONF_DEFAULT = 0.40

CLASS_COLORS = {
    'helmet'   : (0, 200, 0),
    'no helmet': (0, 0, 220),
    'vest'     : (0, 210, 210),
    'no vest'  : (0, 100, 220),
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
# FUNGSI DETEKSI
# ============================================================
def detect(model, frame, conf):
    t0 = time.time()
    results = model.predict(frame, conf=conf, verbose=False)
    inf_ms = (time.time() - t0) * 1000

    annotated = frame.copy()
    stats = {'helmet': 0, 'no_helmet': 0, 'vest': 0, 'no_vest': 0,
             'person': 0, 'alert': False, 'alert_msg': [], 'inf_ms': inf_ms}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id   = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = model.names.get(cls_id, str(cls_id))
            cls_lower = cls_name.lower()

            # Statistik
            if cls_lower == 'helmet':
                stats['helmet'] += 1
            elif cls_lower == 'no helmet':
                stats['no_helmet'] += 1
                stats['alert'] = True
                if 'Tanpa Helm' not in stats['alert_msg']:
                    stats['alert_msg'].append('Tanpa Helm')
            elif cls_lower == 'vest':
                stats['vest'] += 1
            elif cls_lower == 'no vest':
                stats['no_vest'] += 1
                stats['alert'] = True
                if 'Tanpa Vest' not in stats['alert_msg']:
                    stats['alert_msg'].append('Tanpa Vest')
            elif cls_lower == 'person':
                stats['person'] += 1

            # Gambar bounding box
            color = CLASS_COLORS.get(cls_lower, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label = f"{cls_name} {conf_val:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.65, 2)
            cv2.rectangle(annotated, (x1, max(0, y1-th-12)), (x1+tw+6, y1), color, -1)
            cv2.putText(annotated, label, (x1+3, y1-4), font, 0.65, (255,255,255), 2)

    return annotated, stats

# ============================================================
# UI STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Deteksi APD Pekerja Pabrik - K3",
    page_icon="🪖",
    layout="wide"
)

# CSS Custom
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stApp { background-color: #0f0f1a; }
    h1, h2, h3 { color: #ffffff !important; }
    .metric-card {
        background: #16213e;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #0f3460;
    }
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
    @keyframes blink {
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🪖 Sistem Deteksi APD Pekerja Pabrik")
st.markdown("**Deteksi Helm dan Rompi Keselamatan secara Real-Time — K3 Smart Factory**")
st.divider()

# Load model
with st.spinner("⏳ Memuat model YOLOv8..."):
    model = load_model()
st.success(f"✅ Model siap! Kelas: {list(model.names.values())}")

# Sidebar
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
    st.markdown("Sistem ini menggunakan **YOLOv8s** untuk mendeteksi kepatuhan penggunaan APD pekerja pabrik secara real-time.")

# Tab input
tab1, tab2, tab3 = st.tabs(["🖼️ Gambar", "🎬 Video", "📷 Kamera"])

# ── TAB 1: GAMBAR ───────────────────────────────────────────
with tab1:
    st.markdown("### Upload Gambar")
    uploaded = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png", "bmp"],
        key="img_upload"
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Gambar Asli**")
            st.image(img, use_column_width=True)

        with st.spinner("🔍 Mendeteksi..."):
            annotated_bgr, stats = detect(model, frame_bgr, conf)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col2:
            st.markdown("**Hasil Deteksi**")
            st.image(annotated_rgb, use_column_width=True)

        # Status alert
        if stats['alert']:
            msg = " & ".join(stats['alert_msg'])
            st.markdown(f'<div class="alert-box">⚠️ PERINGATAN: Ada Pekerja {msg}!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ AMAN: Semua Pekerja Menggunakan APD Lengkap</div>', unsafe_allow_html=True)

        # Statistik
        st.divider()
        st.markdown("### 📊 Statistik Deteksi")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Helm", stats['helmet'])
        c2.metric("⚠️ No Helm", stats['no_helmet'])
        c3.metric("✅ Vest", stats['vest'])
        c4.metric("⚠️ No Vest", stats['no_vest'])
        c5.metric("⏱️ Waktu", f"{stats['inf_ms']:.1f}ms")

# ── TAB 2: VIDEO ────────────────────────────────────────────
with tab2:
    st.markdown("### Upload Video")
    video_file = st.file_uploader(
        "Pilih file video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid_upload"
    )

    if video_file:
        # Simpan ke temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)

        st.info(f"📹 Video: {total_frames} frame | {fps_video:.1f} FPS")

        # Tombol proses
        if st.button("▶️ Proses Video", type="primary"):
            stframe = st.empty()
            progress = st.progress(0)
            status_placeholder = st.empty()

            # Stats akumulasi
            total_alert = 0
            frame_count = 0
            process_every = max(1, int(fps_video / 10))  # proses 10 frame/detik

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                prog = min(frame_count / total_frames, 1.0)
                progress.progress(prog)

                if frame_count % process_every == 0:
                    annotated, stats = detect(model, frame, conf)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    stframe.image(annotated_rgb, use_column_width=True)

                    if stats['alert']:
                        total_alert += 1
                        msg = " & ".join(stats['alert_msg'])
                        status_placeholder.markdown(
                            f'<div class="alert-box">⚠️ {msg}!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        status_placeholder.markdown(
                            '<div class="safe-box">✅ APD Lengkap</div>',
                            unsafe_allow_html=True
                        )

            cap.release()
            os.unlink(tfile.name)
            st.success(f"✅ Video selesai diproses! Total frame dengan peringatan: {total_alert}")

# ── TAB 3: KAMERA ───────────────────────────────────────────
with tab3:
    st.markdown("### Kamera Live")
    st.info("📷 Gunakan webcam untuk deteksi real-time")

    # Streamlit camera input
    camera_img = st.camera_input("Ambil foto dari kamera")

    if camera_img:
        img = Image.open(camera_img).convert("RGB")
        frame = np.array(img)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Mendeteksi..."):
            annotated_bgr, stats = detect(model, frame_bgr, conf)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("**Hasil Deteksi**")
        st.image(annotated_rgb, use_column_width=True)

        if stats['alert']:
            msg = " & ".join(stats['alert_msg'])
            st.markdown(f'<div class="alert-box">⚠️ PERINGATAN: Ada Pekerja {msg}!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-box">✅ AMAN: APD Lengkap</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("✅ Helm", stats['helmet'])
        c2.metric("⚠️ No Helm", stats['no_helmet'])
        c3.metric("✅ Vest", stats['vest'])
        c4.metric("⚠️ No Vest", stats['no_vest'])
        c5.metric("⏱️ Waktu", f"{stats['inf_ms']:.1f}ms")
