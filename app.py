"""
Prototipe Sistem Deteksi dan Klasifikasi Otomatis (SDK-O) Rudal Jelajah
Sistem Deteksi & Klasifikasi Objek Udara Berbasis LSTM
Skenario 7 (x15 Balanced) | CRISP-DM Deployment Phase
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import logging
import traceback
import io
import json
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# KONFIGURASI LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    filename='app_errors.log',
    level=logging.ERROR,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_error(context: str, e: Exception):
    tb = traceback.format_exc()
    logging.error(f"[{context}] {str(e)}\n{tb}")

# ──────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SDK-O",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# DETEKSI MODE RAHASIA (URL PARAMETER)
# Akses via: http://localhost:8501?dev=true
# Hanya diketahui oleh peneliti
# ──────────────────────────────────────────────
query_params = st.query_params
DEV_MODE = query_params.get("dev", "false").lower() == "true"

# ──────────────────────────────────────────────
# KONFIGURASI PERSISTEN (FILE-BASED)
# File .application_config menyimpan pilihan model yang
# ditetapkan peneliti via dev mode.
# ──────────────────────────────────────────────
_CONFIG_FILE = '.application_config'

def _load_config():
    """Muat konfigurasi dari file. Return default jika belum ada."""
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'selected_split': 'S1 (70:15:15)', 'use_comparison': False}

def _save_config(cfg: dict):
    """Simpan konfigurasi ke file."""
    try:
        with open(_CONFIG_FILE, 'w') as f:
            json.dump(cfg, f)
    except Exception as e:
        log_error("save_config", e)

_cfg = _load_config()

# ──────────────────────────────────────────────
# KONSTANTA
# ──────────────────────────────────────────────
FEATURES          = ['Kecepatan', 'Ketinggian', 'Maneuver_Value']
CONFIDENCE_THRESH = 0.70
DURASI            = 20
WINDOW_TRANSISI   = 15

MANEUVER_MAP = {
    'Steady_Straight':     0,
    'Steady_Sea_Skimming': 1,
    'Terminal_Dash':       2,
    'Pop_Up_Dive':         3,
    'Evasive_Maneuver':    4,
    'Loitering_Pattern':   5,
    'Waypoint_Navigation': 6,
}

CLASS_INFO = {
    'Objek_NonAncaman': {
        'label': 'NON-ANCAMAN (AMAN)',
        'desc':  'Objek teridentifikasi sebagai pesawat sipil / non-rudal.',
        'type':  'success',
        'short': 'Non-Ancaman',
    },
    'Subsonic_Missile': {
        'label': 'RUDAL SUBSONIK — BAHAYA',
        'desc':  'Terdeteksi rudal dengan kecepatan di bawah Mach 1.',
        'type':  'warning',
        'short': 'Subsonik',
    },
    'Supersonic_Missile': {
        'label': 'RUDAL SUPERSONIK — ANCAMAN KRITIS',
        'desc':  'Terdeteksi rudal dengan kecepatan tinggi / supersonik.',
        'type':  'error',
        'short': 'Supersonik',
    },
}

# Path artefak
MODEL_S1     = 'models/model_S1.keras'
MODEL_S2     = 'models/model_S2.keras'
MODEL_S3     = 'models/model_S3.keras'
SCALER_PATH  = 'scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'
DATASET_PATH = 'dataset_final.csv'

# Default model untuk mode normal (S1 — hasil terbaik skenario 7)
DEFAULT_MODEL = MODEL_S1

SPLIT_LABELS = {
    'S1 (70:15:15)': MODEL_S1,
    'S2 (80:10:10)': MODEL_S2,
    'S3 (70:20:10)': MODEL_S3,
}

# ──────────────────────────────────────────────
# MUAT ARTEFAK (CACHED)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_scaler_encoder():
    errors  = []
    scaler, encoder = None, None
    for path in [SCALER_PATH, ENCODER_PATH]:
        if not os.path.exists(path):
            errors.append(f"File tidak ditemukan: `{path}`")
    if errors:
        return None, None, errors
    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        log_error("load_scaler", e)
        errors.append(f"Gagal memuat scaler: {str(e)}")
    try:
        encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        log_error("load_encoder", e)
        errors.append(f"Gagal memuat label encoder: {str(e)}")
    return scaler, encoder, errors


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    if not os.path.exists(model_path):
        return None, f"File tidak ditemukan: `{model_path}`"
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        return model, None
    except Exception as e:
        log_error(f"load_model:{model_path}", e)
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return None, f"File tidak ditemukan: `{DATASET_PATH}`"
    try:
        return pd.read_csv(DATASET_PATH), None
    except Exception as e:
        log_error("load_dataset", e)
        return None, str(e)

# ──────────────────────────────────────────────
# TEMPLATE CSV
# ──────────────────────────────────────────────
def generate_csv_template() -> bytes:
    rows = []
    for _ in range(DURASI):
        rows.append({
            'Kecepatan':      round(800.0 + np.random.normal(0, 4.0), 2),
            'Ketinggian':     round(50.0  + np.random.normal(0, 0.5), 2),
            'Maneuver_Value': 0,
        })
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')

# ──────────────────────────────────────────────
# FUNGSI PREDIKSI DENGAN STATUS BERTAHAP
# ──────────────────────────────────────────────
def predict_with_steps(df_input, model, scaler, encoder, status_container=None):
    def update(msg):
        if status_container:
            status_container.info(msg)
    try:
        update("⏳ **[1/4] Mengekstrak fitur dari data trajektori...**")
        raw = df_input[FEATURES].values.astype(float)
        if np.isnan(raw).any() or np.isinf(raw).any():
            raise ValueError("Data mengandung nilai NaN atau Inf.")

        update("⏳ **[2/4] Menormalisasi data dengan StandardScaler (Z-Score)...**")
        scaled = scaler.transform(raw)

        update(f"⏳ **[3/4] Reshape matriks ({DURASI}, {len(FEATURES)}) → (1, {DURASI}, {len(FEATURES)})...**")
        matrix3d = scaled.reshape(1, DURASI, len(FEATURES))

        update("⏳ **[4/4] Menjalankan inferensi model LSTM...**")
        probs    = model(matrix3d, training=False).numpy()[0]
        idx_best = int(np.argmax(probs))
        conf     = float(probs[idx_best])
        label    = encoder.classes_[idx_best]

        if status_container:
            status_container.success("✅ **Inferensi selesai.**")
        return probs, conf, label

    except Exception as e:
        if status_container:
            status_container.error(f"❌ **Proses gagal:** {str(e)}")
        log_error("predict_with_steps", e)
        raise

# ──────────────────────────────────────────────
# FUNGSI VISUALISASI
# ──────────────────────────────────────────────
def chart_speed(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Kecepatan'],
        mode='lines+markers', line=dict(width=2), name='Kecepatan (km/h)'
    ))
    fig.update_layout(title='Profil Kecepatan per Detik',
                      xaxis_title='Timestamp (detik)', yaxis_title='Kecepatan (km/h)',
                      height=280, margin=dict(l=40, r=20, t=40, b=40))
    return fig

def chart_altitude(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Timestamp'], y=df['Ketinggian'],
        mode='lines+markers', line=dict(color='green', width=2), name='Ketinggian (m)'
    ))
    fig.update_layout(title='Profil Ketinggian per Detik',
                      xaxis_title='Timestamp (detik)', yaxis_title='Ketinggian (m)',
                      height=280, margin=dict(l=40, r=20, t=40, b=40))
    return fig

def chart_confidence_tahap1(probs):
    """Bar chart Tahap 1 — Non-Ancaman vs Rudal (biner)."""
    prob_non    = float(probs[0]) * 100
    prob_rudal  = (float(probs[1]) + float(probs[2])) * 100
    fig = go.Figure(go.Bar(
        x=['Non-Ancaman', 'Rudal (Ancaman)'],
        y=[prob_non, prob_rudal],
        marker_color=['green', 'crimson'],
        text=[f'{prob_non:.1f}%', f'{prob_rudal:.1f}%'],
        textposition='outside',
    ))
    fig.add_hline(y=70, line_dash='dash', line_color='gray',
                  annotation_text='Threshold 70%', annotation_font_color='gray')
    fig.update_layout(
        title='Tahap 1 — Deteksi Ancaman: Non-Ancaman vs Rudal',
        yaxis_title='Probabilitas (%)', yaxis_range=[0, 120],
        height=300, margin=dict(l=40, r=20, t=50, b=40), showlegend=False
    )
    return fig

def chart_confidence_tahap2(probs):
    """Bar chart Tahap 2 — Subsonik vs Supersonik (hanya jika terdeteksi rudal)."""
    prob_sub = float(probs[1]) * 100
    prob_sup = float(probs[2]) * 100
    total    = prob_sub + prob_sup
    # Normalisasi relatif terhadap total probabilitas rudal
    sub_rel  = (prob_sub / total * 100) if total > 0 else 0
    sup_rel  = (prob_sup / total * 100) if total > 0 else 0
    fig = go.Figure(go.Bar(
        x=['Rudal Subsonik', 'Rudal Supersonik'],
        y=[sub_rel, sup_rel],
        marker_color=['orange', 'red'],
        text=[f'{sub_rel:.1f}%', f'{sup_rel:.1f}%'],
        textposition='outside',
    ))
    fig.add_hline(y=50, line_dash='dash', line_color='gray',
                  annotation_text='Batas 50%', annotation_font_color='gray')
    fig.update_layout(
        title='Tahap 2 — Klasifikasi Tipe Rudal: Subsonik vs Supersonik',
        yaxis_title='Probabilitas Relatif (%)', yaxis_range=[0, 120],
        height=300, margin=dict(l=40, r=20, t=50, b=40), showlegend=False
    )
    return fig

def chart_comparison(results: dict):
    split_names  = list(results.keys())
    class_labels = ['Non-Ancaman', 'Subsonik', 'Supersonik']
    colors       = ['green', 'orange', 'red']
    fig = go.Figure()
    for i, (cls_label, color) in enumerate(zip(class_labels, colors)):
        pct_vals = [float(results[s][0][i]) * 100 for s in split_names]
        fig.add_trace(go.Bar(
            name=cls_label, x=split_names, y=pct_vals,
            marker_color=color, opacity=0.85,
            text=[f'{v:.1f}%' for v in pct_vals], textposition='outside',
        ))
    fig.add_hline(y=70, line_dash='dash', line_color='gray',
                  annotation_text='Threshold 70%', annotation_font_color='gray')
    fig.update_layout(
        title='Perbandingan Confidence Ketiga Model (S1 vs S2 vs S3)',
        barmode='group', yaxis_title='Probabilitas (%)', yaxis_range=[0, 120],
        height=360, margin=dict(l=40, r=20, t=50, b=40), legend_title='Kelas',
    )
    return fig

# ──────────────────────────────────────────────
# RENDER HASIL HIERARKI — SINGLE MODEL
# ──────────────────────────────────────────────
def render_result(df_data, probs, conf, label, encoder):
    st.divider()

    kec_mean = df_data['Kecepatan'].mean()
    alt_mean = df_data['Ketinggian'].mean()
    m_val    = int(df_data['Maneuver_Value'].iloc[0])
    m_name   = next((k for k, v in MANEUVER_MAP.items() if v == m_val), str(m_val)).replace('_', ' ')

    # ── Metrik data trajektori ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Kecepatan Rerata (km/h)", f"{kec_mean:.1f}")
    c2.metric("Ketinggian Rerata (m)",   f"{alt_mean:.1f}")
    c3.metric("Jenis Manuver",           m_name)

    # ── Grafik profil ──
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_speed(df_data),    use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(chart_altitude(df_data), use_container_width=True, config={'displayModeBar': False})

    st.divider()

    # ════════════════════════════════════════════
    # TAHAP 1 — DETEKSI ANCAMAN (Non-Ancaman vs Rudal)
    # ════════════════════════════════════════════
    st.subheader("Tahap 1 — Deteksi Ancaman")

    prob_rudal = float(probs[1]) + float(probs[2])
    prob_non   = float(probs[0])

    # Cek confidence threshold pada level biner
    if conf < CONFIDENCE_THRESH:
        st.warning(
            f"**UNKNOWN / RAGU-RAGU**\n\n"
            f"Confidence tertinggi: **{conf*100:.1f}%** "
            f"(di bawah threshold {int(CONFIDENCE_THRESH*100)}%)\n\n"
            f"Sistem tidak dapat menentukan jenis objek secara determinan."
        )
        st.plotly_chart(chart_confidence_tahap1(probs),
                        use_container_width=True, config={'displayModeBar': False})
        with st.expander("Lihat Data Trajektori (20 Timestamp)"):
            cols_show = [c for c in ['Timestamp', 'Kecepatan', 'Ketinggian', 'Maneuver_Value']
                         if c in df_data.columns]
            st.dataframe(df_data[cols_show].style.format(
                {'Kecepatan': '{:.2f}', 'Ketinggian': '{:.2f}'}),
                use_container_width=True, height=300)
        return  # Hentikan — tidak lanjut ke Tahap 2

    if label == 'Objek_NonAncaman':
        # Non-Ancaman — selesai di Tahap 1
        st.success(
            f"**STATUS: NON-ANCAMAN (AMAN)**\n\n"
            f"Objek teridentifikasi sebagai pesawat sipil / non-rudal.\n\n"
            f"Confidence: **{prob_non*100:.2f}%**"
        )
        st.plotly_chart(chart_confidence_tahap1(probs),
                        use_container_width=True, config={'displayModeBar': False})
        st.info("Objek diklasifikasikan sebagai Non-Ancaman. Klasifikasi tipe tidak diperlukan.")

    else:
        # Terdeteksi Rudal — lanjut ke Tahap 2
        st.error(
            f"**⚠️ TERDETEKSI: OBJEK ANCAMAN (RUDAL)**\n\n"
            f"Probabilitas ancaman: **{prob_rudal*100:.2f}%** — "
            f"Melanjutkan ke klasifikasi tipe rudal..."
        )
        st.plotly_chart(chart_confidence_tahap1(probs),
                        use_container_width=True, config={'displayModeBar': False})

        st.divider()

        # ════════════════════════════════════════
        # TAHAP 2 — KLASIFIKASI TIPE RUDAL
        # ════════════════════════════════════════
        st.subheader("Tahap 2 — Klasifikasi Tipe Rudal")

        if label == 'Subsonic_Missile':
            st.warning(
                f"**STATUS: RUDAL SUBSONIK — BAHAYA**\n\n"
                f"Terdeteksi rudal jelajah dengan kecepatan di bawah Mach 1 "
                f"(contoh: Exocet).\n\n"
                f"Confidence: **{float(probs[1])*100:.2f}%**"
            )
        else:
            st.error(
                f"**STATUS: RUDAL SUPERSONIK — ANCAMAN KRITIS**\n\n"
                f"Terdeteksi rudal jelajah dengan kecepatan tinggi / supersonik "
                f"(contoh: BrahMos).\n\n"
                f"Confidence: **{float(probs[2])*100:.2f}%**"
            )

        st.plotly_chart(chart_confidence_tahap2(probs),
                        use_container_width=True, config={'displayModeBar': False})

    with st.expander("Lihat Data Trajektori (20 Timestamp)"):
        cols_show = [c for c in ['Timestamp', 'Kecepatan', 'Ketinggian', 'Maneuver_Value']
                     if c in df_data.columns]
        st.dataframe(df_data[cols_show].style.format(
            {'Kecepatan': '{:.2f}', 'Ketinggian': '{:.2f}'}),
            use_container_width=True, height=300)

# ──────────────────────────────────────────────
# RENDER HASIL PERBANDINGAN (DEV MODE)
# ──────────────────────────────────────────────
def render_comparison(df_data, results: dict, encoder):
    st.divider()
    st.subheader("Hasil Perbandingan Ketiga Model")
    st.caption("Data yang sama diproses oleh S1, S2, dan S3 secara bersamaan.")

    rows_summary = []
    for split_name, (probs, conf, label) in results.items():
        info    = CLASS_INFO[label]
        verdict = "UNKNOWN" if conf < CONFIDENCE_THRESH else info['short']
        rows_summary.append({
            'Model Split':  split_name,
            'Prediksi':     verdict,
            'Confidence':   f"{conf*100:.2f}%",
            'Non-Ancaman':  f"{float(probs[0])*100:.1f}%",
            'Subsonik':     f"{float(probs[1])*100:.1f}%",
            'Supersonik':   f"{float(probs[2])*100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows_summary), use_container_width=True, hide_index=True)

    cols = st.columns(3)
    for i, (split_name, (probs, conf, label)) in enumerate(results.items()):
        with cols[i]:
            st.markdown(f"**{split_name}**")
            if conf < CONFIDENCE_THRESH:
                st.warning(f"UNKNOWN\n\n{conf*100:.1f}%")
            else:
                info = CLASS_INFO[label]
                msg  = f"**{info['short']}**\n\n{conf*100:.2f}%"
                if info['type'] == 'success':
                    st.success(msg)
                elif info['type'] == 'warning':
                    st.warning(msg)
                else:
                    st.error(msg)

    # Chart perbandingan Tahap 1 (biner)
    st.markdown("**Perbandingan Tahap 1 — Non-Ancaman vs Rudal**")
    fig_cmp = chart_comparison(results)
    st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_speed(df_data),    use_container_width=True, config={'displayModeBar': False})
    with col2:
        st.plotly_chart(chart_altitude(df_data), use_container_width=True, config={'displayModeBar': False})

    with st.expander("Lihat Data Trajektori (20 Timestamp)"):
        cols_show = [c for c in ['Timestamp', 'Kecepatan', 'Ketinggian', 'Maneuver_Value']
                     if c in df_data.columns]
        st.dataframe(
            df_data[cols_show].style.format({'Kecepatan': '{:.2f}', 'Ketinggian': '{:.2f}'}),
            use_container_width=True, height=300,
        )

# ──────────────────────────────────────────────
# INISIALISASI
# ──────────────────────────────────────────────
scaler, encoder, scaler_errors = load_scaler_encoder()
df_dataset, dataset_error      = load_dataset()

model_status = {}
for lbl, path in SPLIT_LABELS.items():
    m, err = load_model_cached(path)
    model_status[lbl] = {'model': m, 'error': err}

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Prototipe Sistem Deteksi dan Klasifikasi Otomatis (SDK-O) Rudal Jelajah")
    st.caption("")
    st.divider()

    st.subheader("Mode Input Data")
    mode = st.radio(
        label="Pilih jalur:",
        options=["📂 Data Historis", "📤 Unggah CSV"],
        label_visibility="collapsed"
    )

    # ── MENU RAHASIA — hanya muncul jika ?dev=true ──
    # Akses: http://localhost:8501?dev=true
    # Pilihan disimpan ke file .application_config sehingga
    # tetap aktif meski URL berubah atau aplikasi di-restart.
    if DEV_MODE:
        st.divider()
        st.caption("⚙️ Developer Mode")

        use_comparison = st.toggle(
            "Bandingkan S1 / S2 / S3",
            value=_cfg.get('use_comparison', False),
        )
        if not use_comparison:
            split_options = list(SPLIT_LABELS.keys())
            saved_index   = split_options.index(_cfg.get('selected_split', 'S1 (70:15:15)')) \
                            if _cfg.get('selected_split') in split_options else 0
            selected_split = st.selectbox(
                "Model Split Aktif:",
                options=split_options,
                index=saved_index,
            )
        else:
            selected_split = _cfg.get('selected_split', 'S1 (70:15:15)')
            st.caption("Ketiga model dijalankan bersamaan.")

        # Simpan ke file config setiap ada perubahan
        new_cfg = {'selected_split': selected_split, 'use_comparison': use_comparison}
        if new_cfg != _cfg:
            _save_config(new_cfg)
            _cfg = new_cfg

    else:
        use_comparison = _cfg.get('use_comparison', False)
        selected_split = _cfg.get('selected_split', 'S1 (70:15:15)')
    # ── AKHIR MENU RAHASIA ──

    # Tampilkan info skenario aktif kepada user (selalu terlihat)
    st.divider()
    active_label = "Semua Model (S1, S2, S3)" if use_comparison else selected_split
    st.caption(f"Model aktif : {active_label}")

    st.divider()
    st.caption(f"Durasi window  : {DURASI} detik")
    st.caption(f"Phase terminal : detik ke-{WINDOW_TRANSISI + 1}")
    st.caption(f"Conf. threshold: {int(CONFIDENCE_THRESH * 100)}%")
    st.caption("Scaler         : StandardScaler (Z-Score)")
    st.caption("Dataset        : Skenario 7 — x15 Balanced")
    st.caption("Log error      : app_errors.log")

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("Prototipe Sistem Deteksi dan Klasifikasi Otomatis (SDK-O) Rudal Jelajah")
st.caption(
    "Sistem Deteksi & Klasifikasi Objek Udara Berbasis LSTM  |  "
    "Skenario 7 — x15 Balanced  |  CRISP-DM Deployment Phase"
)
st.divider()

# Blokir jika scaler/encoder bermasalah
if scaler_errors:
    st.error("**Sistem tidak dapat dimulai. Periksa error berikut:**")
    for e in scaler_errors:
        st.error(f"• {e}")
    st.info(
        "**Struktur folder yang diperlukan:**\n"
        "```\n"
        "project-main/\n"
        "├── app.py\n"
        "├── convert_models.py\n"
        "├── export_test_csv.py\n"
        "├── scaler.pkl\n"
        "├── dataset_final.csv\n"
        "└── models/\n"
        "    ├── model_S1.keras\n"
        "    ├── model_S2.keras\n"
        "    ├── model_S3.keras\n"
        "    └── label_encoder.pkl\n"
        "```\n"
        "Jalankan `python3 convert_models.py` terlebih dahulu.\n"
        "Detail error tersimpan di `app_errors.log`."
    )
    st.stop()

# Warm-up model setelah halaman tampil
if 'models_warmed_up' not in st.session_state:
    wm = st.empty()
    wm.info("⏳ Sistem sedang mempersiapkan model, harap tunggu sebentar...")
    try:
        dummy = np.zeros((1, DURASI, len(FEATURES)), dtype=np.float32)
        for lbl, info in model_status.items():
            if info['model'] is not None:
                info['model'](dummy, training=False)
        st.session_state['models_warmed_up'] = True
        wm.empty()
    except Exception as e:
        log_error("warm_up", e)
        wm.warning(f"⚠️ Warm-up gagal (tidak kritis): {str(e)}")

# Tentukan model aktif
if DEV_MODE and use_comparison:
    active_models = {k: v['model'] for k, v in model_status.items()}
    missing = [k for k, v in model_status.items() if v['model'] is None]
    if missing:
        st.error(f"Model tidak tersedia: {missing}")
        st.stop()
else:
    active_model = model_status[selected_split]['model']
    if active_model is None:
        err = model_status[selected_split]['error']
        st.error(f"Model tidak dapat dimuat: {err}")
        st.stop()

# ══════════════════════════════════════════════
# JALUR 1: DATA HISTORIS
# ══════════════════════════════════════════════
if mode == "📂 Data Historis":
    st.subheader("Jalur 1 — Penarikan Data Historis")
    st.caption("Pilih Trajectory ID dari dataset yang tersedia, lalu jalankan deteksi.")

    if dataset_error:
        st.error(f"Dataset tidak dapat dimuat: {dataset_error}")
        st.caption("Detail error tersimpan di `app_errors.log`.")
        st.stop()

    id_col = 'Trajectory_ID'
    if id_col not in df_dataset.columns:
        st.error(f"Kolom `{id_col}` tidak ditemukan. Kolom tersedia: `{list(df_dataset.columns)}`")
        st.stop()

    all_ids        = sorted(df_dataset[id_col].unique())
    min_id, max_id = int(min(all_ids)), int(max(all_ids))

    col_sl, col_info = st.columns([3, 1])
    with col_sl:
        selected_id = st.slider(f"Trajectory ID ({min_id} — {max_id})",
                                min_value=min_id, max_value=max_id,
                                value=min_id, step=1)
    with col_info:
        st.metric("ID Aktif", selected_id)

    df_traj      = df_dataset[df_dataset[id_col] == selected_id].copy()
    missing_cols = [c for c in FEATURES if c not in df_traj.columns]
    if missing_cols:
        st.error(f"Kolom fitur tidak ditemukan: `{missing_cols}`")
        st.stop()
    if len(df_traj) < DURASI:
        st.warning(f"Trajektori ID {selected_id} hanya **{len(df_traj)} baris** (minimum **{DURASI}**).")
        st.stop()

    df_traj = df_traj.head(DURASI).reset_index(drop=True)
    if 'Timestamp' not in df_traj.columns:
        df_traj.insert(0, 'Timestamp', range(1, DURASI + 1))

    label_asli = df_traj['Label_Type'].iloc[0] if 'Label_Type' in df_traj.columns else '-'
    st.success(f"Trajektori ID **{selected_id}** berhasil dimuat — Label asli: **{label_asli}**")

    if st.button("Jalankan Deteksi", key="btn_hist", type="primary"):
        try:
            if DEV_MODE and use_comparison:
                results    = {}
                status_box = st.empty()
                for split_name, mdl in active_models.items():
                    status_box.info(f"⏳ Memproses model **{split_name}**...")
                    probs, conf, label = predict_with_steps(df_traj, mdl, scaler, encoder)
                    results[split_name] = (probs, conf, label)
                status_box.success("✅ Semua model selesai diproses.")
                render_comparison(df_traj, results, encoder)
            else:
                status_box = st.empty()
                probs, conf, label = predict_with_steps(
                    df_traj, active_model, scaler, encoder, status_box)
                render_result(df_traj, probs, conf, label, encoder)
        except Exception:
            st.error("Proses deteksi gagal. Detail error tersimpan di `app_errors.log`.")

# ══════════════════════════════════════════════
# JALUR 2: UNGGAH CSV
# ══════════════════════════════════════════════
elif mode == "📤 Unggah CSV":
    st.subheader("Jalur 2 — Unggah Data Eksternal (CSV)")
    st.caption("Unggah file CSV dari sensor atau sistem lain untuk dideteksi.")

    st.info(
        f"**Format yang diterima:**\n"
        f"- Jumlah baris : tepat **{DURASI} baris**\n"
        f"- Kolom wajib  : `Kecepatan`, `Ketinggian`, `Maneuver_Value`\n"
        f"- Format file  : `.csv` (separator koma, encoding UTF-8)"
    )

    with st.expander("📥 Unduh Template CSV"):
        st.markdown("Gunakan template ini sebagai panduan format. **Jangan ubah nama kolom.**")
        st.markdown("**Referensi nilai `Maneuver_Value`:**")
        df_ref = pd.DataFrame([
            {'Nilai': v, 'Nama Manuver': k.replace('_', ' ')}
            for k, v in MANEUVER_MAP.items()
        ])
        st.dataframe(df_ref, use_container_width=True, hide_index=True)
        st.download_button(
            label="⬇️  Unduh template_input.csv",
            data=generate_csv_template(),
            file_name="template_input.csv",
            mime="text/csv",
        )
        st.caption("Template berisi 20 baris data dummy. Ganti isinya sesuai kebutuhan.")

    st.divider()
    uploaded = st.file_uploader("Pilih atau seret file CSV", type=['csv'])

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
        except Exception as e:
            log_error("read_uploaded_csv", e)
            st.error(f"Gagal membaca file: {str(e)}")
            st.stop()

        if len(df_up) != DURASI:
            st.error(
                f"**Format Data Tidak Valid** — "
                f"Baris terdeteksi: **{len(df_up)}**, diperlukan: **{DURASI}**"
            )
            st.stop()

        missing = [c for c in FEATURES if c not in df_up.columns]
        if missing:
            st.error(
                f"**Format Data Tidak Valid** — Kolom tidak ditemukan: `{missing}`\n\n"
                f"Kolom terdeteksi: `{list(df_up.columns)}`"
            )
            st.stop()

        if 'Timestamp' not in df_up.columns:
            df_up.insert(0, 'Timestamp', range(1, DURASI + 1))

        st.success(
            f"Validasi berhasil — **{len(df_up)} baris**, "
            f"**{len(df_up.columns)} kolom** terdeteksi."
        )

        if st.button("Jalankan Deteksi", key="btn_upload", type="primary"):
            try:
                if DEV_MODE and use_comparison:
                    results    = {}
                    status_box = st.empty()
                    for split_name, mdl in active_models.items():
                        status_box.info(f"⏳ Memproses model **{split_name}**...")
                        probs, conf, label = predict_with_steps(df_up, mdl, scaler, encoder)
                        results[split_name] = (probs, conf, label)
                    status_box.success("✅ Semua model selesai diproses.")
                    render_comparison(df_up, results, encoder)
                else:
                    status_box = st.empty()
                    probs, conf, label = predict_with_steps(
                        df_up, active_model, scaler, encoder, status_box)
                    render_result(df_up, probs, conf, label, encoder)
            except Exception:
                st.error("Proses deteksi gagal. Detail error tersimpan di `app_errors.log`.")
