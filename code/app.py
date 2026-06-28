"""
🔍 Automated Proctoring System — Streamlit Dashboard
=====================================================
Dashboard interaktif untuk analisis kecurangan wawancara daring.

Fitur:
- Upload video untuk analisis
- Real-time visualization (frame-by-frame)
- Timeline dashboard (risk score over time)
- Summary panel (overall risk, indicators, recommendations)
- Configuration panel (adjust thresholds)

Usage:
    streamlit run app.py
    streamlit run app.py -- --config configs/config.yaml
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(__file__))

try:
    import streamlit as st
except ImportError:
    print("[ERROR] Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from src.pipeline import ProctorPipeline
from src.cheating_scorer import RiskLevel


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Automated Proctoring System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .risk-normal { color: #4CAF50; font-weight: bold; font-size: 1.2em; }
    .risk-warning { color: #FFC107; font-weight: bold; font-size: 1.2em; }
    .risk-suspicious { color: #FF9800; font-weight: bold; font-size: 1.2em; }
    .risk-high_risk { color: #f44336; font-weight: bold; font-size: 1.2em; }
    .risk-critical { color: #9C27B0; font-weight: bold; font-size: 1.2em; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.9em; opacity: 0.8; }
    .metric-card p { margin: 0; font-size: 1.8em; font-weight: bold; }

    .indicator-bar {
        height: 8px;
        border-radius: 4px;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State
# ============================================================================
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "frame_data" not in st.session_state:
    st.session_state.frame_data = []


# ============================================================================
# Helper Functions
# ============================================================================

def get_risk_color(risk_level: str) -> str:
    """Warna untuk risk level."""
    colors = {
        "normal": "#4CAF50",
        "warning": "#FFC107",
        "suspicious": "#FF9800",
        "high_risk": "#f44336",
        "critical": "#9C27B0",
    }
    return colors.get(risk_level, "#9E9E9E")


def get_risk_emoji(risk_level: str) -> str:
    """Emoji untuk risk level."""
    emojis = {
        "normal": "✅",
        "warning": "⚠️",
        "suspicious": "⚠️",
        "high_risk": "🔴",
        "critical": "🚨",
    }
    return emojis.get(risk_level, "❓")


def analyze_video(video_path: str, config_path: str = "configs/config.yaml",
                  max_frames: Optional[int] = None, progress_bar=None):
    """Analisis video menggunakan ProctorPipeline."""
    pipeline = ProctorPipeline(config_path)
    pipeline.run(
        source=video_path,
        max_frames=max_frames,
        show_video=False,
        save_output=False,
    )
    results = pipeline.get_results()
    return results


def create_timeline_chart(results: Dict):
    """Buat timeline chart interaktif dari scoring intervals."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly not installed. Install with: pip install plotly")
        return

    scoring = results.get("scoring", {})
    intervals = scoring.get("intervals", [])

    if not intervals:
        st.info("Belum ada data interval untuk ditampilkan.")
        return

    timestamps = []
    scores = []
    risk_levels = []
    colors = []

    for iv in intervals:
        mid_time = (iv.get("interval_start", 0) + iv.get("interval_end", 0)) / 2
        timestamps.append(mid_time)
        scores.append(iv.get("total_score", 0))
        rl = iv.get("risk_level", "normal")
        risk_levels.append(rl)
        colors.append(get_risk_color(rl))

    fig = go.Figure()

    # Area chart background
    fig.add_trace(go.Scatter(
        x=timestamps, y=scores,
        fill="tozeroy",
        fillcolor="rgba(102, 126, 234, 0.1)",
        line=dict(color="rgba(102, 126, 234, 0.5)", width=1),
        showlegend=False,
    ))

    # Main line with colored markers
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=scores,
        mode="lines+markers",
        marker=dict(color=colors, size=10, line=dict(width=1, color="white")),
        line=dict(color="#667eea", width=2),
        name="Risk Score",
        hovertemplate="<b>Time:</b> %{x:.1f}s<br>"
                     "<b>Score:</b> %{y:.3f}<br>"
                     "<extra></extra>",
    ))

    # Threshold lines
    thresholds = scoring.get("thresholds", {})
    if thresholds:
        for name, val in thresholds.items():
            fig.add_hline(
                y=val, line_dash="dash",
                line_color=get_risk_color(name),
                annotation_text=name.replace("_", " ").title(),
                annotation_position="top right",
            )

    fig.update_layout(
        title="📈 Risk Score Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Risk Score",
        yaxis_range=[0, 1],
        height=400,
        template="plotly_white",
        margin=dict(l=50, r=30, t=50, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)


def create_indicator_chart(results: Dict):
    """Buat chart breakdown indikator."""
    if not PLOTLY_AVAILABLE:
        return

    scoring = results.get("scoring", {})
    intervals = scoring.get("intervals", [])

    if not intervals:
        return

    # Average indicator scores across all intervals
    indicator_totals = {}
    for iv in intervals:
        for name, val in iv.get("indicator_scores", {}).items():
            indicator_totals[name] = indicator_totals.get(name, 0) + val

    n = len(intervals)
    indicator_avgs = {k: v / n for k, v in indicator_totals.items()}

    if not indicator_avgs:
        return

    names = list(indicator_avgs.keys())
    values = list(indicator_avgs.values())

    # Sort by value
    sorted_pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_pairs)

    colors = ["#f44336" if v > 0.5 else "#FF9800" if v > 0.2 else "#4CAF50"
              for v in values]

    fig = go.Figure(go.Bar(
        x=list(values),
        y=[n.replace("_", " ").title() for n in names],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        title="📊 Indicator Breakdown (Average)",
        xaxis_title="Average Score",
        xaxis_range=[0, 1],
        height=350,
        template="plotly_white",
        margin=dict(l=150, r=30, t=50, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/search--v1.png", width=64)
    st.title("🔍 Proctoring System")
    st.markdown("---")

    mode = st.radio(
        "Mode Analisis",
        ["📁 Upload Video", "⚙️ Konfigurasi"],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    **Info Sistem:**
    - Face Detection: MediaPipe
    - Object Detection: YOLOv8
    - Temporal Model: LSTM
    - Scoring: Weighted Multi-Indicator
    """)

    st.markdown("---")
    st.caption("Skripsi — Universitas Indonesia 2025")


# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🔍 Automated Proctoring System")
st.markdown("**Sistem Deteksi Kecurangan Wawancara Daring Berbasis Analisis Nonverbal Multimodal**")

# ============================================================================
# Upload Video Mode
# ============================================================================
if mode == "📁 Upload Video":
    st.markdown("---")

    col_upload, col_options = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload video wawancara",
            type=["mp4", "avi", "mkv", "mov"],
            help="Format: MP4, AVI, MKV, MOV",
        )

    with col_options:
        max_frames = st.number_input(
            "Max frames (0 = semua)",
            min_value=0, max_value=10000, value=0, step=100,
        )
        config_path = st.text_input("Config path", value="configs/config.yaml")

    if uploaded_file is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        st.success(f"✅ Video uploaded: {uploaded_file.name}")

        # Video preview
        with st.expander("🎬 Preview Video", expanded=False):
            st.video(tfile.name)

        # Analyze button
        if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
            with st.spinner("🔄 Menganalisis video... Ini bisa memakan waktu beberapa menit"):
                try:
                    mf = max_frames if max_frames > 0 else None
                    results = analyze_video(tfile.name, config_path, mf)
                    st.session_state.analysis_results = results
                    st.success("✅ Analisis selesai!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

        # Clean up temp file
        try:
            os.unlink(tfile.name)
        except:
            pass

    # ============================================================================
    # RESULTS
    # ============================================================================
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results

        st.markdown("---")
        st.header("📊 Hasil Analisis")

        # === METRIC CARDS ===
        scoring = results.get("scoring", {})
        overall = scoring.get("overall", {})
        session = results.get("session", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            risk = overall.get("risk_level", "unknown")
            emoji = get_risk_emoji(risk)
            st.metric(
                "Risk Level",
                f"{emoji} {risk.replace('_', ' ').title()}",
            )

        with col2:
            score = overall.get("overall_score", 0)
            st.metric(
                "Risk Score",
                f"{score:.3f}",
                delta=f"{'High' if score > 0.6 else 'Normal'}" if score > 0 else None,
                delta_color="inverse",
            )

        with col3:
            total_frames = session.get("total_frames", 0)
            st.metric("Total Frames", f"{total_frames:,}")

        with col4:
            duration = session.get("duration_seconds", 0)
            st.metric("Duration", f"{duration:.1f}s")

        st.markdown("---")

        # === TABS ===
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Timeline", "📊 Indicators", "📋 Detail", "💾 Export"
        ])

        with tab1:
            create_timeline_chart(results)

        with tab2:
            create_indicator_chart(results)

            # Event log
            events = results.get("events", [])
            if events:
                st.subheader("🔔 Event Log")
                for ev in events[-20:]:  # Last 20 events
                    ev_type = ev.get("type", "unknown")
                    ev_time = ev.get("timestamp", 0)
                    ev_msg = ev.get("message", "")
                    emoji = "⚠️" if "suspicious" in ev_type.lower() else "ℹ️"
                    st.text(f"{emoji} [{ev_time:.1f}s] {ev_type}: {ev_msg}")

        with tab3:
            # Cheating type probabilities
            st.subheader("🎯 Deteksi per Kategori")
            probs = overall.get("cheating_probabilities", {})
            if probs:
                for cat, prob in probs.items():
                    col_name, col_bar = st.columns([1, 3])
                    with col_name:
                        st.text(cat.replace("_", " ").title())
                    with col_bar:
                        st.progress(min(prob, 1.0))

            # Scoring intervals detail
            intervals = scoring.get("intervals", [])
            if intervals:
                st.subheader("📋 Detail Interval")
                import pandas as pd
                df_data = []
                for iv in intervals:
                    df_data.append({
                        "Start (s)": f"{iv.get('interval_start', 0):.1f}",
                        "End (s)": f"{iv.get('interval_end', 0):.1f}",
                        "Score": f"{iv.get('total_score', 0):.3f}",
                        "Risk Level": iv.get("risk_level", "normal"),
                        "Top Indicators": ", ".join(iv.get("top_indicators", [])),
                    })
                if df_data:
                    st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        with tab4:
            st.subheader("💾 Export Results")

            # JSON export
            json_str = json.dumps(results, indent=2, default=str)
            st.download_button(
                "📥 Download JSON Report",
                data=json_str,
                file_name="proctoring_report.json",
                mime="application/json",
                use_container_width=True,
            )

            # Show raw JSON
            with st.expander("Raw JSON"):
                st.json(results)


# ============================================================================
# Configuration Mode
# ============================================================================
elif mode == "⚙️ Konfigurasi":
    st.markdown("---")
    st.header("⚙️ Konfigurasi Sistem")

    config_path = "configs/config.yaml"
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Cannot load config: {e}")
        config = {}

    if config:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Head Pose Thresholds")
            hp = config.get("head_pose", {})
            new_yaw = st.slider("Yaw Threshold (°)", 10.0, 60.0,
                               hp.get("yaw_threshold", 30.0), 1.0)
            new_pitch = st.slider("Pitch Threshold (°)", 10.0, 60.0,
                                 hp.get("pitch_threshold", 25.0), 1.0)
            new_roll = st.slider("Roll Threshold (°)", 10.0, 60.0,
                                hp.get("roll_threshold", 20.0), 1.0)

            st.subheader("👁️ Eye Gaze")
            eg = config.get("eye_gaze", {})
            new_offscreen = st.slider("Offscreen Threshold", 0.1, 0.8,
                                     eg.get("offscreen_gaze_threshold", 0.3), 0.05)
            new_reading_ratio = st.slider("Reading H/V Ratio", 1.0, 5.0,
                                         eg.get("reading_horizontal_ratio", 2.0), 0.1)

        with col2:
            st.subheader("📊 Scoring Weights")
            sc = config.get("scoring", {})
            weights = sc.get("weights", {})

            new_weights = {}
            for name, val in weights.items():
                new_weights[name] = st.slider(
                    name.replace("_", " ").title(),
                    0.0, 0.5, float(val), 0.01,
                )

            # Show weight sum
            weight_sum = sum(new_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                st.warning(f"⚠️ Weights sum: {weight_sum:.2f} (should be 1.0)")
            else:
                st.success(f"✅ Weights sum: {weight_sum:.2f}")

            st.subheader("🚨 Risk Thresholds")
            thresholds = sc.get("thresholds", {})
            use_thesis_scheme = ("warning" in thresholds) or (
                str(sc.get("risk_scheme", "legacy4")).lower() == "thesis3"
            )
            if use_thesis_scheme:
                new_warning = st.slider("Warning threshold", 0.05, 0.8,
                                        float(thresholds.get("warning", 0.3)), 0.05)
                new_suspicious = st.slider("Suspicious threshold", 0.2, 0.95,
                                          float(thresholds.get("suspicious", 0.7)), 0.05)
            else:
                new_suspicious = st.slider("Suspicious threshold", 0.1, 0.9,
                                          thresholds.get("suspicious", 0.3), 0.05)
                new_high_risk = st.slider("High Risk threshold", 0.2, 0.95,
                                         thresholds.get("high_risk", 0.6), 0.05)
                new_critical = st.slider("Critical threshold", 0.3, 1.0,
                                        thresholds.get("critical", 0.8), 0.05)

        # Save config
        if st.button("💾 Simpan Konfigurasi", type="primary"):
            try:
                config["head_pose"]["yaw_threshold"] = new_yaw
                config["head_pose"]["pitch_threshold"] = new_pitch
                config["head_pose"]["roll_threshold"] = new_roll
                config["eye_gaze"]["offscreen_gaze_threshold"] = new_offscreen
                config["eye_gaze"]["reading_horizontal_ratio"] = new_reading_ratio
                config["scoring"]["weights"] = new_weights
                if use_thesis_scheme:
                    config["scoring"]["risk_scheme"] = "thesis3"
                    config["scoring"]["thresholds"] = {
                        "warning": new_warning,
                        "suspicious": new_suspicious,
                    }
                else:
                    config["scoring"]["risk_scheme"] = "legacy4"
                    config["scoring"]["thresholds"] = {
                        "suspicious": new_suspicious,
                        "high_risk": new_high_risk,
                        "critical": new_critical,
                    }

                with open(config_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)

                st.success("✅ Konfigurasi tersimpan!")
            except Exception as e:
                st.error(f"❌ Error saving config: {e}")

        # Show current config
        with st.expander("📄 Current config.yaml"):
            st.code(yaml.dump(config, default_flow_style=False), language="yaml")


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.8em;'>
    Automated Proctoring System v1.0 |
    Skripsi oleh Christopher Satya Fredella Balakosa |
    Teknik Komputer — Universitas Indonesia 2025
</div>
""", unsafe_allow_html=True)
