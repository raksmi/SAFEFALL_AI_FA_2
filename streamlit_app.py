"""
CareVision HealthTech Pvt. Ltd. - SafeFall AI
FA-2, Step 7: Model Deployment using Streamlit

Pose estimation: YOLOv8 Pose (Ultralytics), installed directly from
Ultralytics' GitHub source repository (see requirements.txt — an HTTPS
archive URL, not a PyPI package name). Pose weights ('yolov8n-pose.pt')
are auto-downloaded from Ultralytics' GitHub Releases on first run and
cached locally.

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Then deploy on Streamlit Cloud (https://streamlit.io) by pushing this repo
(with fall_detection_model.h5, class_labels.json, requirements.txt,
packages.txt) to GitHub and connecting it in Streamlit Cloud's "New app"
flow.

Core dashboard features (per FA-2 Step 7 requirements):
  - Upload an image OR a video
  - Runs YOLOv8 Pose estimation + the trained CNN activity classifier
  - Displays fall alerts, prediction confidence, and pose visualization
  - Shows running monitoring analytics: total activities, fall count,
    normal-activity count, and an activity distribution chart

Extra features on top:
  - Tabbed dashboard (Live Monitor / Analytics / Incident Log) instead of
    disconnected side-by-side panels, plus a hero status header
  - Interactive Plotly charts (bar, donut, confidence timeline) instead
    of static matplotlib
  - Live webcam snapshot input (st.camera_input)
  - Adjustable alert confidence threshold
  - Audible alert tone on a confirmed fall (generated locally)
  - Resident/caregiver profile fields feeding a "would notify" demo card
  - Timestamped incident log, downloadable as CSV
  - "Time since last fall" live metric
  - Auto-generated plain-English session summary
  - Defensive handling of legacy/stale session-state entries
  - Uploaded videos are written to a temp file using their ORIGINAL
    extension (not hardcoded to .mp4) — feeding FFmpeg a mismatched
    container/extension is what was crashing the whole app on .avi
    uploads; temp files are also cleaned up after use
"""

import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import os
import subprocess
import time
import io
import wave
import base64
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH = "fall_detection_model.h5"
LABELS_PATH = "class_labels.json"
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"
IMG_SIZE = (224, 224)
FALL_CLASS = "fall"
VIDEO_FRAME_SKIP = 5
DEFAULT_ALERT_THRESHOLD = 0.60

ACTIVITY_COLORS = {
    "fall":     "#e63946",
    "walking":  "#4895ef",
    "sitting":  "#9d4edd",
    "standing": "#f4a261",
    "normal":   "#2a9d8f",
}
DEFAULT_COLOR = "#888888"


def color_for(activity: str) -> str:
    return ACTIVITY_COLORS.get(activity, DEFAULT_COLOR)


def load_classes():
    try:
        with open(LABELS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return ["fall", "walking", "sitting", "standing", "normal"]


CLASSES = load_classes()

st.set_page_config(
    page_title="SafeFall AI - Elderly Monitoring Dashboard",
    layout="wide",
    page_icon="🏥",
)

# --------------------------------------------------------------------------
# STYLING
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @keyframes flash-red {
        0%   { background-color: #7a0000; }
        50%  { background-color: #d10000; }
        100% { background-color: #7a0000; }
    }
    .emergency-banner {
        animation: flash-red 1s infinite;
        color: white; padding: 22px; border-radius: 10px; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-bottom: 14px;
        border: 2px solid #ffffff40;
    }
    .notify-card {
        background: linear-gradient(135deg, #241010, #1a0d0d);
        border-left: 5px solid #e63946; padding: 14px 18px; border-radius: 6px;
        margin-top: 10px;
    }
    .metric-card {
        border-radius: 10px; padding: 14px 16px; margin-bottom: 10px; color: white;
    }
    .metric-card .label { font-size: 0.8rem; opacity: 0.85; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; }
    .summary-card {
        background: linear-gradient(135deg, #10231f, #0d1a17);
        border-left: 5px solid #2a9d8f; padding: 16px 20px; border-radius: 8px;
        line-height: 1.7;
    }
    .hero {
        background: linear-gradient(120deg, #101820 0%, #16232e 60%, #101820 100%);
        border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
        border: 1px solid #2a3a47;
    }
    .hero h1 {
        margin: 0; font-size: 2rem;
        background: linear-gradient(90deg, #e63946, #4895ef, #2a9d8f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .status-chip {
        display: inline-block; padding: 5px 14px; border-radius: 20px;
        font-weight: 700; font-size: 0.85rem; margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, color):
    st.markdown(
        f"""<div class="metric-card" style="background:{color}22;border:1px solid {color};">
        <div class="label">{label}</div><div class="value" style="color:{color};">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# CACHED RESOURCES
# --------------------------------------------------------------------------
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_pose_model():
    return YOLO(YOLO_POSE_MODEL_NAME)


@st.cache_data
def generate_beep_base64():
    framerate = 44100
    duration = 0.45
    freq = 880.0
    t = np.linspace(0, duration, int(framerate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi) * np.linspace(1, 0.2, t.size)
    audio = (tone * (2 ** 15 - 1)).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def play_alert_sound():
    b64 = generate_beep_base64()
    st.markdown(
        f"""<audio autoplay><source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_fall_time" not in st.session_state:
    st.session_state.last_fall_time = None


def normalize_entry(h):
    """Resilient to stale session state left over from an older app
    version (e.g. a browser tab still open from before a redeploy)."""
    if isinstance(h, dict):
        return h
    label, confidence = h
    return {"timestamp": "-", "activity": label, "confidence": confidence}


# --------------------------------------------------------------------------
# CORE INFERENCE
# --------------------------------------------------------------------------
def strip_audio_track(input_path: str) -> str:
    """Remux the video WITHOUT its audio track using a real ffmpeg
    subprocess (isolated from OpenCV's own internal decoder).

    Some Le2i dataset .avi files carry a corrupted/malformed audio stream
    that crashes FFmpeg's mp3 decoder the moment anything tries to probe
    it (OpenCV's VideoCapture does this internally even though we never
    read audio). Since the app never needs audio at all, the safest fix
    is to remove it entirely before OpenCV ever opens the file — '-an'
    drops audio, '-vcodec copy' re-muxes the video stream instantly with
    no re-encoding (fast, lossless).

    Returns the path to the audio-free file, or the original path
    unchanged if ffmpeg isn't available or the strip step fails for any
    reason (best-effort — better to try the original file than to block
    the user entirely)."""
    suffix = Path(input_path).suffix or ".mp4"
    out_fd, out_path = tempfile.mkstemp(suffix=suffix)
    os.close(out_fd)
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-an", "-vcodec", "copy", out_path],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and os.path.getsize(out_path) > 0:
            return out_path
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    # Fall back to the original file if stripping failed for any reason.
    if os.path.exists(out_path):
        os.remove(out_path)
    return input_path


def run_pose_estimation(frame_bgr, pose_model):
    results = pose_model.predict(source=frame_bgr, verbose=False)
    result = results[0]
    annotated = result.plot()
    pose_found = result.boxes is not None and len(result.boxes) > 0
    return annotated, pose_found


def classify_frame(frame_bgr, model):
    """Same preprocess_input as train_model.py so inference matches training."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASSES[idx]
    confidence = float(preds[idx])
    return label, confidence


def record_prediction(label, confidence):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "activity": label,
        "confidence": confidence,
    }
    st.session_state.history.append(entry)
    if label == FALL_CLASS:
        st.session_state.last_fall_time = datetime.now()


def show_fall_alert(confidence, threshold, resident_name, caregiver_name, caregiver_contact, room):
    if confidence < threshold:
        st.warning(
            f"⚠️ Possible fall detected (confidence {confidence:.1%}) — below your "
            f"alert threshold of {threshold:.0%}, so no emergency alert was raised. "
            f"Still logged for review."
        )
        return

    st.markdown(
        f'<div class="emergency-banner">🚨 EMERGENCY: FALL DETECTED — {confidence:.1%} confidence 🚨</div>',
        unsafe_allow_html=True,
    )
    play_alert_sound()
    st.markdown(
        f"""
        <div class="notify-card">
        <b>📟 Alert that would be sent</b> <i>(demo only — no real message is sent;
        wire this up to Twilio/SendGrid/etc. for production)</i><br><br>
        <b>To:</b> {caregiver_name or 'Caregiver'} ({caregiver_contact or 'no contact set'})<br>
        <b>Resident:</b> {resident_name or 'Unnamed resident'}<br>
        <b>Location:</b> {room or 'Unspecified room'}<br>
        <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <b>Message:</b> "Fall detected for {resident_name or 'resident'} in
        {room or 'monitored area'} at {confidence:.1%} confidence. Please check immediately."
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_session_summary(history):
    if not history:
        return None
    total = len(history)
    counts = Counter(h["activity"] for h in history)
    most_common_activity, most_common_n = counts.most_common(1)[0]
    fall_count = counts.get(FALL_CLASS, 0)
    fall_rate = fall_count / total

    timestamps = [h["timestamp"] for h in history if h["timestamp"] != "-"]
    duration_line = ""
    if len(timestamps) >= 2:
        try:
            t0 = datetime.strptime(timestamps[0], "%Y-%m-%d %H:%M:%S")
            t1 = datetime.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
            span_min = max((t1 - t0).total_seconds() / 60, 0)
            duration_line = f" over roughly {span_min:.1f} minute(s) of monitoring"
        except ValueError:
            pass

    risk_word = "high" if fall_rate > 0.15 else ("moderate" if fall_rate > 0.03 else "low")
    return (
        f"**{total} predictions**{duration_line}. "
        f"Most frequent activity: **{most_common_activity}** ({most_common_n} times, "
        f"{most_common_n/total:.0%}). "
        f"**{fall_count} fall(s)** detected — a {fall_rate:.1%} fall rate, "
        f"which reads as **{risk_word} risk** for this session."
    )


# --------------------------------------------------------------------------
# UI - SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🏥 SafeFall AI")
st.sidebar.markdown("**CareVision HealthTech Pvt. Ltd.**\n\nElderly Fall Detection Monitoring")

with st.sidebar.expander("👤 Resident / caregiver profile", expanded=False):
    resident_name = st.text_input("Resident name", value="", placeholder="e.g. Mr. Sharma")
    room = st.text_input("Room / location", value="", placeholder="e.g. Room 4B")
    caregiver_name = st.text_input("Caregiver name", value="", placeholder="e.g. Nurse Priya")
    caregiver_contact = st.text_input("Caregiver phone/email", value="", placeholder="e.g. +91 98xxxxxxx")

st.sidebar.markdown("---")
alert_threshold = st.sidebar.slider(
    "🎚️ Alert confidence threshold", min_value=0.30, max_value=0.95,
    value=DEFAULT_ALERT_THRESHOLD, step=0.05,
    help="Lower = more sensitive (more alerts, more false alarms). "
         "Higher = fewer false alarms, but a real fall near the threshold "
         "might not trigger an emergency alert.",
)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Input type", ["Upload Image", "Upload Video", "Webcam Snapshot"])

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset monitoring session"):
    st.session_state.history = []
    st.session_state.last_fall_time = None
    st.sidebar.success("Session analytics reset.")

# --------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------
history_preview = [normalize_entry(h) for h in st.session_state.history]
recent_fall = (
    st.session_state.last_fall_time is not None
    and (datetime.now() - st.session_state.last_fall_time).total_seconds() < 30
)
status_text = "🚨 ALERT ACTIVE" if recent_fall else "🟢 MONITORING"
status_color = "#e63946" if recent_fall else "#2a9d8f"

st.markdown(
    f"""
    <div class="hero">
        <h1>SafeFall AI — Elderly Fall Detection Dashboard</h1>
        <p style="opacity:0.8;margin:4px 0 0 0;">
            Computer Vision + Deep Learning monitoring for elderly safety
        </p>
        <span class="status-chip" style="background:{status_color}33;color:{status_color};
              border:1px solid {status_color};">{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    classifier = load_classifier()
    pose_model = load_pose_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(
        f"Could not load '{MODEL_PATH}'. Train the model first with train_model.py "
        f"and place the .h5 file next to this app. ({e})"
    )

tab_monitor, tab_analytics, tab_log = st.tabs(["🎥 Live Monitor", "📊 Analytics & Insights", "📝 Incident Log"])


def process_and_show_image(frame):
    annotated, pose_found = run_pose_estimation(frame, pose_model)
    label, confidence = classify_frame(frame, classifier)
    record_prediction(label, confidence)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Pose Estimation Output", width='stretch')

    if label == FALL_CLASS:
        show_fall_alert(confidence, alert_threshold, resident_name, caregiver_name, caregiver_contact, room)
    else:
        st.success(f"✅ Activity: {label.capitalize()} (confidence {confidence:.1%})")

    if not pose_found:
        st.warning("No body landmarks detected in this image — check lighting/angle.")


# --------------------------------------------------------------------------
# TAB 1: LIVE MONITOR
# --------------------------------------------------------------------------
with tab_monitor:
    if not model_loaded:
        st.info("Dashboard preview only — connect a trained model to enable live predictions.")
    elif mode == "Upload Image":
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_show_image(frame)

    elif mode == "Webcam Snapshot":
        st.caption("Take a live snapshot from your device camera — useful for a quick spot-check.")
        snapshot = st.camera_input("Camera")
        if snapshot is not None:
            file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_show_image(frame)

    elif mode == "Upload Video":
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            # IMPORTANT: use the ORIGINAL file extension, not a hardcoded one.
            # Forcing e.g. .avi bytes into a ".mp4"-named temp file makes
            # FFmpeg misjudge the container and can crash the whole process
            # (not just raise a catchable Python exception).
            original_suffix = Path(uploaded_vid.name).suffix.lower() or ".mp4"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)
            tfile.write(uploaded_vid.read())
            tfile.close()
            temp_path = tfile.name

            with st.spinner("Preparing video (stripping audio track for safe decoding)..."):
                safe_path = strip_audio_track(temp_path)

            try:
                cap = cv2.VideoCapture(safe_path)
                if not cap.isOpened():
                    st.error(
                        "Could not open this video file. Try re-exporting it as a "
                        "standard H.264 .mp4 — some older/unusual codecs aren't "
                        "supported by the server's video backend."
                    )
                else:
                    frame_placeholder = st.empty()
                    alert_placeholder = st.empty()
                    progress = st.progress(0)

                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_idx += 1

                        if frame_idx % VIDEO_FRAME_SKIP == 0:
                            annotated, pose_found = run_pose_estimation(frame, pose_model)
                            label, confidence = classify_frame(frame, classifier)
                            record_prediction(label, confidence)

                            frame_placeholder.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_idx} — {label.capitalize()} ({confidence:.1%})",
                                width='stretch',
                            )
                            with alert_placeholder.container():
                                if label == FALL_CLASS:
                                    show_fall_alert(confidence, alert_threshold, resident_name,
                                                     caregiver_name, caregiver_contact, room)
                                else:
                                    st.info(f"Monitoring... current activity: {label.capitalize()}")

                        progress.progress(min(frame_idx / total_frames, 1.0))
                        time.sleep(0.02)

                    cap.release()
                    st.success("Video processing complete.")
            finally:
                # Clean up both temp files regardless of success/failure.
                for p in {temp_path, safe_path}:
                    if os.path.exists(p):
                        os.remove(p)

# --------------------------------------------------------------------------
# TAB 2: ANALYTICS
# --------------------------------------------------------------------------
with tab_analytics:
    history = [normalize_entry(h) for h in st.session_state.history]
    total = len(history)
    fall_count = sum(1 for h in history if h["activity"] == FALL_CLASS)
    normal_count = total - fall_count

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Activities", total, "#4895ef")
    with c2:
        metric_card("Falls Detected", fall_count, ACTIVITY_COLORS[FALL_CLASS])
    with c3:
        metric_card("Normal Activity", normal_count, ACTIVITY_COLORS["normal"])
    with c4:
        if st.session_state.last_fall_time:
            elapsed = datetime.now() - st.session_state.last_fall_time
            mins = int(elapsed.total_seconds() // 60)
            metric_card("⏱️ Since last fall", f"{mins} min ago" if mins > 0 else "Just now", "#f4a261")
        else:
            metric_card("⏱️ Since last fall", "No falls yet", "#2a9d8f")

    if history:
        avg_conf = np.mean([h["confidence"] for h in history])
        st.markdown("**🧾 Session summary**")
        st.markdown(f'<div class="summary-card">{build_session_summary(history)} '
                     f'Average confidence across all predictions: **{avg_conf:.1%}**.</div>',
                     unsafe_allow_html=True)
        st.markdown("---")

        counts = Counter(h["activity"] for h in history)
        df_counts = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig_bar = px.bar(
                df_counts, x="Activity", y="Count", color="Activity",
                color_discrete_map=ACTIVITY_COLORS, title="Activity Distribution",
            )
            fig_bar.update_layout(showlegend=False, height=340)
            st.plotly_chart(fig_bar, width='stretch')

        with chart_col2:
            fig_pie = px.pie(
                df_counts, names="Activity", values="Count", color="Activity",
                color_discrete_map=ACTIVITY_COLORS, title="Activity Share", hole=0.45,
            )
            fig_pie.update_layout(height=340)
            st.plotly_chart(fig_pie, width='stretch')

        # Interactive confidence timeline
        timeline_df = pd.DataFrame(history)
        timeline_df["index"] = range(1, len(timeline_df) + 1)
        fig_line = px.scatter(
            timeline_df, x="index", y="confidence", color="activity",
            color_discrete_map=ACTIVITY_COLORS,
            title="Confidence Timeline (color = predicted activity)",
            labels={"index": "Prediction #", "confidence": "Confidence", "activity": "Activity"},
            hover_data={"timestamp": True},
        )
        fig_line.add_hline(
            y=alert_threshold, line_dash="dash", line_color="#e63946",
            annotation_text=f"Alert threshold ({alert_threshold:.0%})",
            annotation_position="top left",
        )
        fig_line.update_traces(mode="lines+markers", line=dict(color="#555", width=1))
        fig_line.update_layout(height=320, yaxis_range=[0, 1.05])
        st.plotly_chart(fig_line, width='stretch')
    else:
        st.caption("No predictions yet — upload an image or video in the Live Monitor tab to begin.")

    st.markdown("---")
    st.caption(
        "⚠️ Known deployment limitations: accuracy can degrade under poor lighting, "
        "unusual camera angles, occlusion, or postures resembling a fall (e.g. lying "
        "on a sofa). Periodic retraining with new footage is recommended. Alert "
        "notifications above are a UI demo only — no SMS/email actually sends."
    )

# --------------------------------------------------------------------------
# TAB 3: INCIDENT LOG
# --------------------------------------------------------------------------
with tab_log:
    history = [normalize_entry(h) for h in st.session_state.history]
    if history:
        log_df = pd.DataFrame(history)
        log_df["confidence"] = (log_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(log_df.iloc[::-1], width='stretch', height=420)

        csv_bytes = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download incident log (CSV)",
            data=csv_bytes,
            file_name=f"safefall_incident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No incidents logged yet.")
