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

-----------------------------------------------------------------------
CHANGELOG (this revision)
-----------------------------------------------------------------------
Bug fixes (the "unknown error" on video upload):
  1. Temp video file was written with a hardcoded ".mp4" suffix regardless
     of the real upload type (.avi / .mov). OpenCV picks its decoder off
     the file extension, so a .mov/.avi upload landed in a .mp4-named temp
     file and could fail to open. We now keep the original extension.
  2. The temp file was never flushed/closed before OpenCV tried to read
     it, which is a race condition on some filesystems (looked fine
     locally, failed on Streamlit Cloud's storage). Now flushed, closed,
     and always cleaned up in a `finally` block.
  3. No error handling around the frame loop at all, so any decode
     failure surfaced to the user as Streamlit's generic "unknown error
     occurred" with zero context. Now wrapped, with a plain-English
     message plus an expandable traceback for debugging.
  4. Running full YOLO pose estimation on every sampled video frame, with
     no memory cleanup and no cap on how many frames get processed, is a
     likely OOM/timeout cause on Streamlit Community Cloud's free-tier
     resource limits. Added a "Fast mode" (classification only, pose
     estimation skipped) that's now the video default, a max-frames-to-
     analyze cap, and periodic `gc.collect()`.
  5. requirements.txt now pins `opencv-python-headless` instead of
     `opencv-python` — the non-headless build needs GUI/OpenGL system
     libraries that Streamlit Cloud's base image doesn't have, which is
     one of the most common silent-crash causes for this exact stack.

UI (restructured for cohesion, per request):
  - Matplotlib swapped for Plotly throughout — interactive, and it's what
    was meant by "py plot".
  - Single persistent "vitals strip" header (status pill + live
    confidence sparkline + key metrics) always visible, so Live Monitor
    and Analytics no longer feel like two disconnected screens.
  - Everything reorganized into tabs: Live Monitor / Analytics /
    Incident Log / About, instead of one long stacked page.
  - New visual system: dark clinical-monitor palette + alarm colors
    modeled on the real IEC 60601-1-8 medical alarm convention
    (red = high priority, amber = medium priority, cyan = low priority /
    technical, green = normal), monospace readouts for numbers.
"""

import streamlit as st
import numpy as np
import cv2
import json
import os
import gc
import tempfile
import time
import io
import wave
import base64
import traceback
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# ==========================================================================
# CONFIG
# ==========================================================================
MODEL_PATH = "fall_detection_model.h5"
LABELS_PATH = "class_labels.json"
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"
IMG_SIZE = (224, 224)
FALL_CLASS = "fall"
VIDEO_FRAME_SKIP = 5
MAX_VIDEO_FRAMES_DEFAULT = 120       # cap analyzed frames to protect against cloud OOM/timeout
DEFAULT_ALERT_THRESHOLD = 0.60

# Alarm-standard palette (loosely modeled on IEC 60601-1-8 medical alarm
# priority colors: red / amber / cyan / green).
COLORS = {
    "bg":        "#0A0F1A",
    "surface":   "#121A2B",
    "surface2":  "#182338",
    "border":    "#223350",
    "text":      "#E8EDF7",
    "muted":     "#7E8BA6",
    "red":       "#FF3B3B",   # high-priority alarm (fall)
    "amber":     "#FFB020",   # medium-priority (below-threshold fall)
    "cyan":      "#22D3EE",   # low-priority / technical (info, pose)
    "green":     "#21C97A",   # normal / stable
    "blue":      "#4C8DFF",   # brand / neutral accent
}

ACTIVITY_COLORS = {
    "fall":     COLORS["red"],
    "walking":  COLORS["blue"],
    "sitting":  "#9D6DF9",
    "standing": COLORS["amber"],
    "normal":   COLORS["green"],
}
DEFAULT_COLOR = COLORS["muted"]


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
    page_title="SafeFall AI - Monitoring Console",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="expanded",
)

# ==========================================================================
# STYLING — clinical monitor console
# ==========================================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .stApp {{
        background: {COLORS['bg']};
        color: {COLORS['text']};
    }}
    section[data-testid="stSidebar"] {{
        background: {COLORS['surface']};
        border-right: 1px solid {COLORS['border']};
    }}
    .mono {{ font-family: 'IBM Plex Mono', monospace; }}

    /* ---- vitals strip ---- */
    .vitals-strip {{
        display: flex;
        align-items: stretch;
        gap: 0;
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 6px;
    }}
    .vitals-status {{
        min-width: 190px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: flex-start;
        padding: 16px 20px;
        gap: 4px;
    }}
    .status-pill {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.06em;
        padding: 3px 10px;
        border-radius: 5px;
        display: inline-block;
    }}
    .status-sub {{ color: {COLORS['muted']}; font-size: 0.72rem; margin-top: 2px; }}
    .vitals-metric {{
        flex: 1;
        padding: 14px 18px;
        border-left: 1px solid {COLORS['border']};
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .vitals-metric .label {{
        color: {COLORS['muted']};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 3px;
    }}
    .vitals-metric .value {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 1.35rem;
    }}

    @keyframes flash-red {{
        0%   {{ background-color: #7a0000; }}
        50%  {{ background-color: {COLORS['red']}; }}
        100% {{ background-color: #7a0000; }}
    }}
    .emergency-banner {{
        animation: flash-red 0.7s infinite;
        color: white;
        padding: 20px 22px;
        border-radius: 10px;
        text-align: left;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 10px 0 14px 0;
        border: 2px solid #ffffff33;
    }}
    .below-threshold-banner {{
        background: {COLORS['surface2']};
        border-left: 4px solid {COLORS['amber']};
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0 14px 0;
        color: {COLORS['text']};
    }}
    .ok-banner {{
        background: {COLORS['surface2']};
        border-left: 4px solid {COLORS['green']};
        padding: 14px 18px;
        border-radius: 6px;
        margin: 10px 0 14px 0;
        color: {COLORS['text']};
    }}
    .notify-card {{
        background: linear-gradient(135deg, {COLORS['surface2']}, {COLORS['surface']});
        border-left: 4px solid {COLORS['red']};
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 8px;
        font-size: 0.92rem;
        line-height: 1.6;
    }}
    .metric-card {{
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: {COLORS['surface']};
        border: 1px solid {COLORS['border']};
    }}
    .metric-card .label {{
        font-size: 0.72rem;
        color: {COLORS['muted']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .metric-card .value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2px;
    }}
    .summary-card {{
        background: {COLORS['surface']};
        border-left: 4px solid {COLORS['cyan']};
        padding: 16px 20px;
        border-radius: 8px;
        line-height: 1.75;
        border: 1px solid {COLORS['border']};
    }}
    .panel-title {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {COLORS['muted']};
        margin-bottom: 8px;
        margin-top: 4px;
    }}
    h1 {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        color: {COLORS['text']} !important;
    }}
    .app-caption {{ color: {COLORS['muted']}; margin-top: -8px; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {COLORS['surface']};
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLORS['surface2']};
        border-bottom: 2px solid {COLORS['cyan']};
    }}
    .stButton>button {{
        border-radius: 7px;
        border: 1px solid {COLORS['border']};
    }}
    .footnote {{ color: {COLORS['muted']}; font-size: 0.8rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label, value, color):
    st.markdown(
        f"""<div class="metric-card" style="border-color:{color}55;">
        <div class="label">{label}</div><div class="value" style="color:{color};">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS["surface"],
    plot_bgcolor=COLORS["surface"],
    font=dict(family="IBM Plex Mono, monospace", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ==========================================================================
# CACHED RESOURCES
# ==========================================================================
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


# ==========================================================================
# SESSION STATE
# ==========================================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "last_fall_time" not in st.session_state:
    st.session_state.last_fall_time = None


def normalize_entry(h):
    """Resilient to stale session state from an older app version (e.g. a
    browser tab still open from before a redeploy)."""
    if isinstance(h, dict):
        return h
    label, confidence = h
    return {"timestamp": "-", "activity": label, "confidence": confidence}


# ==========================================================================
# CORE INFERENCE
# ==========================================================================
def run_pose_estimation(frame_bgr, pose_model):
    results = pose_model.predict(source=frame_bgr, verbose=False)
    result = results[0]
    annotated = result.plot()
    pose_found = result.boxes is not None and len(result.boxes) > 0
    return annotated, pose_found


def classify_frame(frame_bgr, model):
    """Same preprocess_input as train_model.py, so inference matches
    training exactly (mismatched scaling silently tanks accuracy)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    return CLASSES[idx], float(preds[idx])


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
        st.markdown(
            f"""<div class="below-threshold-banner">
            ⚠️ <b>Possible fall</b> — {confidence:.1%} confidence, below your
            {threshold:.0%} alert threshold. Logged for review, no emergency
            alert raised.</div>""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""<div class="emergency-banner">🚨 EMERGENCY — FALL DETECTED
        ({confidence:.1%} confidence)</div>""",
        unsafe_allow_html=True,
    )
    play_alert_sound()
    st.markdown(
        f"""
        <div class="notify-card">
        <b>📟 Alert that would be sent</b> <span style="color:{COLORS['muted']}">
        (demo only — no real message is sent; wire this up to Twilio/SendGrid/etc.
        for production)</span><br><br>
        <b>To:</b> {caregiver_name or 'Caregiver'} ({caregiver_contact or 'no contact set'})<br>
        <b>Resident:</b> {resident_name or 'Unnamed resident'}<br>
        <b>Location:</b> {room or 'Unspecified room'}<br>
        <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <b>Message:</b> "Fall detected for {resident_name or 'resident'} in
        {room or 'monitored area'} at {confidence:.1%} confidence. Please check
        immediately."
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


# ==========================================================================
# VIDEO PIPELINE — hardened
# ==========================================================================
def process_video(uploaded_vid, pose_model, classifier, fast_mode, max_frames,
                   alert_threshold, resident_name, caregiver_name, caregiver_contact, room):
    """Runs the fall-detection pipeline over an uploaded video.

    Fixes vs. the previous version:
      - keeps the ORIGINAL file extension for the temp file (was hardcoded
        to .mp4, which breaks OpenCV's codec auto-detection for .avi/.mov
        uploads — the most likely cause of the video-only crash)
      - flushes + closes the temp file before OpenCV opens it
      - always cleans up the temp file, even on failure
      - checks cap.isOpened() and fails with a clear, actionable message
        instead of silently doing nothing / crashing generically
      - caps the number of analyzed frames and skips pose estimation in
        fast mode, to stay within Streamlit Cloud's memory/CPU limits
      - wraps everything so a real error message (with traceback in an
        expander) is shown instead of Streamlit's generic error screen
    """
    suffix = Path(uploaded_vid.name).suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_vid.read())
            tfile.flush()
            os.fsync(tfile.fileno())
            tmp_path = tfile.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error(
                "Couldn't open this video file. Try re-exporting it as a "
                "standard H.264 .mp4 — some .avi/.mov codecs aren't "
                "supported by the server's video backend."
            )
            return

        frame_placeholder = st.empty()
        alert_placeholder = st.empty()
        progress = st.progress(0)
        status_line = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        frame_idx = 0
        analyzed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % VIDEO_FRAME_SKIP == 0:
                if analyzed >= max_frames:
                    status_line.info(
                        f"Reached the {max_frames}-frame analysis cap for this "
                        f"session (keeps the app responsive on shared hosting). "
                        f"Raise it in the sidebar if you need the full video."
                    )
                    break

                if fast_mode:
                    display_frame = frame
                    pose_found = None
                else:
                    display_frame, pose_found = run_pose_estimation(frame, pose_model)

                label, confidence = classify_frame(frame, classifier)
                record_prediction(label, confidence)
                analyzed += 1

                cap_txt = f"Frame {frame_idx} — {label.capitalize()} ({confidence:.1%})"
                frame_placeholder.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    caption=cap_txt,
                    width='stretch',
                )
                with alert_placeholder.container():
                    if label == FALL_CLASS:
                        show_fall_alert(confidence, alert_threshold, resident_name,
                                         caregiver_name, caregiver_contact, room)
                    else:
                        st.markdown(
                            f'<div class="ok-banner">Monitoring — current activity: '
                            f'<b>{label.capitalize()}</b> ({confidence:.1%})</div>',
                            unsafe_allow_html=True,
                        )

                del frame
                if analyzed % 20 == 0:
                    gc.collect()

            if total_frames:
                progress.progress(min(frame_idx / total_frames, 1.0))
            time.sleep(0.01)

        cap.release()
        progress.progress(1.0)
        st.success(f"Video processing complete — analyzed {analyzed} frame(s).")

    except Exception as e:
        st.error(
            "Something went wrong while processing this video. Details below "
            "for debugging — this no longer fails silently."
        )
        with st.expander("Show error details"):
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def process_and_show_image(frame, pose_model, classifier, alert_threshold,
                            resident_name, caregiver_name, caregiver_contact, room):
    try:
        annotated, pose_found = run_pose_estimation(frame, pose_model)
        label, confidence = classify_frame(frame, classifier)
        record_prediction(label, confidence)

        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Pose Estimation Output",
            width='stretch',
        )

        if label == FALL_CLASS:
            show_fall_alert(confidence, alert_threshold, resident_name,
                             caregiver_name, caregiver_contact, room)
        else:
            st.markdown(
                f'<div class="ok-banner">✅ Activity: <b>{label.capitalize()}</b> '
                f'(confidence {confidence:.1%})</div>',
                unsafe_allow_html=True,
            )
        if not pose_found:
            st.warning("No body landmarks detected in this image — check lighting/angle.")
    except Exception as e:
        st.error("Something went wrong while processing this image.")
        with st.expander("Show error details"):
            st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)))


# ==========================================================================
# PLOTLY CHARTS
# ==========================================================================
def bar_chart(df_counts):
    fig = go.Figure(go.Bar(
        x=df_counts["Activity"], y=df_counts["Count"],
        marker_color=[color_for(a) for a in df_counts["Activity"]],
        text=df_counts["Count"], textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Activity distribution", height=340,
                       xaxis=dict(gridcolor=COLORS["border"]),
                       yaxis=dict(gridcolor=COLORS["border"]))
    return fig


def pie_chart(df_counts):
    fig = go.Figure(go.Pie(
        labels=df_counts["Activity"], values=df_counts["Count"],
        marker=dict(colors=[color_for(a) for a in df_counts["Activity"]]),
        hole=0.45, textinfo="percent+label",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Activity share", height=340,
                       showlegend=False)
    return fig


def timeline_chart(history, alert_threshold):
    indices = list(range(1, len(history) + 1))
    confidences = [h["confidence"] for h in history]
    colors = [color_for(h["activity"]) for h in history]
    activities = [h["activity"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=confidences, mode="lines", line=dict(color=COLORS["border"], width=1),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=indices, y=confidences, mode="markers",
        marker=dict(color=colors, size=9, line=dict(width=1, color=COLORS["bg"])),
        text=activities, hovertemplate="Prediction %{x}<br>%{text} — %{y:.1%}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=alert_threshold, line_dash="dash", line_color=COLORS["red"],
                  annotation_text=f"Alert threshold ({alert_threshold:.0%})",
                  annotation_font_color=COLORS["red"])
    fig.update_layout(**PLOTLY_LAYOUT, title="Confidence timeline", height=300,
                       xaxis=dict(title="Prediction #", gridcolor=COLORS["border"]),
                       yaxis=dict(title="Confidence", range=[0, 1.05], gridcolor=COLORS["border"]))
    return fig


def sparkline(history):
    """Small strip sparkline for the vitals header."""
    recent = history[-40:]
    if not recent:
        fig = go.Figure()
    else:
        y = [h["confidence"] for h in recent]
        colors = [color_for(h["activity"]) for h in recent]
        fig = go.Figure(go.Scatter(
            x=list(range(len(y))), y=y, mode="lines+markers",
            line=dict(color=COLORS["cyan"], width=1.5),
            marker=dict(color=colors, size=5),
        ))
    fig.update_layout(
        paper_bgcolor=COLORS["surface"], plot_bgcolor=COLORS["surface"],
        margin=dict(l=0, r=0, t=0, b=0), height=60,
        xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 1.05]),
        showlegend=False,
    )
    return fig


# ==========================================================================
# SIDEBAR
# ==========================================================================
st.sidebar.markdown(
    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.05rem;'
    f'font-weight:700;">🩺 SAFEFALL AI</div>'
    f'<div class="footnote">CareVision HealthTech Pvt. Ltd.<br>Elderly fall-detection console</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

with st.sidebar.expander("👤 Patient / caregiver", expanded=True):
    resident_name = st.text_input("Resident name", value="", placeholder="e.g. Mr. Sharma")
    room = st.text_input("Room / location", value="", placeholder="e.g. Room 4B")
    caregiver_name = st.text_input("Caregiver name", value="", placeholder="e.g. Nurse Priya")
    caregiver_contact = st.text_input("Caregiver phone/email", value="", placeholder="e.g. +91 98xxxxxxx")

with st.sidebar.expander("⚙️ Detection settings", expanded=True):
    alert_threshold = st.slider(
        "Alert confidence threshold", min_value=0.30, max_value=0.95,
        value=DEFAULT_ALERT_THRESHOLD, step=0.05,
        help="Lower = more sensitive (more alerts, more false alarms). "
             "Higher = fewer false alarms, but a real fall near the threshold "
             "might not trigger an emergency alert.",
    )
    fast_mode = st.checkbox(
        "Fast mode for video (skip pose overlay, classify only)", value=True,
        help="Recommended on shared/cloud hosting — pose estimation on every "
             "sampled frame is the main reason video uploads can time out or "
             "run out of memory.",
    )
    max_frames = st.slider(
        "Max frames to analyze per video", min_value=20, max_value=400,
        value=MAX_VIDEO_FRAMES_DEFAULT, step=20,
        help="Caps how many sampled frames get processed, to keep the app "
             "responsive on limited hosting.",
    )

st.sidebar.markdown("---")
mode = st.sidebar.radio("Input source", ["Upload Image", "Upload Video", "Webcam Snapshot"])

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset monitoring session", width='stretch'):
    st.session_state.history = []
    st.session_state.last_fall_time = None
    st.sidebar.success("Session analytics reset.")

# ==========================================================================
# MODEL LOADING
# ==========================================================================
classifier = None
pose_model = None
classifier_ok, pose_ok = False, False

try:
    classifier = load_classifier()
    classifier_ok = True
except Exception as e:
    st.sidebar.error(f"Classifier failed to load: {e}")

try:
    pose_model = load_pose_model()
    pose_ok = True
except Exception as e:
    st.sidebar.error(f"Pose model failed to load: {e}")

model_loaded = classifier_ok and pose_ok

# ==========================================================================
# HEADER + VITALS STRIP
# ==========================================================================
st.title("SafeFall AI — Monitoring Console")
st.markdown(
    '<div class="app-caption">Computer vision + deep learning fall detection for elderly care</div>',
    unsafe_allow_html=True,
)
st.write("")

history = [normalize_entry(h) for h in st.session_state.history]
total = len(history)
fall_count = sum(1 for h in history if h["activity"] == FALL_CLASS)
normal_count = total - fall_count
last_entry = history[-1] if history else None

if last_entry and last_entry["activity"] == FALL_CLASS and last_entry["confidence"] >= alert_threshold:
    status_label, status_color = "ALERT", COLORS["red"]
elif last_entry and last_entry["activity"] == FALL_CLASS:
    status_label, status_color = "REVIEW", COLORS["amber"]
elif last_entry:
    status_label, status_color = "STABLE", COLORS["green"]
else:
    status_label, status_color = "STANDBY", COLORS["cyan"]

if st.session_state.last_fall_time:
    elapsed = datetime.now() - st.session_state.last_fall_time
    mins = int(elapsed.total_seconds() // 60)
    time_since_fall = f"{mins} min ago" if mins > 0 else "just now"
else:
    time_since_fall = "no falls yet"

avg_conf = f"{np.mean([h['confidence'] for h in history]):.1%}" if history else "—"

vs1, vs2 = st.columns([1.1, 3])
with vs1:
    st.markdown(
        f"""<div class="vitals-strip" style="height:100%;">
        <div class="vitals-status">
            <span class="status-pill" style="background:{status_color}22;color:{status_color};
            border:1px solid {status_color};">{status_label}</span>
            <div class="status-sub">{"Model ready" if model_loaded else "Model not loaded"}</div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )
with vs2:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        metric_card("Total activities", total, COLORS["blue"])
    with m2:
        metric_card("Falls detected", fall_count, COLORS["red"])
    with m3:
        metric_card("Avg. confidence", avg_conf, COLORS["cyan"])
    with m4:
        metric_card("Time since last fall", time_since_fall, COLORS["amber"])

if history:
    st.plotly_chart(sparkline(history), config={"displayModeBar": False}, width='stretch')

st.write("")

# ==========================================================================
# TABS
# ==========================================================================
tab_live, tab_analytics, tab_log, tab_about = st.tabs(
    ["🎥 Live Monitor", "📊 Analytics", "📝 Incident Log", "ℹ️ About"]
)

# ---- LIVE MONITOR ----
with tab_live:
    if not model_loaded:
        st.info(
            "Dashboard preview only — the classifier and/or pose model didn't "
            "load, so predictions are disabled. Check the sidebar for the "
            "specific loading error."
        )
    else:
        if mode == "Upload Image":
            uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_img is not None:
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Couldn't decode this image — try a different file.")
                else:
                    process_and_show_image(frame, pose_model, classifier, alert_threshold,
                                            resident_name, caregiver_name, caregiver_contact, room)

        elif mode == "Webcam Snapshot":
            st.caption("Take a live snapshot from your device camera — useful for a quick spot-check.")
            snapshot = st.camera_input("Camera")
            if snapshot is not None:
                file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Couldn't decode this snapshot — try again.")
                else:
                    process_and_show_image(frame, pose_model, classifier, alert_threshold,
                                            resident_name, caregiver_name, caregiver_contact, room)

        elif mode == "Upload Video":
            uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
            if uploaded_vid is not None:
                with st.spinner("Analyzing video…"):
                    process_video(uploaded_vid, pose_model, classifier, fast_mode, max_frames,
                                  alert_threshold, resident_name, caregiver_name, caregiver_contact, room)

# ---- ANALYTICS ----
with tab_analytics:
    if not history:
        st.caption("No predictions yet — run something in Live Monitor to populate analytics.")
    else:
        st.markdown('<div class="panel-title">Session summary</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card">{build_session_summary(history)}</div>',
                     unsafe_allow_html=True)
        st.write("")

        counts = Counter(h["activity"] for h in history)
        df_counts = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(bar_chart(df_counts), width='stretch')
        with c2:
            st.plotly_chart(pie_chart(df_counts), width='stretch')

        st.plotly_chart(timeline_chart(history, alert_threshold), width='stretch')

# ---- INCIDENT LOG ----
with tab_log:
    if not history:
        st.caption("No predictions yet.")
    else:
        log_df = pd.DataFrame(history)
        log_df_display = log_df.copy()
        log_df_display["confidence"] = (log_df_display["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(log_df_display.iloc[::-1], width='stretch', height=360)

        csv_bytes = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download incident log (CSV)",
            data=csv_bytes,
            file_name=f"safefall_incident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ---- ABOUT ----
with tab_about:
    st.markdown('<div class="panel-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        "- **Pose estimation:** YOLOv8 Pose (Ultralytics)\n"
        "- **Activity classification:** MobileNetV2-based CNN, trained via `train_model.py`\n"
        f"- **Classes:** {', '.join(CLASSES)}\n"
        f"- **Alert threshold:** configurable, currently {alert_threshold:.0%}\n"
    )
    st.markdown('<div class="panel-title">Known limitations</div>', unsafe_allow_html=True)
    st.markdown(
        "Accuracy can degrade under poor lighting, unusual camera angles, "
        "occlusion, or postures resembling a fall (e.g. lying on a sofa). "
        "Periodic retraining with new footage is recommended. Alert "
        "notifications in this app are a UI demo only — no SMS/email "
        "actually sends; wire that up to a real provider for production."
    )
    st.markdown('<div class="panel-title">Deployment notes</div>', unsafe_allow_html=True)
    st.markdown(
        "- Requires `opencv-python-headless`, not `opencv-python`, on "
        "Streamlit Community Cloud (the non-headless build needs system "
        "GUI libraries the base image doesn't have).\n"
        "- Video analysis is capped (see sidebar) to stay within shared-"
        "hosting memory/CPU limits — raise the cap if you're self-hosting "
        "with more resources.\n"
        "- YOLOv8 pose weights auto-download from Ultralytics' GitHub "
        "Releases on first run and are cached afterwards."
    )
