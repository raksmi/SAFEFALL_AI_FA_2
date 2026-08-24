"""
CareVision HealthTech Pvt. Ltd. - SafeFall AI
FA-2, Step 7: Model Deployment using Streamlit

============================================================================
STABILITY NOTES (read before touching video handling — hard-won lessons)
============================================================================
This app crashed repeatedly during development. Every one of the causes
below is now handled. If video processing ever breaks again, check these
first, in this order:

  1. DUPLICATE OPENCV INSTALLS. Only ever depend on ONE of opencv-python /
     opencv-python-headless. Ultralytics pulls in opencv-python itself —
     do not also add opencv-python-headless to requirements.txt. Having
     both installs two different native builds into the same site-packages
     'cv2' folder and corrupts it.

  2. UNALIGNED NUMPY ARRAYS INTO TENSORFLOW. TensorFlow's ops can hard-crash
     natively with "Check failed: IsAligned()" when given a numpy array
     whose memory layout (from a chain of cv2 operations — cvtColor,
     resize, preprocess_input, expand_dims) isn't aligned the way its
     Eigen-based kernels expect. This is a documented TensorFlow issue, not
     a logic bug. Fix: np.ascontiguousarray() on the final array right
     before model.predict() (see classify_frame()) forces a fresh, safely
     aligned buffer every time.

  3. UNSTABLE ULTRALYTICS VERSION. requirements.txt must point at a tagged
     GitHub RELEASE (e.g. .../archive/refs/tags/vX.Y.Z.zip), never
     .../heads/main.zip. main is untested nightly code.

  4. STALE/INCOMPATIBLE COMMITTED WEIGHTS. Do not commit yolov8n-pose.pt
     to the repo. Ultralytics checks the working directory for a file
     with that name BEFORE downloading — an old/mismatched local copy
     silently shadows the correct auto-downloaded one and can crash on
     load if it was saved by a different Ultralytics/PyTorch version.

  5. MEMORY PRESSURE. Streamlit Community Cloud's free tier has ~1GB RAM.
     TensorFlow + PyTorch/Ultralytics loaded together already use a large
     share of that. Video mode must: sample sparsely (VIDEO_FRAME_SKIP),
     cap total frames processed (MAX_FRAMES_PER_VIDEO), downscale large
     frames (downscale_if_large), reject oversized uploads
     (MAX_VIDEO_MB), and explicitly free per-frame arrays + gc.collect()
     periodically. Under real memory pressure, native libraries can
     corrupt their own heap before the OS kills the process — this shows
     up as different-looking crashes each time, which is exactly what
     happened here before these limits existed.

  6. WRONG TEMP FILE EXTENSION. Always write an uploaded video to a temp
     file using ITS OWN extension (Path(uploaded.name).suffix), never a
     hardcoded one — a mismatched container/extension can make FFmpeg
     misdetect the format.

None of these are hypothetical — each one caused a real crash in this
project before being fixed. Do not remove any of the mitigations below
without a specific reason.
============================================================================

Pose estimation: YOLOv8 Pose (Ultralytics), installed directly from a
tagged Ultralytics GitHub release (see requirements.txt). Pose weights
('yolov8n-pose.pt') auto-download from Ultralytics' GitHub Releases on
first run — do NOT commit this file to the repo (see note 4 above).

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy on Streamlit Cloud by pushing this repo (with
fall_detection_model.h5, class_labels.json, requirements.txt,
packages.txt — NOT yolov8n-pose.pt, NOT raw_dataset/, NOT
processed_dataset/) to GitHub and connecting it via "New app".

Core dashboard features (FA-2 Step 7 requirements):
  - Upload an image OR a video
  - Runs YOLOv8 Pose estimation + the trained CNN activity classifier
  - Displays fall alerts, prediction confidence, and pose visualization
  - Shows running monitoring analytics

Extra features:
  - Tabbed dashboard (Live Monitor / Analytics / Incident Log) + hero
    status header
  - Interactive Plotly charts (bar, donut, confidence timeline)
  - Live webcam snapshot input
  - Adjustable alert confidence threshold
  - Audible alert tone on a confirmed fall (generated locally)
  - Resident/caregiver profile fields feeding a "would notify" demo card
  - Timestamped incident log, downloadable as CSV
  - "Time since last fall" live metric
  - Auto-generated plain-English session summary
  - Defensive handling of legacy/stale session-state entries
"""

import gc
import io
import os
import base64
import json
import tempfile
import time
import wave
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "fall_detection_model.h5"
LABELS_PATH = "class_labels.json"
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"
IMG_SIZE = (224, 224)
FALL_CLASS = "fall"
DEFAULT_ALERT_THRESHOLD = 0.60

# --- Resource limits (see stability note 5 above) ---
VIDEO_FRAME_SKIP = 15          # only sample every Nth frame
MAX_FRAMES_PER_VIDEO = 60      # hard cap regardless of video length
MAX_FRAME_DIMENSION = 640      # downscale any frame larger than this
POSE_INFERENCE_SIZE = 480      # YOLO internal inference resolution
MAX_VIDEO_MB = 150             # reject uploads larger than this outright

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

# ============================================================================
# STYLING
# ============================================================================
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


# ============================================================================
# CACHED RESOURCES
# ============================================================================
@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_pose_model():
    """See stability note 4: never let a locally-committed
    yolov8n-pose.pt shadow this — always let it auto-download."""
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


# ============================================================================
# SESSION STATE
# ============================================================================
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


# ============================================================================
# VIDEO SAFETY HELPERS
# ============================================================================
def downscale_if_large(frame_bgr, max_dim=MAX_FRAME_DIMENSION):
    """Shrink oversized frames before either model sees them — smaller
    arrays mean less peak memory per frame (stability note 5)."""
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame_bgr
    scale = max_dim / longest
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))


def cleanup_temp_files(*paths):
    for p in set(paths):
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


# ============================================================================
# CORE INFERENCE
# ============================================================================
def run_pose_estimation(frame_bgr, pose_model):
    results = pose_model.predict(source=frame_bgr, verbose=False, imgsz=POSE_INFERENCE_SIZE)
    result = results[0]
    annotated = result.plot()
    pose_found = result.boxes is not None and len(result.boxes) > 0
    del results, result
    return annotated, pose_found


def classify_frame(frame_bgr, model):
    """Same preprocess_input as train_model.py so inference matches
    training exactly (mismatched scaling silently tanks accuracy)."""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    # Force a clean, contiguous, aligned copy. Arrays coming out of a
    # chain of cv2 operations can end up with a memory layout TensorFlow's
    # ops reject outright with "Check failed: IsAligned()" — a documented
    # TF issue, not a bug in this pipeline's logic. np.ascontiguousarray
    # guarantees a fresh, properly-aligned buffer every time.
    img = np.ascontiguousarray(img)

    preds = model.predict(img, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = CLASSES[idx]
    confidence = float(preds[idx])
    del img, preds
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


# ============================================================================
# SIDEBAR
# ============================================================================
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

# ============================================================================
# HERO HEADER
# ============================================================================
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
    frame = downscale_if_large(frame)
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

    del annotated, frame


# ----------------------------------------------------------------------
# TAB 1: LIVE MONITOR
# ----------------------------------------------------------------------
with tab_monitor:
    if not model_loaded:
        st.info("Dashboard preview only — connect a trained model to enable live predictions.")

    elif mode == "Upload Image":
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Couldn't read this image file — it may be corrupted. Try a different file.")
                else:
                    process_and_show_image(frame)
            except Exception as e:
                st.error(f"Something went wrong processing this image: {e}")

    elif mode == "Webcam Snapshot":
        st.caption("Take a live snapshot from your device camera — useful for a quick spot-check.")
        snapshot = st.camera_input("Camera")
        if snapshot is not None:
            try:
                file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error("Couldn't read the camera snapshot. Try again.")
                else:
                    process_and_show_image(frame)
            except Exception as e:
                st.error(f"Something went wrong processing this snapshot: {e}")

    elif mode == "Upload Video":
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_vid is not None:
            size_mb = uploaded_vid.size / (1024 * 1024)
            if size_mb > MAX_VIDEO_MB:
                st.error(
                    f"This video is {size_mb:.0f}MB — larger than the {MAX_VIDEO_MB}MB "
                    f"limit for this server. Please upload a shorter clip or a lower-"
                    f"resolution export."
                )
            else:
                sample_every = st.slider(
                    "Analyse every Nth frame", min_value=5, max_value=30,
                    value=VIDEO_FRAME_SKIP, step=5,
                    help="Higher = fewer frames analyzed = faster and lighter on memory, "
                         "but you might miss a brief fall between samples.",
                )
                run_btn = st.button("▶️ Run analysis", type="primary")

                # IMPORTANT: gated behind a button, not automatic on upload.
                # Streamlit reruns the ENTIRE script on any widget interaction
                # anywhere in the app — without this button, moving the alert
                # threshold slider (or any other widget) would silently
                # re-process the whole video from scratch every single time,
                # piling extra memory/compute pressure on top of whatever
                # else is running. Processing only on an explicit click is
                # both the expected UX and meaningfully safer here.
                if not run_btn:
                    st.caption("Video loaded — click **Run analysis** to process it.")
                else:
                    # Stability note 6: keep the ORIGINAL extension, never hardcode one.
                    original_suffix = Path(uploaded_vid.name).suffix.lower() or ".mp4"
                    temp_path = None
                    try:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix)
                        tfile.write(uploaded_vid.read())
                        tfile.close()
                        temp_path = tfile.name

                        cap = cv2.VideoCapture(temp_path)
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
                            stats_placeholder = st.empty()

                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                            frame_idx = 0
                            processed_count = 0

                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_idx += 1

                                if frame_idx % sample_every == 0:
                                    if processed_count >= MAX_FRAMES_PER_VIDEO:
                                        stats_placeholder.info(
                                            f"Stopped after {MAX_FRAMES_PER_VIDEO} sampled "
                                            f"frames to keep memory use safe on this server — "
                                            f"analytics below reflect everything processed so far."
                                        )
                                        break

                                    frame = downscale_if_large(frame)
                                    annotated, pose_found = run_pose_estimation(frame, pose_model)
                                    label, confidence = classify_frame(frame, classifier)
                                    record_prediction(label, confidence)
                                    processed_count += 1

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

                                    # Stability note 5: free large per-frame arrays promptly.
                                    del annotated
                                    if processed_count % 10 == 0:
                                        gc.collect()

                                del frame
                                progress.progress(min(frame_idx / total_frames, 1.0))
                                time.sleep(0.02)

                            cap.release()
                            gc.collect()
                            st.success(f"Video processing complete — {processed_count} frame(s) analyzed.")

                    except Exception as e:
                        st.error(
                            f"Something went wrong processing this video: {e}. "
                            f"If this keeps happening with this specific file, try "
                            f"re-exporting it as a standard H.264 .mp4."
                        )
                    finally:
                        cleanup_temp_files(temp_path)

# ----------------------------------------------------------------------
# TAB 2: ANALYTICS
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# TAB 3: INCIDENT LOG
# ----------------------------------------------------------------------
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
