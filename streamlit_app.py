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

Extra features added on top:
  - Live webcam snapshot input (st.camera_input), not just file upload
  - Adjustable alert confidence threshold (fewer false alarms vs.
    more sensitive detection — a real tradeoff caregivers would tune)
  - Audible alert tone on a confirmed fall (generated locally, no
    external audio file needed)
  - Full-width flashing red emergency banner, not just a small toast
  - Resident/caregiver profile fields — an alert "would notify" card
    shows exactly what a real SMS/email integration would send
    (clearly labeled as a demo — no message actually leaves the app)
  - Timestamped incident log for the session, downloadable as CSV
  - "Time since last fall" live metric
"""

import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import time
import io
import wave
import base64
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
MODEL_PATH = "fall_detection_model.h5"
LABELS_PATH = "class_labels.json"       # written by train_model.py
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"  # auto-downloaded from GitHub on first use
IMG_SIZE = (224, 224)
FALL_CLASS = "fall"
VIDEO_FRAME_SKIP = 5   # classify every Nth frame of an uploaded video
DEFAULT_ALERT_THRESHOLD = 0.60


def load_classes():
    """Load the exact class order train_model.py trained on. Falls back to
    the full 5-class list if class_labels.json is missing."""
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
# STYLING — flashing emergency banner
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
        color: white;
        padding: 22px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 14px;
        border: 2px solid #ffffff40;
    }
    .notify-card {
        background-color: #1e1e1e;
        border-left: 5px solid #d10000;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
    }
    </style>
    """,
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
    """
    Load YOLOv8 Pose. If the weight file isn't cached locally, Ultralytics
    downloads it automatically from its GitHub Releases page — not PyPI.
    """
    return YOLO(YOLO_POSE_MODEL_NAME)


@st.cache_data
def generate_beep_base64():
    """Generate a short alert tone locally (no external audio asset needed)
    and return it as a base64 WAV string for inline HTML playback."""
    framerate = 44100
    duration = 0.45
    freq = 880.0
    t = np.linspace(0, duration, int(framerate * duration), False)
    tone = np.sin(freq * t * 2 * np.pi) * np.linspace(1, 0.2, t.size)  # slight fade
    audio = (tone * (2 ** 15 - 1)).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def play_alert_sound():
    """Embed an autoplaying alert tone. Browsers may block autoplay until
    the user has interacted with the page at least once — this is a
    browser policy, not a bug in the app."""
    b64 = generate_beep_base64()
    st.markdown(
        f"""<audio autoplay>
        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
        </audio>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: timestamp, label, confidence
if "last_fall_time" not in st.session_state:
    st.session_state.last_fall_time = None


# --------------------------------------------------------------------------
# CORE INFERENCE FUNCTIONS
# --------------------------------------------------------------------------
def run_pose_estimation(frame_bgr, pose_model):
    """Run YOLOv8 Pose on a frame and return the annotated frame plus
    whether a person/pose was detected."""
    results = pose_model.predict(source=frame_bgr, verbose=False)
    result = results[0]
    annotated = result.plot()  # BGR numpy array with skeleton drawn
    pose_found = result.boxes is not None and len(result.boxes) > 0
    return annotated, pose_found


def classify_frame(frame_bgr, model):
    """Resize/normalize a frame and run the CNN classifier.
    Uses the SAME preprocess_input as train_model.py so inference matches
    training exactly (mismatched scaling silently tanks accuracy)."""
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
    """Render the flashing banner, play a tone, and show the 'would notify'
    demo card — only for falls at/above the configured confidence threshold."""
    if confidence < threshold:
        st.warning(
            f"⚠️ Possible fall detected (confidence {confidence:.1%}) — below your "
            f"alert threshold of {threshold:.0%}, so no emergency alert was raised. "
            f"Still logged for review."
        )
        return

    st.markdown(
        f'<div class="emergency-banner">🚨 EMERGENCY: FALL DETECTED — '
        f'{confidence:.1%} confidence 🚨</div>',
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
        {room or 'monitored area'} at {confidence:.1%} confidence. Please check
        immediately."
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# UI - SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🏥 SafeFall AI")
st.sidebar.markdown("**CareVision HealthTech Pvt. Ltd.**\n\nElderly Fall Detection Monitoring")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Resident / caregiver profile")
resident_name = st.sidebar.text_input("Resident name", value="", placeholder="e.g. Mr. Sharma")
room = st.sidebar.text_input("Room / location", value="", placeholder="e.g. Room 4B")
caregiver_name = st.sidebar.text_input("Caregiver name", value="", placeholder="e.g. Nurse Priya")
caregiver_contact = st.sidebar.text_input("Caregiver phone/email", value="", placeholder="e.g. +91 98xxxxxxx")

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
# UI - MAIN
# --------------------------------------------------------------------------
st.title("SafeFall AI — Real-Time Elderly Fall Detection Dashboard")
st.caption("Computer Vision + Deep Learning monitoring system for elderly safety")

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

col_main, col_stats = st.columns([2, 1])


def process_and_show_image(frame):
    """Shared pipeline for Upload Image and Webcam Snapshot modes."""
    annotated, pose_found = run_pose_estimation(frame, pose_model)
    label, confidence = classify_frame(frame, classifier)
    record_prediction(label, confidence)

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Pose Estimation Output",
        use_container_width=True,
    )

    if label == FALL_CLASS:
        show_fall_alert(confidence, alert_threshold, resident_name,
                         caregiver_name, caregiver_contact, room)
    else:
        st.success(f"✅ Activity: {label.capitalize()} (confidence {confidence:.1%})")

    if not pose_found:
        st.warning("No body landmarks detected in this image — check lighting/angle.")


# ---- LEFT: input + prediction ----
with col_main:
    if mode == "Upload Image" and model_loaded:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_show_image(frame)

    elif mode == "Webcam Snapshot" and model_loaded:
        st.caption("Take a live snapshot from your device camera — useful for a quick spot-check.")
        snapshot = st.camera_input("Camera")
        if snapshot is not None:
            file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            process_and_show_image(frame)

    elif mode == "Upload Video" and model_loaded:
        uploaded_vid = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())

            cap = cv2.VideoCapture(tfile.name)
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
                        use_container_width=True,
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

    if not model_loaded:
        st.info("Dashboard preview only — connect a trained model to enable live predictions.")

# ---- RIGHT: monitoring analytics ----
with col_stats:
    st.subheader("📊 Monitoring Analytics")
    history = st.session_state.history

    total = len(history)
    fall_count = sum(1 for h in history if h["activity"] == FALL_CLASS)
    normal_count = total - fall_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Activities", total)
    m2.metric("Falls Detected", fall_count)
    m3.metric("Normal Activity", normal_count)

    if st.session_state.last_fall_time:
        elapsed = datetime.now() - st.session_state.last_fall_time
        mins = int(elapsed.total_seconds() // 60)
        st.metric("⏱️ Time since last fall", f"{mins} min ago" if mins > 0 else "Just now")
    else:
        st.metric("⏱️ Time since last fall", "No falls yet")

    if history:
        avg_conf = np.mean([h["confidence"] for h in history])
        st.metric("Avg. Prediction Confidence", f"{avg_conf:.1%}")

        counts = Counter(h["activity"] for h in history)
        df_counts = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(df_counts["Activity"], df_counts["Count"], color=[
            "#e74c3c" if a == FALL_CLASS else "#3498db" for a in df_counts["Activity"]
        ])
        ax.set_title("Activity Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        st.markdown("---")
        st.subheader("📝 Incident log")
        log_df = pd.DataFrame(history)
        log_df["confidence"] = (log_df["confidence"] * 100).round(1).astype(str) + "%"
        st.dataframe(log_df.iloc[::-1], use_container_width=True, height=180)

        csv_bytes = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download incident log (CSV)",
            data=csv_bytes,
            file_name=f"safefall_incident_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.caption("No predictions yet — upload an image or video to begin monitoring.")

    st.markdown("---")
    st.caption(
        "⚠️ Known deployment limitations: accuracy can degrade under poor lighting, "
        "unusual camera angles, occlusion, or postures resembling a fall (e.g. lying "
        "on a sofa). Periodic retraining with new footage is recommended. Alert "
        "notifications above are a UI demo only — no SMS/email actually sends."
    )
