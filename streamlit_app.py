"""
CareVision HealthTech Pvt. Ltd. - SafeFall AI
FA-2, Step 7: Model Deployment using Streamlit

Pose estimation: YOLOv8 Pose (Ultralytics), installed directly from
Ultralytics' GitHub source repository (see requirements.txt: a
git+https://github.com/... line, not a PyPI package name). Pose weights
('yolov8n-pose.pt') are auto-downloaded from Ultralytics' GitHub Releases
on first run and cached locally.

Run locally:
    pip install "git+https://github.com/ultralytics/ultralytics.git" --break-system-packages
    streamlit run streamlit_app.py

Then deploy on Streamlit Cloud (https://streamlit.io) by pushing this repo
(with fall_detection_model.h5, class_labels.json, requirements.txt) to
GitHub and connecting it in Streamlit Cloud's "New app" flow.

Dashboard features (per FA-2 Step 7 requirements):
  - Upload an image OR a video
  - Runs YOLOv8 Pose estimation + the trained CNN activity classifier
  - Displays fall alerts, prediction confidence, and pose visualization
  - Shows running monitoring analytics: total activities, fall count,
    normal-activity count, and an activity distribution chart
"""

import streamlit as st
import numpy as np
import cv2
import json
import tempfile
import time
from pathlib import Path
from collections import Counter

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


def load_classes():
    """Load the exact class order train_model.py trained on. Falls back to
    the full 5-class list if class_labels.json is missing."""
    try:
        with open(LABELS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return ["fall", "walking", "sitting", "standing", "normal"]


CLASSES = load_classes()

st.set_page_config(page_title="SafeFall AI - Elderly Monitoring Dashboard", layout="wide")


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


# --------------------------------------------------------------------------
# SESSION STATE FOR RUNNING ANALYTICS
# --------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of (label, confidence) tuples


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
    st.session_state.history.append((label, confidence))


# --------------------------------------------------------------------------
# UI - SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.title("🏥 SafeFall AI")
st.sidebar.markdown("**CareVision HealthTech Pvt. Ltd.**\n\nElderly Fall Detection Monitoring")
mode = st.sidebar.radio("Input type", ["Upload Image", "Upload Video"])
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset monitoring session"):
    st.session_state.history = []
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

# ---- LEFT: input + prediction ----
with col_main:
    if mode == "Upload Image" and model_loaded:
        uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_img is not None:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            annotated, pose_found = run_pose_estimation(frame, pose_model)
            label, confidence = classify_frame(frame, classifier)
            record_prediction(label, confidence)

            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                      caption="Pose Estimation Output", use_column_width=True)

            if label == FALL_CLASS:
                st.error(f"🚨 EMERGENCY ALERT: Fall Detected! (confidence {confidence:.1%})")
            else:
                st.success(f"✅ Activity: {label.capitalize()} (confidence {confidence:.1%})")

            if not pose_found:
                st.warning("No body landmarks detected in this image — check lighting/angle.")

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
                        use_column_width=True,
                    )
                    if label == FALL_CLASS:
                        alert_placeholder.error(f"🚨 EMERGENCY ALERT: Fall Detected at frame {frame_idx}!")
                    else:
                        alert_placeholder.info(f"Monitoring... current activity: {label.capitalize()}")

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
    fall_count = sum(1 for label, _ in history if label == FALL_CLASS)
    normal_count = sum(1 for label, _ in history if label != FALL_CLASS)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Activities", total)
    m2.metric("Falls Detected", fall_count)
    m3.metric("Normal Activity", normal_count)

    if history:
        avg_conf = np.mean([c for _, c in history])
        st.metric("Avg. Prediction Confidence", f"{avg_conf:.1%}")

        counts = Counter(label for label, _ in history)
        df = pd.DataFrame({"Activity": list(counts.keys()), "Count": list(counts.values())})

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(df["Activity"], df["Count"], color=[
            "#e74c3c" if a == FALL_CLASS else "#3498db" for a in df["Activity"]
        ])
        ax.set_title("Activity Distribution")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        st.caption("No predictions yet — upload an image or video to begin monitoring.")

    st.markdown("---")
    st.caption(
        "⚠️ Known deployment limitations: accuracy can degrade under poor lighting, "
        "unusual camera angles, occlusion, or postures resembling a fall (e.g. lying "
        "on a sofa). Periodic retraining with new footage is recommended."
    )
