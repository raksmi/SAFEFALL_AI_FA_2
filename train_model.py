"""
CareVision HealthTech Pvt. Ltd. - SafeFall AI
FA-2: Building and Deploying the Model
Step 4 (Model Selection) + Step 5 (Training) + Step 6 (Evaluation)

Model choice (as recommended in the brief):
  - Pose Estimation : YOLOv8 Pose (Ultralytics), installed directly from
    Ultralytics' GitHub source repository rather than PyPI.
  - Activity Classifier : CNN (MobileNetV2 transfer learning) on top of the
    preprocessed 224x224 frames produced in FA-1.

Pipeline:
  1. Load images from processed_dataset/train and processed_dataset/test
     (produced by the FA-1 script) and further split train -> train/val (85/15
     of the 70% train chunk), giving an overall ~70/15/15 split.
  2. Build a CNN (MobileNetV2 backbone + custom head). Classes are
     auto-detected from processed_dataset/train — FA-1 now labels fall
     frames from the ground-truth annotation window and non-fall frames
     via YOLOv8 Pose posture rules, so all five classes (fall, walking,
     sitting, standing, normal) should be present if the source videos
     cover those activities.
  3. Train, plot accuracy/loss curves.
  4. Evaluate on the test set: accuracy, precision, recall, F1-score,
     confusion matrix.
  5. Save the trained model (fall_detection_model.h5) for use in the
     Streamlit app.
  6. Run YOLOv8 Pose on a few sample frames to save pose-landmark
     visualization screenshots (Step 6 evidence requirement).

Install once:
    # 1) Install Ultralytics (YOLO) directly from its GitHub source repo,
    #    not from PyPI:
    pip install "git+https://github.com/ultralytics/ultralytics.git" --break-system-packages

    # 2) Everything else:
    pip install tensorflow opencv-python scikit-learn matplotlib seaborn pillow --break-system-packages

Note on weights: the first time YOLO("yolov8n-pose.pt") runs, Ultralytics
automatically downloads the pose weight file from its official GitHub
Releases page (github.com/ultralytics/assets/releases) and caches it
locally — the weights come from GitHub, not from a PyPI package payload.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
DATA_DIR = "processed_dataset"          # output of the FA-1 script
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4
MODEL_OUT = "fall_detection_model.h5"
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"   # auto-downloaded from GitHub on first use


def load_yolo_pose_model():
    """
    Load YOLOv8 Pose. If 'yolov8n-pose.pt' isn't already cached locally,
    Ultralytics automatically fetches it from the official GitHub Releases
    page (github.com/ultralytics/assets/releases) — not from PyPI.
    """
    print(f"Loading YOLOv8 Pose ('{YOLO_POSE_MODEL_NAME}') — auto-downloaded "
          f"from Ultralytics' GitHub releases on first use if not cached.")
    return YOLO(YOLO_POSE_MODEL_NAME)

# NOTE: FA-1 now only extracts frames from ANNOTATED videos, and its
# labeling heuristic only ever produces two folders: 'fall' and 'normal'
# (walking/sitting/standing would need pose-based sub-labels the Le2i
# annotation files don't provide). So classes are auto-detected from
# whatever subfolders actually exist under processed_dataset/train,
# instead of being hardcoded to a fixed 5-class list that could be missing
# folders and crash flow_from_directory.


def detect_classes(train_dir: Path):
    """Return sorted list of class subfolders that actually contain images."""
    classes = sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and any(d.glob("*.jpg"))
    ])
    if not classes:
        raise FileNotFoundError(
            f"No class folders with images found under '{train_dir}'. "
            f"Run fall_detection_fa1.py first."
        )
    return classes


# --------------------------------------------------------------------------
# STEP 5a: DATA GENERATORS (70% train / 15% val / 15% test)
# --------------------------------------------------------------------------
def build_generators():
    """
    processed_dataset/train/<class>/*.jpg  -> 70% of all data (from FA-1 split)
    processed_dataset/test/<class>/*.jpg   -> 30% of all data (from FA-1 split)

    We split the FA-1 'train' folder further: ~17.65% held out as validation,
    giving an overall ~70% train / 15% val / 15% test split.
    """
    train_dir = Path(DATA_DIR) / "train"
    test_dir = Path(DATA_DIR) / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected '{train_dir}' and '{test_dir}' from the FA-1 preprocessing "
            f"script. Run fall_detection_fa1.py first."
        )

    classes = detect_classes(train_dir)
    print(f"Detected classes: {classes}")

    # NOTE: MobileNetV2's frozen ImageNet weights were trained on inputs
    # scaled to [-1, 1] via preprocess_input, NOT plain 0-1 rescaling. We
    # use the matching preprocessing function here (and identically at
    # inference time in streamlit_app.py) so the frozen backbone actually
    # sees the input distribution it was trained on.
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1765,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=classes, subset="training", shuffle=True,
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=classes, subset="validation", shuffle=False,
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", classes=classes, shuffle=False,
    )

    print(f"Train samples: {train_gen.samples} | Val samples: {val_gen.samples} | Test samples: {test_gen.samples}")
    return train_gen, val_gen, test_gen, classes


# --------------------------------------------------------------------------
# STEP 5b: MODEL ARCHITECTURE (MobileNetV2 transfer learning CNN)
# --------------------------------------------------------------------------
def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone for initial training

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# STEP 5c: TRAINING
# --------------------------------------------------------------------------
def train_model():
    train_gen, val_gen, test_gen, classes = build_generators()
    model = build_model(num_classes=len(classes))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_OUT, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_training_curves(history)
    evaluate_model(model, test_gen, classes)
    return model, classes


# --------------------------------------------------------------------------
# STEP 6a: ACCURACY / LOSS GRAPHS
# --------------------------------------------------------------------------
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title("Accuracy Graph")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss Graph")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()
    print("Saved: training_curves.png")


# --------------------------------------------------------------------------
# STEP 6b: EVALUATION (accuracy, precision, recall, F1, confusion matrix)
# --------------------------------------------------------------------------
def evaluate_model(model, test_gen, classes):
    print("\n=== STEP 6: Model Evaluation on Test Set ===")
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix - Fall Detection Activity Classifier")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    with open("evaluation_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    print("Saved: evaluation_metrics.txt")


# --------------------------------------------------------------------------
# STEP 4/6: YOLOv8 POSE VISUALIZATION (screenshots for evidence)
# --------------------------------------------------------------------------
def save_pose_estimation_samples(classes, num_samples=5):
    """Run YOLOv8 Pose on a few sample test frames and save annotated
    images, to satisfy the 'pose detection output screenshots' requirement."""
    print("\n=== Generating pose estimation sample screenshots (YOLOv8 Pose) ===")
    pose_model = load_yolo_pose_model()

    out_dir = Path("pose_estimation_samples")
    out_dir.mkdir(exist_ok=True)

    test_dir = Path(DATA_DIR) / "test"
    sample_paths = []
    for cls in classes:
        cls_dir = test_dir / cls
        if cls_dir.exists():
            imgs = list(cls_dir.glob("*.jpg"))[:1]
            sample_paths.extend(imgs)
        if len(sample_paths) >= num_samples:
            break

    for img_path in sample_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        results = pose_model.predict(source=image, verbose=False)
        annotated = results[0].plot()  # BGR numpy array with skeleton drawn
        out_path = out_dir / f"pose_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved {out_path}")

    print(f"Pose estimation samples saved in '{out_dir}/'\n")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    trained_model, detected_classes = train_model()
    save_pose_estimation_samples(detected_classes)

    with open("class_labels.json", "w") as f:
        json.dump(detected_classes, f)

    print(f"\nPipeline complete. Trained model saved as '{MODEL_OUT}'.")
    print(f"Classes trained on: {detected_classes}")
    print("Saved 'class_labels.json' — streamlit_app.py loads this automatically (Step 7).")
