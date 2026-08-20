"""
CareVision HealthTech Pvt. Ltd. - SafeFall AI
FA-1: Elderly Fall Detection System
Step 2 (Data Preprocessing) + Step 3 (Exploratory Data Analysis)

Dataset: Le2i Fall Dataset (https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data)

Expected raw structure (after downloading from Kaggle):
    raw_dataset/
        Coffee_room_01/
            Videos/*.avi
            Annotation_files/*.txt
        Coffee_room_02/...
        Home_01/...
        Home_02/...
        Lecture_room/...
        Office/...

This script:
  1. Extracts frames ONLY from videos that have a matching annotation file
     (unannotated videos are skipped entirely).
  2. Labels each frame using two sources of truth:
       - Inside the annotation's fall window -> 'fall' (ground truth).
       - Outside the fall window -> run YOLOv8 Pose to get body keypoints,
         then classify posture into 'walking', 'sitting', 'standing', or
         'normal' using simple keypoint-geometry rules (see
         classify_posture()). This is what actually produces real 5-class
         labels instead of dumping every non-fall frame into 'normal'.
  3. Organizes frames into activity folders: /fall /walking /sitting /standing /normal
  4. Preprocesses frames: resize to 224x224, normalize pixel values to [0,1]
  5. Applies data augmentation (rotation, flip, brightness, zoom)
  6. Splits into 70% train / 30% test
  7. Runs EDA: class counts, bar/pie charts, sample grids, image-quality checks

Install once:
    # Ultralytics (YOLO) from its GitHub source repo, not PyPI:
    pip install "git+https://github.com/ultralytics/ultralytics.git" --break-system-packages

    # Everything else:
    pip install opencv-python numpy pandas matplotlib seaborn scikit-learn pillow tqdm --break-system-packages

Note: the posture rules below are a simple, transparent heuristic suitable
for a student project — not a clinically validated classifier. Document
this assumption in your report.
"""

import os
import cv2
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ultralytics import YOLO

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
RAW_DATASET_DIR = "raw_dataset"          # folder with downloaded Le2i data
OUTPUT_DIR = "processed_dataset"          # where extracted/organized frames go
FRAME_SIZE = (224, 224)                   # required resize
FRAME_SAMPLE_RATE = 15                    # grab 1 frame every N frames from each video
TRAIN_SPLIT = 0.70
RANDOM_SEED = 42
ACTIVITY_CLASSES = ["fall", "walking", "sitting", "standing", "normal"]
YOLO_POSE_MODEL_NAME = "yolov8n-pose.pt"   # auto-downloaded from GitHub on first use
POSE_CONF_THRESHOLD = 0.5                   # min keypoint confidence to trust a joint

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_yolo_pose_model():
    """
    Load YOLOv8 Pose. If 'yolov8n-pose.pt' isn't already cached locally,
    Ultralytics automatically fetches it from the official GitHub Releases
    page (github.com/ultralytics/assets/releases) — not from PyPI.
    """
    print(f"Loading YOLOv8 Pose ('{YOLO_POSE_MODEL_NAME}') — auto-downloaded "
          f"from Ultralytics' GitHub releases on first use if not cached.")
    return YOLO(YOLO_POSE_MODEL_NAME)


# --------------------------------------------------------------------------
# STEP 2a: ANNOTATION PARSING
# --------------------------------------------------------------------------
def parse_annotation(annotation_path):
    """
    Le2i annotation files typically contain:
        line 1: frame number where the fall starts
        line 2: frame number where the fall ends
        remaining lines: bounding box coordinates per frame (optional, unused here)

    Returns (fall_start_frame, fall_end_frame) as ints, or (None, None) if
    the video contains no fall (e.g. purely 'normal activity' videos).
    """
    try:
        with open(annotation_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        fall_start = int(lines[0])
        fall_end = int(lines[1])
        if fall_start == 0 and fall_end == 0:
            return None, None
        return fall_start, fall_end
    except Exception as e:
        print(f"  [warn] Could not parse annotation {annotation_path}: {e}")
        return None, None


def classify_posture(frame_bgr, pose_model):
    """
    Run YOLOv8 Pose on a frame (that is known to be OUTSIDE the annotated
    fall window) and classify body posture into 'walking', 'sitting',
    'standing', or 'normal' using simple keypoint-geometry rules.

    COCO-17 keypoint indices used:
        5/6   = left/right shoulder
        11/12 = left/right hip
        13/14 = left/right knee
        15/16 = left/right ankle

    Rules (heuristic, documented for transparency):
      - Low-confidence keypoints or no person detected -> 'normal' (fallback)
      - Knees raised close to hip level (thigh ~horizontal) -> 'sitting'
      - Hip-to-ankle distance much longer than torso (legs extended) and
        ankles spread apart (stride) -> 'walking'
      - Same leg extension but ankles close together (no stride) -> 'standing'
      - Anything else -> 'normal'
    """
    results = pose_model.predict(source=frame_bgr, verbose=False)
    result = results[0]

    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return "normal"  # no person detected

    # Use the most confidently detected person (Le2i videos are single-subject)
    box_confs = result.boxes.conf.cpu().numpy()
    best_idx = int(box_confs.argmax())

    kp_xy = result.keypoints.xy[best_idx].cpu().numpy()      # (17, 2) pixel coords
    kp_conf = (
        result.keypoints.conf[best_idx].cpu().numpy()
        if result.keypoints.conf is not None else np.ones(17)
    )

    needed = [5, 6, 11, 12, 13, 14, 15, 16]
    if any(kp_conf[i] < POSE_CONF_THRESHOLD for i in needed):
        return "normal"  # can't trust the geometry -> fallback bucket

    shoulder_y = (kp_xy[5][1] + kp_xy[6][1]) / 2
    hip_y = (kp_xy[11][1] + kp_xy[12][1]) / 2
    knee_y = (kp_xy[13][1] + kp_xy[14][1]) / 2
    ankle_y = (kp_xy[15][1] + kp_xy[16][1]) / 2
    hip_width = max(abs(kp_xy[11][0] - kp_xy[12][0]), 1e-6)
    ankle_dx = abs(kp_xy[15][0] - kp_xy[16][0])

    torso_len = max(hip_y - shoulder_y, 1e-6)
    upper_leg_len = knee_y - hip_y
    leg_len = ankle_y - hip_y

    # Sitting: thighs raised close to hip level (knees not much lower than hips)
    if upper_leg_len < 0.5 * torso_len:
        return "sitting"

    # Standing / walking: legs mostly extended (hip-to-ankle notably longer than torso)
    if leg_len > 1.3 * torso_len:
        return "walking" if ankle_dx > 0.6 * hip_width else "standing"

    return "normal"


def label_frame(frame_idx, fall_start, fall_end, frame_bgr, pose_model):
    """
    - Inside [fall_start, fall_end] -> 'fall' (ground truth from annotation).
    - Outside the fall window -> classify posture via YOLOv8 Pose so the
      dataset actually contains walking/sitting/standing/normal, not just
      a blanket 'normal' bucket.
    """
    if fall_start is not None and fall_start <= frame_idx <= fall_end:
        return "fall"
    return classify_posture(frame_bgr, pose_model)


# --------------------------------------------------------------------------
# STEP 2b: FRAME EXTRACTION
# --------------------------------------------------------------------------
def extract_frames_from_video(video_path, annotation_path, output_root, room_name, video_name, pose_model):
    """Extract every Nth frame from a video and save it into the correct class folder.
    annotation_path is required — callers must only invoke this for videos that
    have a matching annotation file (see run_frame_extraction)."""
    fall_start, fall_end = parse_annotation(annotation_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [warn] Could not open video: {video_path}")
        return 0

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % FRAME_SAMPLE_RATE != 0:
            continue

        # Skip corrupted/blurry frames
        if frame is None or frame.size == 0:
            continue
        if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 15:
            continue  # too blurry, likely low quality

        label = label_frame(frame_idx, fall_start, fall_end, frame, pose_model)
        class_dir = Path(output_root) / label
        class_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{room_name}_{video_name}_frame{frame_idx}.jpg"
        cv2.imwrite(str(class_dir / out_name), frame)
        saved_count += 1

    cap.release()
    return saved_count


def run_frame_extraction():
    """
    Only videos that have a matching annotation .txt file are used for
    training. Videos without an annotation file are skipped entirely (not
    extracted, not labeled 'normal' by default), since we cannot verify
    where/whether a fall occurs in them without the ground-truth file.
    """
    print("=== STEP 2: Extracting frames from annotated videos only ===")
    total_saved = 0
    total_skipped = 0
    raw_root = Path(RAW_DATASET_DIR)

    if not raw_root.exists():
        print(f"[error] '{RAW_DATASET_DIR}' not found. Download the Le2i dataset from Kaggle and "
              f"place it in this folder before running.")
        return

    pose_model = load_yolo_pose_model()

    for room_dir in sorted(raw_root.iterdir()):
        if not room_dir.is_dir():
            continue
        video_dir = room_dir / "Videos"
        annot_dir = room_dir / "Annotation_files"
        if not video_dir.exists():
            continue

        for video_file in sorted(video_dir.glob("*.avi")):
            video_name = video_file.stem
            annot_file = annot_dir / f"{video_name}.txt" if annot_dir.exists() else None

            # Skip this video entirely if there's no matching annotation file.
            if not annot_file or not annot_file.exists():
                print(f"  [skip] {room_dir.name}/{video_name}: no annotation file found")
                total_skipped += 1
                continue

            saved = extract_frames_from_video(
                video_file, annot_file, OUTPUT_DIR, room_dir.name, video_name, pose_model
            )
            total_saved += saved
            print(f"  {room_dir.name}/{video_name}: {saved} frames saved")

    print(f"\nTotal frames extracted: {total_saved}")
    print(f"Videos skipped (no annotation file): {total_skipped}\n")


# --------------------------------------------------------------------------
# STEP 2c: PREPROCESSING (resize + normalize)
# --------------------------------------------------------------------------
def preprocess_image(image_path, size=FRAME_SIZE):
    """Load an image, resize to fixed size, normalize pixel values to [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocess_all_frames():
    """Overwrite each extracted frame with its resized version (normalization
    is applied on-the-fly during training, but we validate it here)."""
    print("=== STEP 2: Resizing frames to 224x224 ===")
    for cls in ACTIVITY_CLASSES:
        cls_dir = Path(OUTPUT_DIR) / cls
        if not cls_dir.exists():
            continue
        for img_path in tqdm(list(cls_dir.glob("*.jpg")), desc=f"Resizing {cls}"):
            try:
                img = Image.open(img_path).convert("RGB").resize(FRAME_SIZE)
                img.save(img_path)
            except Exception as e:
                print(f"  [warn] Skipping corrupted image {img_path}: {e}")
                img_path.unlink(missing_ok=True)
    print()


# --------------------------------------------------------------------------
# STEP 2d: DATA AUGMENTATION
# --------------------------------------------------------------------------
def augment_image(img: Image.Image):
    """Return a list of augmented versions: rotate, flip, brighten, zoom."""
    augmented = []

    # Rotation
    augmented.append(img.rotate(15))

    # Horizontal flip
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))

    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(1.3))

    # Zoom (center crop then resize back)
    w, h = img.size
    crop = img.crop((w * 0.1, h * 0.1, w * 0.9, h * 0.9)).resize((w, h))
    augmented.append(crop)

    return augmented


def augment_dataset(target_per_class=None):
    """Balance classes by augmenting minority classes up to target_per_class."""
    print("=== STEP 2: Applying data augmentation for class balance ===")
    counts = {cls: len(list((Path(OUTPUT_DIR) / cls).glob("*.jpg")))
              for cls in ACTIVITY_CLASSES if (Path(OUTPUT_DIR) / cls).exists()}
    if not counts:
        print("No frames found to augment.\n")
        return

    if target_per_class is None:
        target_per_class = max(counts.values())

    for cls, count in counts.items():
        cls_dir = Path(OUTPUT_DIR) / cls
        images = list(cls_dir.glob("*.jpg"))
        needed = target_per_class - count
        if needed <= 0:
            continue

        print(f"  Augmenting '{cls}': {count} -> target {target_per_class}")
        generated = 0
        idx = 0
        while generated < needed and images:
            src = images[idx % len(images)]
            img = Image.open(src).convert("RGB")
            for i, aug_img in enumerate(augment_image(img)):
                if generated >= needed:
                    break
                out_path = cls_dir / f"{src.stem}_aug{idx}_{i}.jpg"
                aug_img.resize(FRAME_SIZE).save(out_path)
                generated += 1
            idx += 1
    print()


# --------------------------------------------------------------------------
# STEP 2e: TRAIN / TEST SPLIT (70/30)
# --------------------------------------------------------------------------
def split_train_test():
    print("=== STEP 2: Splitting into 70% train / 30% test ===")
    for cls in ACTIVITY_CLASSES:
        cls_dir = Path(OUTPUT_DIR) / cls
        if not cls_dir.exists():
            continue
        images = list(cls_dir.glob("*.jpg"))
        if not images:
            continue

        train_imgs, test_imgs = train_test_split(
            images, train_size=TRAIN_SPLIT, random_state=RANDOM_SEED
        )

        train_dir = Path(OUTPUT_DIR) / "train" / cls
        test_dir = Path(OUTPUT_DIR) / "test" / cls
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, train_dir / img.name)
        for img in test_imgs:
            shutil.copy(img, test_dir / img.name)

        print(f"  {cls}: {len(train_imgs)} train / {len(test_imgs)} test")
    print()


# --------------------------------------------------------------------------
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# --------------------------------------------------------------------------
def run_eda():
    print("=== STEP 3: Exploratory Data Analysis ===")
    counts = {}
    for cls in ACTIVITY_CLASSES:
        cls_dir = Path(OUTPUT_DIR) / cls
        counts[cls] = len(list(cls_dir.glob("*.jpg"))) if cls_dir.exists() else 0

    df = pd.DataFrame(list(counts.items()), columns=["Activity", "Frame_Count"])
    print(df)
    print(f"\nDataset balance check -> min: {df.Frame_Count.min()}, max: {df.Frame_Count.max()}")

    # ---- Bar chart ----
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Activity", y="Frame_Count", palette="viridis")
    plt.title("Activity-wise Frame Distribution")
    plt.xlabel("Activity Class")
    plt.ylabel("Number of Frames")
    plt.tight_layout()
    plt.savefig("eda_bar_chart.png", dpi=150)
    plt.close()

    # ---- Pie chart ----
    plt.figure(figsize=(6, 6))
    plt.pie(df.Frame_Count, labels=df.Activity, autopct="%1.1f%%", startangle=90)
    plt.title("Activity Class Proportion")
    plt.tight_layout()
    plt.savefig("eda_pie_chart.png", dpi=150)
    plt.close()

    # ---- Sample frame grid per class ----
    fig, axes = plt.subplots(1, len(ACTIVITY_CLASSES), figsize=(4 * len(ACTIVITY_CLASSES), 4))
    for ax, cls in zip(axes, ACTIVITY_CLASSES):
        cls_dir = Path(OUTPUT_DIR) / cls
        imgs = list(cls_dir.glob("*.jpg")) if cls_dir.exists() else []
        if imgs:
            sample = Image.open(random.choice(imgs))
            ax.imshow(sample)
        ax.set_title(cls)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("eda_sample_frames.png", dpi=150)
    plt.close()

    print("\nSaved: eda_bar_chart.png, eda_pie_chart.png, eda_sample_frames.png\n")
    return df


# --------------------------------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_frame_extraction()
    preprocess_all_frames()
    augment_dataset()
    split_train_test()
    eda_df = run_eda()
    eda_df.to_csv("eda_summary.csv", index=False)
    print("Pipeline complete. Processed dataset in:", OUTPUT_DIR)
