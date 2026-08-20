# SafeFall AI — FA-1 + FA-2 (YOLOv8-Pose + CNN, 5-class)

CareVision HealthTech Pvt. Ltd. — Elderly Fall Detection System
Pipeline: **YOLOv8 Pose** (posture keypoints) + **MobileNetV2 CNN** (activity
classifier) → **Streamlit** dashboard.

Classes: `fall`, `walking`, `sitting`, `standing`, `normal`

---

## 0. What each file does

| File | Assignment step | What it does |
|---|---|---|
| `fall_detection_fa1.py` | FA-1, Steps 2 & 3 | Extracts frames from Le2i videos, labels them (fall = annotation window, others = YOLO-pose posture rules), resizes to 224×224, augments, splits 70/30 train/test, runs EDA (bar/pie charts, sample grid). |
| `train_model.py` | FA-2, Steps 4–6 | Loads `processed_dataset/`, builds a MobileNetV2-based CNN, trains it (70/15/15 via train/val/test), plots accuracy/loss curves, evaluates (accuracy/precision/recall/F1/confusion matrix), saves pose-estimation sample screenshots. |
| `streamlit_app.py` | FA-2, Step 7 | Deploys the trained model as an interactive dashboard: upload image/video, live pose overlay, fall alerts, monitoring analytics. |
| `requirements.txt` | — | Everything you need to `pip install`. |

---

## 1. Prerequisites

- **Python 3.10 or 3.11** (recommended — best TensorFlow/Ultralytics compatibility on Windows). Check with:
  ```
  python --version
  ```
- An internet connection the first time you run anything (YOLOv8 pose
  weights auto-download from Ultralytics' GitHub releases; MobileNetV2
  ImageNet weights auto-download from Keras).
- The **Le2i Fall Dataset** downloaded from Kaggle:
  https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia/data

---

## 2. Set up the project folder (Command Prompt)

Open **Command Prompt** and run:

```bat
mkdir SafeFallAI
cd SafeFallAI
python -m venv venv
venv\Scripts\activate
```

Copy `fall_detection_fa1.py`, `train_model.py`, `streamlit_app.py`, and
`requirements.txt` into this `SafeFallAI` folder, then:

```bat
pip install --upgrade pip
pip install -r requirements.txt
```

This step takes a while the first time (TensorFlow + Ultralytics are large).

---

## 3. Get the dataset into place

1. Download the Le2i dataset zip from Kaggle (link above).
2. Extract it so you end up with this structure **inside** `SafeFallAI`:

```
SafeFallAI\
    raw_dataset\
        Coffee_room_01\
            Videos\            *.avi
            Annotation_files\  *.txt
        Coffee_room_02\...
        Home_01\...
        Home_02\...
        Lecture_room\...
        Office\...
```

   If Kaggle gives you a different top-level folder name, just rename it
   to `raw_dataset`, or edit `RAW_DATASET_DIR` at the top of
   `fall_detection_fa1.py`.

---

## 4. Run FA-1: preprocessing + EDA

```bat
python fall_detection_fa1.py
```

What happens (in order — this can take a while depending on how many
videos you have; YOLO pose runs on every non-fall frame to sub-classify
posture):

1. Extracts frames only from videos that have a matching annotation file.
2. Labels each frame `fall` / `walking` / `sitting` / `standing` / `normal`.
3. Resizes everything to 224×224 and validates it.
4. Augments minority classes so counts are roughly balanced.
5. Splits into `processed_dataset\train\<class>\` and `processed_dataset\test\<class>\` (70/30).
6. Saves EDA evidence to the current folder: `eda_bar_chart.png`,
   `eda_pie_chart.png`, `eda_sample_frames.png`, `eda_summary.csv`.

**Evidence to screenshot for your FA-1 storyboard:** the three `eda_*.png`
files, plus a few sample images from `processed_dataset\train\fall\` etc.

If it's too slow on your machine, lower the amount of data used: reduce
`FRAME_SAMPLE_RATE` upward (fewer frames) or point `RAW_DATASET_DIR` at a
subfolder with only ~20–25 videos (the brief's suggested minimum).

---

## 5. Run FA-2: train + evaluate the model

```bat
python train_model.py
```

What happens:

1. Loads `processed_dataset\train` / `test`, further splits train into train/val.
2. Builds MobileNetV2 (frozen backbone) + a small classification head.
3. Trains up to 25 epochs (early stopping on validation loss).
4. Saves `training_curves.png` (accuracy/loss graphs — required evidence).
5. Evaluates on the held-out test set → prints accuracy/precision/recall/F1,
   saves `confusion_matrix.png` and `evaluation_metrics.txt`.
6. Runs YOLOv8 Pose on a few sample test images and saves annotated
   screenshots to `pose_estimation_samples\` (required evidence).
7. Saves `fall_detection_model.h5` and `class_labels.json` — these are what
   `streamlit_app.py` loads.

**Evidence to screenshot for your FA-2 report:** `training_curves.png`,
`confusion_matrix.png`, `evaluation_metrics.txt`, and the images in
`pose_estimation_samples\`.

---

## 6. Run FA-2: deploy the dashboard locally

Make sure `fall_detection_model.h5` and `class_labels.json` (produced in
step 5) are in the same folder as `streamlit_app.py`, then:

```bat
streamlit run streamlit_app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).
Upload a test image or video to see pose overlay, activity prediction,
fall alerts, and the live monitoring analytics panel.

**Evidence to screenshot/record for FA-2 submission:** the dashboard with
an uploaded image/video showing a fall alert, and the analytics panel.

---

## 7. Deploy for real (Streamlit Cloud — required for the FA-2 "link" evidence)

1. Push this whole folder (including `fall_detection_model.h5`,
   `class_labels.json`, `requirements.txt`) to a GitHub repo.
2. Go to https://streamlit.io → Streamlit Cloud → "New app".
3. Point it at your repo and `streamlit_app.py`.
4. Copy the live URL — that's your FA-2 "Link to deployed project" evidence.

> `.h5` model files aren't huge for this project, so committing it directly
> to GitHub is fine. If it ever exceeds GitHub's 100MB limit, use Git LFS.

---

## Troubleshooting

- **`ultralytics` fails to install** — make sure you're using the `pip
  install -r requirements.txt` from an activated venv; the
  `git+https://github.com/...` line needs `git` installed and on PATH
  (Git for Windows: https://git-scm.com/download/win).
- **YOLO pose weights won't download** — check your internet connection;
  the first run of any script that loads `YOLO("yolov8n-pose.pt")` needs
  it once, then it's cached locally.
- **Training is very slow / seems stuck** — that's normal on CPU-only
  Windows machines. Reduce `EPOCHS` in `train_model.py`, or reduce the
  amount of data processed in `fall_detection_fa1.py`.
- **`No class folders with images found`** — FA-1 didn't produce data for
  one or more classes; re-check `raw_dataset` structure (Step 3 above) and
  that annotation `.txt` filenames match their `.avi` counterparts exactly.
