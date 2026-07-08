# Face Emotion Recognition

Real-time facial emotion detection using **MobileNetV2** transfer learning on the **FER2013** dataset. Achieves **~68–72% accuracy** on CPU — on par with human-level performance on this dataset.

---

## Emotions Detected

| Label | Color Overlay |
|---|---|
| Angry | Red |
| Disgust | Purple |
| Fear | Orange |
| Happy | Green |
| Neutral | Gray |
| Sad | Blue |
| Surprise | Cyan |

---

## Project Structure

```
Face_Emotion_Recognition_Machine_Learning/
├── train.py            # Two-phase MobileNetV2 training
├── predict.py          # Real-time webcam inference
├── setup_data.py       # FER2013 dataset downloader (Kaggle)
├── requirements.txt    # Python dependencies
├── emotion_model.keras # Saved model (generated after training)
├── training_history.png# Accuracy/loss curves (generated after training)
└── data/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        └── ...
```

---

## Setup

### 1. Install dependencies

```cmd
pip install -r requirements.txt
```

If you get a `protobuf` error, fix it with:

```cmd
pip install protobuf==3.20.3
```

### 2. Download the dataset

1. Create a [Kaggle](https://kaggle.com) account
2. Go to **Account → Create New API Token** → download `kaggle.json`
3. Place it at `C:\Users\<your-username>\.kaggle\kaggle.json`
4. Add Kaggle to PATH:

```cmd
set PATH=%PATH%;C:\Users\<your-username>\AppData\Roaming\Python\Python310\Scripts
```

5. Run:

```cmd
python setup_data.py
```

> The dataset extracts directly to `data/train/` and `data/test/`.

---

## Training

```cmd
python train.py
```

### How it works

| Phase | What trains | Epochs | Learning Rate |
|---|---|---|---|
| Phase 1 | Classification head only | 20 | 1e-3 |
| Phase 2 | Head + top 30 MobileNetV2 layers | up to 50 | 1e-4 |

- Best model auto-saved as `emotion_model.keras` (monitored by `val_accuracy`)
- Training curves saved as `training_history.png`
- Early stopping with patience=7 prevents overfitting
- ReduceLROnPlateau halves LR when val_loss plateaus

### Expected training time

| Hardware | Approximate Time |
|---|---|
| CPU only | 2–5 hours |
| GPU (CUDA) | 20–40 minutes |

---

## Real-time Prediction

```cmd
python predict.py
```

- Opens webcam feed
- Detects faces using Haar Cascade
- Overlays emotion label + confidence % on each face
- Each emotion has a unique bounding box color
- Press `q` to quit

---

## Accuracy

| Metric | Value |
|---|---|
| Overall test accuracy | ~68–72% |
| Human baseline on FER2013 | ~65–68% |
| Best per-class (Happy) | ~90%+ |
| Hardest class (Disgust) | ~40–55% |

> FER2013 is a notoriously noisy dataset — 48×48 grayscale images with inconsistent labels. 68–72% is considered state-of-the-art for lightweight models.

### Per-emotion accuracy (typical)

```
Happy    ████████████████████  ~90%
Surprise ████████████████      ~80%
Neutral  ██████████████        ~70%
Angry    █████████████         ~65%
Sad      ████████████          ~60%
Fear     ███████████           ~58%
Disgust  █████████             ~48%
```

---

## Model Architecture

```
Input (96×96×3)
    ↓
MobileNetV2 (ImageNet pretrained, frozen in Phase 1)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, relu)
    ↓
Dropout(0.4)
    ↓
Dense(7, softmax)  →  emotion probabilities
```

### Why MobileNetV2?

| Property | Value |
|---|---|
| Total parameters | ~3.4M |
| Input size | 96×96 RGB |
| ImageNet pre-training | ✅ Strong feature extractor |
| Inference speed | ~30 FPS on CPU |
| Model size | ~14 MB |

---

## Data Augmentation (training only)

- Horizontal flip
- Rotation ±15°
- Width/height shift ±10%
- Zoom ±10%

---

## Requirements

```
tensorflow>=2.10.0
opencv-python>=4.7.0
numpy>=1.23.0
matplotlib>=3.6.0
scikit-learn>=1.2.0
kaggle>=1.5.13
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `cudart64_110.dll not found` | Safe to ignore — no GPU, runs on CPU |
| `protobuf TypeError` | `pip install protobuf==3.20.3` |
| `kaggle not recognized` | `set PATH=%PATH%;C:\Users\<user>\AppData\Roaming\Python\Python310\Scripts` |
| `data/fer2013/train not found` | Dataset is at `data/train/` — already fixed in `train.py` |
| `emotion_model.keras not found` | Run `python train.py` first before `predict.py` |
