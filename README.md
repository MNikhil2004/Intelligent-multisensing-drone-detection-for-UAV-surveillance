# 🛸 Drone Detection System — Approach B Gated Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![PyQt5](https://img.shields.io/badge/PyQt5-41CD52?style=for-the-badge&logo=qt&logoColor=white)

**A multi-modal drone detection system combining RGB video, infrared video, and audio streams through a gated inference pipeline with a desktop GUI.**

[Overview](#-overview) · [Architecture](#-pipeline-architecture) · [Models](#-models) · [Installation](#-installation) · [Usage](#-usage) · [Results](#-results)

</div>

---

## 📌 Overview

This system detects drones in real-world environments using a **gated inference pipeline (Approach B)** that intelligently combines three modalities:

- 📷 **RGB Video** — processed via FrameDiff-based motion gating into a RCapsGRU classifier
- 🌡️ **IR (Infrared) Video** — processed via DualMOG2-based motion gating into a separate RCapsGRU classifier
- 🎙️ **Audio** — segmented with a sliding window and classified by an AudioCNN-GRU model

Final predictions are produced through both **static** and **dynamic weighted fusion**, with an interactive desktop GUI for real-time visualization and analysis.

---


### Fusion Weights

| Fusion Type | RGB Weight | IR Weight | Audio Weight |
|-------------|-----------|----------|-------------|
| **Static**  | 0.40      | 0.35     | 0.25        |
| **Dynamic** | \|P − 0.5\| × 2 (confidence-proportional per modality) | | |

---

## 🧠 Models

### RCapsGRU (RGB & IR)

A Recurrent Capsule Network for spatiotemporal drone classification.
Input Frames (40 × 128×128)
→ Conv2D Feature Extractor
→ CapsuleLayer (routing by agreement)
→ GRU (temporal aggregation)
→ Sigmoid output (drone probability)

- **RGB variant**: 3-channel input, FrameDiff motion-gated frames
- **IR variant**: 1-channel grayscale input, DualMOG2 motion-gated frames

### AudioCNN-GRU

A 2D CNN + Bidirectional GRU operating on log-Mel spectrograms.
Audio Segment (4s @ 16kHz)
→ Log-Mel Spectrogram (128 mels, n_fft=1024, hop=256)
→ CNN (3-layer with BatchNorm + MaxPool)
→ Bidirectional GRU
→ FC → Softmax (drone / not-drone)

### Motion Gating

| Modality | Method    | Why |
|----------|-----------|-----|
| RGB      | FrameDiff | Best F1 = 0.1290, Best IoU = 0.1429 across tracker evaluation |
| IR       | DualMOG2  | Lowest fragmentation (0.07), Best F1 = 0.0696 |

Only frames with detected motion are passed to the neural networks — reducing false positives and computation.

---

## 🖥️ GUI Features

| Feature | Description |
|---------|-------------|
| **Multi-modal input** | Load RGB video, IR video, and/or audio independently |
| **Animated probability bars** | Per-modality and fusion probabilities with live animation |
| **Confidence gauge** | Semicircle gauge (0–100 scale) |
| **Annotated video output** | Bounding boxes, Kalman-tracked target, probability bar overlay |
| **Built-in video player** | Play/pause/stop with seek slider for annotated outputs |
| **Audio visualization** | Waveform, Mel spectrogram, and per-segment probability chart |
| **Detection history log** | Zebra-striped table with all past run results |
| **CSV export** | Export full detection history with timestamps |
| **Drop shadows & hover effects** | Interactive modern UI design |
> 
<img width="384" height="210" alt="image" src="https://github.com/user-attachments/assets/1e9caef8-de4b-4c1f-abff-7c94b6310a63" />

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU *(optional but recommended)*

### Install Dependencies

```bash
pip install PyQt5 opencv-python torch torchvision
pip install librosa numpy matplotlib
```

> **Note:** If you don't have `librosa`, audio analysis will be disabled but RGB/IR detection will still work.

---

## 🚀 Usage

### 1. Set Model Paths

Open `Drone_detection.py.py` and update the paths to your trained model weights:

```python
self.RGB_MODEL_PATH   = r'path\to\rcaps_gru_rgb.pth'
self.IR_MODEL_PATH    = r'path\to\rcaps_gru_ir.pth'
self.AUDIO_MODEL_PATH = r'path\to\audio_cnn_gru.pth'
```

### 2. Run the Application

```bash
python Drone_detection.py
```

### 3. Workflow

Click "Load Models"          → Loads all three neural networks
Select RGB / IR / Audio      → Choose one or more input files
Click "RUN DETECTION"        → Starts gated inference pipeline
View results in tabs:

📊 Results  → Probability bars + Confidence gauge + Verdict
🎬 Videos   → Annotated output video playback
🎙 Audio    → Waveform + Mel spectrogram + segment chart
📋 History  → All past detections in a table


Export CSV if needed


> You do **not** need all three modalities — the pipeline gracefully handles any single or combination of inputs.

---

## 📁 Repository Structure
drone-detection-system/
│
├── Drone_detection.py       # Main application (GUI + inference engine)
├── README.md                    # This file
│
├── models/                      # Trained model weights (not included in repo)
│   ├── rcaps_gru_rgb.pth
│   ├── rcaps_gru_ir.pth
│   └── audio_cnn_gru.pth
│
├── sample_data/                 # Sample clips for testing
│   ├── sample_rgb.mp4
│   ├── sample_ir.mp4
│   └── sample_audio.wav
│
├── docs/
│   ├── project_report.pdf
│   └── screenshots/
│
└── outputs/                     # Annotated output videos (generated at runtime)

---

## 📊 Results

### RGB-Only Detection (P_RGB = 0.912)

<img width="443" height="269" alt="image" src="https://github.com/user-attachments/assets/a0b6ca28-1018-4f58-a92f-8a58f2a90e1c" />


The RCapsNet–GRU model correctly identifies the drone from the visible video stream. The bounding box localizes the target in the frame with a confidence of **91.2%**.

---

### IR-Only Detection (P_IR = 0.928)

<img width="425" height="270" alt="image" src="https://github.com/user-attachments/assets/c3807269-0214-47a0-be43-21377f445aad" />


The thermal heat signature of the drone is reliably captured by the RCapsNet–GRU model operating on single-channel infrared frames with a confidence of **92.8%**.

---

### Audio Analysis

<img width="379" height="213" alt="image" src="https://github.com/user-attachments/assets/1a32e51e-03a0-4db4-8f0e-5b98f63996fe" />


The audio visualization shows three panels:
- **Top** — Waveform (amplitude vs time)
- **Middle** — Log-Mel spectrogram used as model input
- **Bottom** — Per-segment drone probability (4s window, 2s hop). Red bars exceed the 0.5 threshold; green bars fall below it.

---

### RGB + IR Dual-Modality Fusion (P_dynamic = 0.936)

<img width="438" height="267" alt="image" src="https://github.com/user-attachments/assets/3c41944e-cc40-4ec9-84ea-40fd2d1e7e77" />


The combination of visible and thermal evidence produces a higher fused confidence (0.936) than either modality individually, demonstrating the additive benefit of complementary visual sensing.

---

### Scenario Evaluation Table



Full scenario evaluation across all input configurations:

| Group | Scenario | Modalities | Ground Truth | P_RGB | P_IR | P_Audio | P_Static | P_Fusion | Decision |
|-------|----------|------------|-------------|-------|------|---------|----------|----------|----------|
| A | S1 | RGB Only | Drone | 0.912 | N/A | N/A | 0.912 | 0.912 | **Drone** |
| A | S2 | RGB Only | Not Drone | 0.032 | N/A | N/A | 0.032 | 0.032 | Not Drone |
| A | S3 | IR Only | Drone | N/A | 0.928 | N/A | 0.928 | 0.928 | **Drone** |
| A | S4 | IR Only | Not Drone | N/A | 0.048 | N/A | 0.048 | 0.048 | Not Drone |
| A | S5 | Audio Only | Drone | N/A | N/A | 0.996 | 0.996 | 0.996 | **Drone** |
| A | S6 | Audio Only | Not Drone | N/A | N/A | 0.010 | 0.010 | 0.010 | Not Drone |
| B | D1 | RGB + IR | Drone | 0.907 | 0.961 | N/A | 0.933 | 0.936 | **Drone** |
| B | D2 | RGB + IR | Not Drone | 0.035 | 0.297 | N/A | 0.157 | 0.115 | Not Drone |
| B | D3 | RGB + Audio | Drone | 0.904 | N/A | 0.290 | 0.668 | 0.694 | **Drone** |
| B | D4 | RGB + Audio | Not Drone | 0.085 | N/A | 0.030 | 0.053 | 0.040 | Not Drone |
| B | D5 | IR + Audio | Drone | N/A | 0.950 | 0.743 | 0.864 | 0.878 | **Drone** |
| B | D6 | IR + Audio | Not Drone | N/A | 0.051 | 0.010 | 0.030 | 0.0250 | Not Drone |
| B | T1 | RGB+IR+Audio | Drone | 0.903 | 0.935 | 0.68 | 0.858 | 0.877 | **Drone** |
| B | T2 | RGB+IR+Audio | Not Drone | 0.913 | 0.051 | 0.01 | 0.383 | 0.294 | Not Drone |



---

### Confusion Matrix

<img width="343" height="197" alt="image" src="https://github.com/user-attachments/assets/9bde6386-724a-4b45-b3aa-9ffe1bd0d96b" />


| | Predicted: Drone | Predicted: Not Drone |
|---|---|---|
| **Actual: Drone** | 187 | 2 |
| **Actual: Not Drone** | 10 | 281 |

**Overall accuracy: 97.5%** across the full test set.

---

### Tracker Evaluation Summary

| Modality | Tracker   | Best F1    | Best IoU   | Fragmentation |
|----------|-----------|------------|------------|---------------|
| RGB      | FrameDiff | **0.1290** | **0.1429** | — |
| IR       | DualMOG2  | **0.0696** | — | **0.07** |

**Classes evaluated:** Drone · Airplane · Bird · Helicopter

---



## 🔬 Technical Notes

- **Kalman Filter** tracks detected contours across frames for smooth bounding box prediction
- **CLAHE** (Contrast-Limited Adaptive Histogram Equalization) is applied before all motion detection steps
- **DualMOG2** runs two background subtractors in parallel (slow: `history=150`, fast: `history=30`) and OR-merges their masks for robust IR foreground extraction
- Contour area is filtered between **8 px²** and **15,000 px²** to eliminate noise and large irrelevant regions
- Output videos are re-encoded to H.264 via `ffmpeg` when available, falling back to raw `mp4v` otherwise

---

<div align="center">
Built with PyTorch · OpenCV · PyQt5 · librosa
</div>
