# Ml Model with Mirroring using RTSP and HTTPS

This project trains a YOLOv8 model to detect tanks (and optionally other vehicles) using a custom dataset. It also supports streaming video via RTSP to your model for real-time detection.

## Requirements

Python 3.11+

YOLOv8 (ultralytics package)

OpenCV

PyTorch (GPU-enabled)

VLC (for RTSP testing)

MediaMTX (RTSP server)


```bash
https://github.com/bluenviron/mediamtx/releases

Download Python 3.11
pip install --upgrade pip
pip install opencv-python flask requests numpy pillow
```
## Setup
Clone/Prepare Project
```bash
git clone https://github.com/ABHAYISm/Drone-Module-Work
cd <your-project-folder>
```
Create Python Environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```
Install Dependencies
```bash
pip install --upgrade pip
pip install ultralytics opencv-python torch torchvision
```
Prepare Dataset
```bash
dataset/
  train/
    cars/
    tanks/
  val/
    cars/
    tanks/
```

## Training YOLOv8
Initialize Model
```bash
from ultralytics import YOLO

# Start from scratch (no pretrained weights)
model = YOLO('yolov8n.yaml')  # Or yolov8n.yaml model config
```
Train Model
```bash
model.train(
    data='dataset.yaml',      # Your dataset YAML
    epochs=100,               # for better work we can reduce epoch size
    batch=8,                  # Increase batch size for better GPU utilization
    imgsz=640,                # Image size
    device=0                  # GPU device ID if no GPU we can use 'cpu'
)
```

## RTSP Streaming
Install MediaMTX

Download from MediaMTX GitHub

Place mediamtx.exe in your folder

Create Configuration (mediamtx.yml)
```python
paths:
  mystream:
    source: publisher
```
Run MediaMTX
```bash
.\mediamtx.exe .\mediamtx.yml
```
Publish Stream
```bash
import cv2
import rtsp

cap = cv2.VideoCapture(0)  # or screen capture

while True:
    ret, frame = cap.read()
    # Send frame to RTSP server (publisher)

```
Access Stream

From URL,

In VLC Media Player
```bash
Media → Open Network Stream → rtsp://<YOUR_IP>:8554/mystream
```
```
┌─────────────────────────────┐
│       Dataset               │
│  - Images/Videos            │
│  - Labels (YOLO format)     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│       YOLOv8 Training       │
│  - Model: YOLOv8            │
│  - GPU: CUDA / device 0     │
│  - Epochs / Batch size      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   Model Check & Evaluation  │
│  - Validation on test set   │
│  - Metrics: mAP, loss       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  RTSP Stream Publisher      │
│  - Camera / Screen Capture  │
│  - FFmpeg / OpenCV          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│   MediaMTX RTSP Server      │
│  - Config: mediamtx.yml     │
│  - RTSP URL: rtsp://<IP>    │
│  - Protocols: TCP/UDP       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    YOLOv8 Inference         │
│  - Reads from RTSP or video │
│  - Runs detection on frames │
│  - GPU accelerated          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Real-Time Visualization      │
│  - OpenCV / VLC / GUI        │
│  - Bounding boxes & labels   │
│  - Optional logging / saving │
└─────────────────────────────┘

```
## How to Run
First Run Inference.py to host the server

Then run rtsp.py 

Before that ensure all connections are made properly.

