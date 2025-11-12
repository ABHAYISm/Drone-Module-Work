from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np, cv2

app = Flask(__name__)

print("🚀 Loading YOLO model...")
model = YOLO("yolov8n.pt")  # or your custom .pt model

@app.route('/api/detect', methods=['POST'])
def detect():
    file = request.files.get('frame')
    if not file:
        return jsonify([])

    data = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    # Run inference
    results = model(img, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            score = float(box.conf[0])

            # ✅ only show tanks
            if "tank" not in label.lower():
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "score": score
            })

    return jsonify(detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
