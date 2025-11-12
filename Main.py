from ultralytics import YOLO

def run():
    # Load model
    model = YOLO("yolov8n.pt")

    # Train
    model.train(
        data="C:/datasets/cars_tanks/data.yaml",
    epochs=100,
    imgsz=512,
    batch=4,
    device=0,
    workers=1,
    mosaic=0  # or 0 if GPU is detected
    )

    # Validate
    metrics = model.val()
    print(metrics)

    # Predict
    results = model("C:/drone work/test.jpg")
    results.show()

if __name__ == "__main__":
    run()
