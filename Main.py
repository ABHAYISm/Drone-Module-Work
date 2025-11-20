from ultralytics import YOLO

def run():
    # Load model
    model = YOLO("C:\\Users\\abhay\\OneDrive\\Desktop\\pyhtonfordrone\\runs\\detect\\train4\\weights\\best.pt")

    # Train
    model.train(
        data="D:/Military and Civilian Vehicles Classification/data.yaml",
    epochs=150,
    imgsz=640,
    batch=4,
    device=0,
    workers=2,
    mosaic=1,       # IMPORTANT for small object accuracy
    hsv_h=0.015,    # color augmentations help generalization
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.9,
    fliplr=0.5,
    cache=True 
    )

    # Validate
    metrics = model.val()
    print(metrics)

    # Predict
    results = model("C:/drone work/test.jpg")
    results[0].show()

if __name__ == "__main__":
    run()
