from ultralytics import YOLO

def run():
    # Load your trained weights instead of starting training
    model = YOLO("runs/detect/train/weights/last.pt")
    
    # Validate performance
    metrics = model.val()
    print(metrics)

    # Run prediction on an image
    results = model("C:/drone work/test1.jpg")
    results[0].show()

    # Or run live webcam detection
    # results = model.predict(source="D:/testvid.mp4", show=True)

if __name__ == "__main__":
    run()