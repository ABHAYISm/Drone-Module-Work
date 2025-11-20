from ultralytics import YOLO

def run():
    # Load your trained model
    model = YOLO("C:\\Users\\abhay\\OneDrive\\Desktop\\pyhtonfordrone\\runs\\detect\\train5\\weights\\best.pt")
    
    # Validate (optional)
    # metrics = model.val()
    # print(metrics)

    # 🔥 Run prediction on a video
    results = model.predict(
        source="C:/drone work/testvid.mp4",  # <-- your video path
        show=True,          # show a live window
        save=True,          # save output video in runs/detect/predict
        conf=0.25,          # confidence threshold
        imgsz=640,          # image size for detection
        device=0            # use GPU
    )

if __name__ == "__main__":
    run()
