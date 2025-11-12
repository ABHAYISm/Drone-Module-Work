import cv2

url = 'http://192.168.1.6:8080/stream.mjpeg?clientId=OeMov9e4xuuHZRhg'
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Failed to open stream")
else:
    print("✅ Stream opened successfully")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received")
            break
        cv2.imshow("Test Stream", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
