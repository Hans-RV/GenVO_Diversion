import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('object-detection-using-webcam-main\object-detection-using-webcam-main\model.pt')

# Open Webcam
webcamera = cv2.VideoCapture(0)
webcamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not webcamera.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    success, frame = webcamera.read()
    if not success:
        print("Error: Failed to capture frame.")
        break

    # Run YOLO detection
    results = model.track(frame, classes=0, conf=0.5)

    # Check if detections exist
    if results and len(results) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])  # Bounding box coordinates
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height  # Aspect Ratio (Width / Height)

            # If width is larger than height, it's likely a fall
            if aspect_ratio > 0.7:
                label = "I'm Falling!!Help ME!!"
                color = (0, 0, 255)  # Red alert
            else:
                label = "Standing"
                color = (0, 255, 0)  # Green safe

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display Video
    cv2.imshow("Live Camera", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break;

# Cleanup
webcamera.release()
cv2.destroyAllWindows()
