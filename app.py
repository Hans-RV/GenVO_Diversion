import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

# Load YOLO model
model = YOLO('object-detection-using-webcam-main/object-detection-using-webcam-main/model.pt')

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

    feature_vectors = []  # Stores (aspect_ratio, area, y_coordinate)
    bounding_boxes = []  # Stores (x1, y1, x2, y2)

    if results and len(results) > 0:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            area = width * height

            feature_vectors.append([aspect_ratio, area, y2])  # y2 -> lower part of the person
            bounding_boxes.append((x1, y1, x2, y2))

    if len(feature_vectors) > 1:
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_vectors)

        # Identify falling cluster (higher aspect ratio, lower y2)
        cluster_means = np.array([np.mean([feature_vectors[i] for i in range(len(labels)) if labels[i] == c], axis=0)
                                  for c in range(2)])

        falling_cluster = np.argmax(cluster_means[:, 0])  # Cluster with higher aspect ratio is likely 'falling'

        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            aspect_ratio, area, y_position = feature_vectors[i]

            avg_y = np.mean(cluster_means[:, 2])  # Average y-position
            avg_aspect_ratio = np.mean(cluster_means[:, 0])  # Average aspect ratio

            if labels[i] == falling_cluster and aspect_ratio > avg_aspect_ratio and y_position > avg_y:
                label = "I'm Falling!! Help ME!!"
                color = (0, 0, 255)  # Red for fall detection
            elif labels[i] == 1 - falling_cluster and aspect_ratio > 0.7:
                label = "Just Bending"
                color = (255, 0, 0)  # Blue for bending
            else:
                label = "I'm Standing!"
                color = (0, 255, 0)  # Green for standing

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display Video
    cv2.imshow("Live Camera", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
webcamera.release()
cv2.destroyAllWindows()
