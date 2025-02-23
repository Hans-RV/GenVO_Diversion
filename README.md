# Fall-Sense AI (Human Fall Detection System)

## Overview
This project uses *YOLO* (You Only Look Once) for *fall detection* in real-time. The model identifies whether a person is *standing* or *falling* based on bounding box aspect ratios. It can be used *locally via webcam* or *in Colab using video files*.

---

## Problem Statement
Falls are a major health risk, especially for elderly individuals and people with disabilities. A fall can result in severe injuries, and the lack of immediate assistance can lead to critical conditions. Traditional fall detection methods rely on wearable devices, which may not always be worn or functional. This project aims to develop a computer vision-based Fall Detection System that uses YOLO, OpenCV, and AI-driven analysis to detect falls and send real-time notifications via WhatsApp, ensuring timely assistance and reducing health risks.

---

## Features
- *Real-time fall detection* using OpenCV & YOLO.
- *Webcam support* for live detection.
- *Google Colab compatibility* for analyzing pre-recorded videos.
- *Alerts based on fall probability* (bounding box aspect ratio).

---

## Installation
### Local Setup
1. Install dependencies:
   bash
   pip install opencv-python numpy ultralytics torch torchvision
   
2. Clone the repository and place the **YOLO model (model.pt)** inside the project folder.
3. Run the script:
   bash
   python fall_detection.py
   

### Google Colab Setup
1. Install dependencies in Colab:
   python
   !pip install opencv-python numpy ultralytics mediapipe torch torchvision
   
2. Load video files from Google Drive.
3. Run the Colab notebook.

---

## Usage
### Running Locally (Webcam)
- The script captures frames from the webcam and applies *YOLO-based tracking*.
- If a fall is detected (width/height > 0.7), an *alert message* is displayed.
- Press q to exit the live feed.

### Running in Google Colab
- Uses *pre-recorded videos* for fall detection.
- Displays processed frames using cv2_imshow().
- Useful for *evaluating different scenarios* without a live feed.

---

## File Structure

/The Project_Folder

│── fall_detection.py  # Main script for webcam-based detection

│── colab_fall_detection.ipynb  # Google Colab version

│── model.pt  # YOLO model weights


---

## Future Improvements
- *Enhance accuracy* by fine-tuning the model.
- *Add sound or notification alerts* when a fall is detected.
- *Deploy as a mobile/IoT solution* for real-world applications.
