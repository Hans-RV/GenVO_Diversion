# Fall-Sense AI : Fall Detection System using YOLO & Twilio

## Problem Statement
Falls are a major health risk, especially for elderly individuals and people with disabilities. A fall can result in severe injuries, and the lack of immediate assistance can lead to critical conditions. Traditional fall detection methods rely on wearable devices, which may not always be worn or functional. This project aims to develop a computer vision-based Fall Detection System that uses YOLO, OpenCV, and AI-driven analysis to detect falls and send real-time notifications via WhatsApp, ensuring timely assistance and reducing health risks.

## Overview
The GenVO Diversion project aims to implement and test a fall detection system using YOLO-based object detection, GPS tracking, and alert mechanisms via Twilio. The system captures real-time video, detects human figures, classifies their posture, and logs location and time-based data to trigger emergency alerts when necessary.

## Features

- **Real-time Fall Detection**: Uses YOLO for object detection and KMeans clustering for posture classification.
- **GPS Tracking**: Retrieves live location data upon each detection.
- **Emergency Alert System**: Sends alerts via Twilio when a fall is detected.
- **DeepSeek R1 API** : Integrated for additional AI-based insights
- **Logging Mechanism**: Stores detections with timestamps and location coordinates.
- **Testing & Evaluation**: Includes a separate testing framework for validation purposes

## File Structure

- `app.py` - Practical implementation for real-world application.
- `GenVO_Diversion_Final.ipynb` - Testing notebook for evaluation and debugging.
- `model.pt` - A YOLO trained model for object detection.
- `README.md` - Documentation and setup instructions.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- OpenCV
- NumPy
- YOLO (Ultralytics)
- Scikit-learn
- Geocoder
- Twilio Python SDK
- openai

### Usage

# Practical Implementation

- Run the script to start real-time object detection.
- If a fall is detected, an alert is sent via Twilio along with GPS coordinates.
- Logs are stored with timestamps for analysis.

# Testing Framework

- This version helps in fine-tuning the model by analyzing detection accuracy.
- DeepSeek R1 API is utilized for further insights and classification improvements, providing insights to possibles reasons of the trips and falls.
- Runs without alert triggers, focusing on debugging and parameter adjustments.

### Installation

1. Clone the repository:
   ```sh
   git clone <repo_url>
   cd GenVO_Diversion
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up Twilio credentials in `app.py`.
4. Run the script:
   ```sh
   python app.py
   ```

### API Integration

## DeepSeek API

- DeepSeek API is used to enhance object recognition accuracy by refining bounding box classification and improving fall detection.
- Endpoint: https://api.deepseek.com/ai-model
- Usage: The API is queried for additional object classification and confidence scoring.

## Twilio API

- Sends WhatsApp or SMS alerts when a fall is detected.
- Requires a valid Twilio account SID and authentication token.

### Twilio Setup
1. Create an account on [Twilio](https://www.twilio.com/)
2. Get your *Account SID, **Auth Token, and a **Twilio WhatsApp Number*
3. Replace the following in fall_detection.py:
   python
   account_sid = 'your_account_sid'
   auth_token = 'your_auth_token'
   from_wa_number = 'whatsapp:+14155238886'  # Twilio sandbox number
   to_wa_number = 'whatsapp:+your_number'  # Your WhatsApp number
   

## Usage
Run the script to start detecting falls:

sh
python fall_detection.py


## How It Works
1. *Captures live video* from the webcam.
2. *Detects people* using YOLO.
3. *Analyzes posture* using KMeans clustering.
4. *Detects falls* based on aspect ratio and position.
5. *Sends a WhatsApp alert* for the first five falls.

## Stopping the Program
Press q to exit the live feed.

## Limitations
- Needs a *GPU* for real-time processing.
- *Lighting conditions* may affect detection accuracy.
- Requires *internet connectivity* for Twilio messaging.

## Future Enhancements
- Implement *SMS & Call alerts* for emergencies.
- Use *Deep Learning models* for better fall classification.
- Integrate with *IoT sensors* for enhanced accuracy.

## Contact
For questions or contributions, feel free to reach out!
