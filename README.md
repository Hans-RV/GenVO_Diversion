# Fall-Sense AI : Fall Detection System using YOLO & Twilio

## Overview
This project implements a *Fall Detection System* using *YOLO* object detection and *Twilio* messaging to alert a predefined contact in case of a detected fall. The system captures live video, detects human falls, and sends WhatsApp messages for the first five fall detections with real-time location.

## Features
- *Real-time Fall Detection* using YOLO
- *KMeans Clustering* for better fall classification
- *Twilio WhatsApp Messaging* for alerts
- *Geolocation Tracking* to provide location in messages
- *Live Webcam Feed* with bounding boxes & labels

## Installation
### Prerequisites
Ensure you have Python installed (3.7 or later recommended). Install required dependencies using:

sh
pip install opencv-python numpy ultralytics scikit-learn geocoder twilio


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
