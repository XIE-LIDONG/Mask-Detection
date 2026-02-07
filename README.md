Real-Time Mask Detection System with YOLO11

A high-performance real-time mask detection system built on YOLO11, supporting video stream monitoring, dynamic threshold adjustment, and CPU-based low-latency inference.

Project Overview
This project fine-tunes a high-precision mask detection model using the YOLO11 deep learning framework based on the public Face-Mask-Detection dataset from Kaggle. 
Core Language: Python
Deep Learning: Ultralytics YOLO11
Computer Vision: OpenCV (video stream processing, camera access)
Data Processing: NumPy (image data calculation)
Deployment & UI: Streamlit (local deployment, interactive interface)

Core Features
High-Precision Detection
Correctly worn masks (with_mask): 96.9% accuracy
Improperly worn masks (mask_worn_incorrectly): 89.2% accuracy
No masks (without_mask): 93.4% accuracy
Overall real-time detection accuracy: 95%
Efficient Inference

GPU-accelerated training for faster model iteration
Interactive UI & Control
Streamlit-based visual interface
Dynamic confidence threshold adjustment
One-click "Start/Stop Detection" controls
Real-time video stream monitoring

Author
XIE LIDONG
