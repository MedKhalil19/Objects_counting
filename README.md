# Object Counting Project

## Project Overview
This project focuses on accurately counting objects in different contexts using deep learning techniques. We utilize the YOLOv8 object detection model, which is known for its high accuracy and speed, making it suitable for real-time applications. The project supports object detection and counting in real-time streams, images, and videos, providing flexible solutions for various use cases such as vehicle and license plate detection.

## Project Components
The project is divided into four main components:
1. **Real-time Object Counting (yolov8_real_time.py)**  
   This module uses live video streams (e.g., webcam feeds) to detect and count objects in real-time. It leverages the YOLOv8 model for efficient detection, making it capable of handling dynamic scenarios quickly and accurately.

2. **Image-based Object Counting (yolov8_img.py)**  
   This code processes static images to detect and count objects using the YOLOv8 model. It offers precise results for analyzing individual images, making it useful for tasks like analyzing batches of photos for object presence and count.

3. **Video-based Object Counting (yolov8_vid.py)**  
   This module processes entire video files to detect and count objects frame-by-frame using the YOLOv8 model. It generates comprehensive counting reports for video data and is ideal for surveillance, traffic analysis, and similar tasks.

4. **Integrated Interface (main.py)**  
   A user-friendly interface that brings together all functionalities: real-time, image, and video object counting. It allows users to easily select and switch between different modes, making the tool versatile and convenient to use.

---

## Installation and Setup Instructions
To get started with this project on your laptop, follow these steps:
1. **Install Python 3.7.9**  
   Ensure you have Python 3.7.9 installed, as it is compatible with the project dependencies.

2. **Install Required Packages**  
   Use the following command to install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Optional: Add PyTorch for GPU Support**  
   If you have a compatible GPU and want to accelerate processing, install PyTorch with CUDA support:
   ```bash
   # Example for Windows (modify as needed for your environment)
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
   ```
   Adjust the version based on your CUDA version.

4. **Choose Your Counting Mode**  
   Select which counting mode you want to use (real-time, image, or video processing) and run the respective script.
---

⚠️ **Note**: Due to the size of the project video, I will be uploading it to my LinkedIn profile. Watch it on my LinkedIn to prove my work: [objects_counting's LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7261845336736808963/)

---

This should correctly display the clickable link to your LinkedIn video. Let me know if you need any further adjustments!

---

## How YOLOv8 Enhances Object Counting
The project uses YOLOv8 (You Only Look Once, version 8), a state-of-the-art object detection framework. YOLOv8 is designed for speed and accuracy, which allows for real-time performance and high precision across diverse datasets. This makes it well-suited for complex applications, including vehicle detection and more general object counting tasks.

## Contact Information
For more details, feel free to contact me:
- **Email:** khlifimedkhalil@gmail.com  
- **LinkedIn:** [Khlifi Med Khalil](https://www.linkedin.com/in/khlifi-medkhalil/)
