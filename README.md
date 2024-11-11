# Object Counting Project

## Project Overview
This project focuses on accurately counting objects in different contexts using deep learning techniques. It supports object detection and counting in real-time streams, images, and videos, making it versatile for various use cases such as vehicle and license plate detection.

## Project Components
The project is divided into four main components:
1. **Real-time Object Counting**  
   This module uses live video streams (e.g., webcam feeds) to detect and count objects in real-time. It leverages optimized deep learning models for speed and accuracy.

2. **Image-based Object Counting**  
   This code processes static images to detect and count objects, providing reliable results for individual image analysis scenarios.

3. **Video-based Object Counting**  
   The video processing code can handle entire video files, detecting and counting objects frame-by-frame to produce comprehensive results for videos.

4. **Integrated Interface**  
   A user-friendly interface that brings together all functionalities: real-time, image, and video object counting. It allows users to easily select and switch between different modes, making the tool flexible and easy to use.

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
