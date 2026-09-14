# Image Classification System - Streamlit

This project converts the original Tkinter Image Classification System
to a Streamlit web application.

## Features
- Upload JPG, JPEG, PNG, BMP, or WEBP images
- Preview the uploaded image
- Display image resolution and file size
- Classify the image using MobileNetV2
- Show the top 5 ImageNet predictions
- Show model confidence
- Streamlit web interface

## Run the project

1. Open a terminal in this folder.
2. Install dependencies:

   pip install -r requirements.txt

3. Start Streamlit:

   streamlit run app.py

The application will open in your web browser.

## Model
- MobileNetV2
- TensorFlow/Keras
- ImageNet pretrained weights
