# 👶 Baby Cry Sound Classifier 🎶

This Streamlit app allows users to upload a `.wav` file of a baby cry and classifies the sound into one of several categories (e.g., hungry, tired, discomfort, etc.) using a pre-trained deep learning model.

The app is designed to provide a fun and engaging experience while demonstrating the capabilities of machine learning in audio processing.

## 🚀 Features
- **Upload Baby Cry Audio**: The user can upload `.wav` files.
- **Real-time Prediction**: The app processes the audio file and provides an instant prediction about the possible reason for the baby's cry.
- **Cute & Fun UI**: A playful, engaging interface with emoji-enhanced messages.

## 📦 How to Run 

Visit 
https://heybaby.streamlit.app/

## 🧠 Model Details

The machine learning model is a deep convolutional neural network (CNN) trained on a dataset of baby cry sounds. The audio is converted into a spectrogram before being fed into the neural network.

The model predicts one of the following categories:

Belly Pain 😖
Burping 🍼
Discomfort 😣
Hungry 🍽️
Tired 😴
