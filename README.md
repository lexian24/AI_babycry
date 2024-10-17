👶 Baby Cry Sound Classifier 🎶

This Streamlit app allows users to upload a .wav file of a baby cry and classifies the sound into one of several categories (e.g., hungry, tired, discomfort, etc.) using a pre-trained deep learning model.

The app is designed to provide a fun and engaging experience while demonstrating the capabilities of machine learning in audio processing.

🚀 Features

Upload Baby Cry Audio: The user can upload .wav files.
Real-time Prediction: The app processes the audio file and provides an instant prediction about the possible reason for the baby's cry.
Cute & Fun UI: A playful, engaging interface with emoji-enhanced messages.
🛠️ Requirements

Python Version
Python 3.7+
Required Libraries
Make sure to install the following dependencies:

bash
Copy code
streamlit==1.24.0
librosa==0.10.0
numpy==1.23.5
tensorflow==2.12.0
soundfile==0.12.1
scikit-learn==1.3.0  # Optional, for accuracy calculation if needed
You can install all the dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
FFmpeg (for audio processing)
To ensure librosa works correctly, you may need to install FFmpeg. Here’s how:

Linux:
bash
Copy code
sudo apt-get install ffmpeg
macOS: Install via brew:
bash
Copy code
brew install ffmpeg
Windows: Download and install from here.
Model File
Ensure you have the pre-trained model file (trained_baby_cry_model.h5) in the root directory of your project.

If you need to generate a model, you can train one using your own dataset and save it using TensorFlow:

python
Copy code
model.save('trained_baby_cry_model.h5')
📦 How to Run Locally

Clone the repository:
bash
Copy code
git clone https://github.com/your-repo/baby-cry-classifier.git
cd baby-cry-classifier
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:
bash
Copy code
streamlit run app.py
Upload a .wav file and get the predicted reason for the baby cry.
🌐 How to Deploy

You can deploy the app to Streamlit Cloud or any platform supporting Streamlit apps.

Push your code to a GitHub repository.
Sign in to Streamlit Cloud.
Deploy your app by linking your GitHub repository.
Add the necessary environment settings (e.g., model file and requirements.txt).
🧠 Model Details

The machine learning model is a deep convolutional neural network (CNN) trained on a dataset of baby cry sounds. The audio is converted into a spectrogram before being fed into the neural network.

The model predicts one of the following categories:

Belly Pain 😖
Burping 🍼
Discomfort 😣
Hungry 🍽️
Tired 😴
🖼️ User Interface

The UI is designed to be engaging, playful, and intuitive:

Users upload an audio file.
The app displays the sound visually and provides a real-time prediction.
Fun, emoji-enhanced messages for predictions.
📄 License

This project is licensed under the MIT License. See the LICENSE file for more details.

Enjoy using the Baby Cry Sound Classifier! 💖👶
