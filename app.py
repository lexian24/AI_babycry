import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
MODEL_PATH = '/Users/lexiancheo/Desktop/babycry/trained_baby_cry_model.h5'
model = load_model(MODEL_PATH)

# Define the labels
LABELS = ['Belly Pain 😖', 'Burping 🍼', 'Discomfort 😣', 'Hungry 🍽️', 'Tired 😴']

# Function to convert audio to a spectrogram
def wav_to_spectrogram(file, output_shape=(128, 64)):
    y, sr = librosa.load(file, sr=8000, dtype=np.float32)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = librosa.util.fix_length(spectrogram, size=output_shape[1], mode='constant', constant_values=0)
    spectrogram = librosa.util.normalize(spectrogram)
    return np.array(spectrogram).reshape(1, 128, 64, 1)  # Return with proper shape for prediction

# Streamlit UI Customization
st.markdown(
    """
    <style>
    .title {
        color: yellow;
        text-align: center;
        font-size: 50px;
    }
    .subtitle {
        text-align: center;
        font-size: 25px;
        color: #4682B4;
    }
    .footer {
        color: gray;
        text-align: center;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add some vertical spacing
def add_vertical_space(lines=1):
    for _ in range(lines):
        st.markdown("<br>", unsafe_allow_html=True)

# Title with emoji
st.markdown('<div class="title">👶 Baby Cry Sound Classifier 🎶</div>', unsafe_allow_html=True)

# Add spacing
add_vertical_space(2)

# Subtitle
st.markdown('<div class="subtitle">Let\'s find out why the baby is crying! 🍼</div>', unsafe_allow_html=True)

# Add spacing
add_vertical_space(2)

# Upload audio file
uploaded_file = st.file_uploader("Upload a baby cry audio file (.wav only) 🎤", type=["wav"])

# Add some space before audio display
add_vertical_space(2)

# If a file is uploaded
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')  # Play the uploaded audio
    
    # Add space before showing message
    add_vertical_space(1)
    
    st.write("🎧 Here's the sound of the baby crying. Let's analyze it...")

    # Convert the uploaded audio to a spectrogram
    spectrogram = wav_to_spectrogram(uploaded_file)
    
    # Make prediction using the model
    prediction = model.predict(spectrogram)
    
    # Get the class with the highest predicted probability
    predicted_label = LABELS[np.argmax(prediction)]
    
    # Add space before showing prediction result
    add_vertical_space(2)

    # Display a cute prediction message
    st.success(f"**Prediction Result:** The baby might be feeling: {predicted_label} 💖")
else:
    # Add spacing before upload message
    add_vertical_space(1)
    
    st.write("👶 Please upload an audio file to start!")

# Add spacing before footer
add_vertical_space(3)

# Footer
st.markdown('<div class="footer">Created with 💖 by HeyBaby.ai</div>', unsafe_allow_html=True)
