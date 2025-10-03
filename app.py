import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.models import load_model

# Constants
MAX_LENGTH = 500
VOCAB_SIZE = 10000

# Load word index from local JSON file
import json
with open("imdb_word_index.json", "r") as f:
    word_index = json.load(f)

# Reverse word index to decode predictions (not used now, but can help in debugging)
reverse_word_index = {value: key for key, value in word_index.items()}

# Cache the model loading so it loads only once per app run/session
@st.cache_resource
def load_keras_model():
    return load_model('model.keras', compile=False)

model = load_keras_model()

# Preprocess user input
def preprocess_text(text_input, max_length=MAX_LENGTH):
    # Clean and tokenize input
    words = text.text_to_word_sequence(text_input)
    encoded = [word_index.get(word, 2) for word in words]  # 2 = OOV token
    # Clip indices that are >= VOCAB_SIZE to OOV token (2)
    encoded = [i if i < VOCAB_SIZE else 2 for i in encoded]
    padded = sequence.pad_sequences([encoded], maxlen=max_length)
    return padded

# Predict sentiment
def predict_sentiment(review_text):
    processed = preprocess_text(review_text)
    prediction = model.predict(processed, verbose=0)[0][0]

    # Reverted: 1 = positive, 0 = negative
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😞"
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    return sentiment, confidence


# ──────────────────────────────────────────────────────────────
# Streamlit App UI
# ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.markdown("Enter a movie review below and click **Analyze Sentiment** to see if the model thinks it's positive or negative.")

# Session state for sample input
if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

# Button to load a sample review
if st.button("Try a Sample Review"):
    st.session_state.sample_text = (
        "The movie was absolutely fantastic! The performances were top-notch and the storyline was gripping."
    )

# Text area with sample text loaded if button was clicked
user_input = st.text_area("📝 Your Review", height=150, value=st.session_state.sample_text)

# Analyze Sentiment
if st.button("Analyze Sentiment"):
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        st.success(f"**Prediction:** {sentiment}")
        st.metric("Confidence", f"{confidence:.2%}")

        # Progress bar (requires float in [0, 1])
        st.progress(confidence)
    else:
        st.warning("⚠️ Please enter a review before analyzing.")
