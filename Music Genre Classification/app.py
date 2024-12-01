import joblib
import streamlit as st
import numpy as np

st.title("Music Genre Classification:")

# Load the trained model and label encoder
try:
    model = joblib.load('model.pkl')
    encoder = joblib.load('le.pkl')
    st.success("Model and Encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading model or encoder: {e}")
    st.stop()

st.write("Enter the values for the features below:")

# Input features
popularity = st.number_input("Popularity: ", value=0.0)
acousticness = st.number_input("Acousticness: ", value=0.0)
danceability = st.number_input("Danceability: ", value=0.0)
instrumentalness = st.number_input("Instrumentalness: ", value=0.0)
loudness = st.number_input("Loudness: ", value=0.0)
mode = st.number_input("Mode: ", value=0.0)
speechiness = st.number_input("Speechiness: ", value=0.0)
valence = st.number_input("Valence: ", value=0.0)
art_genre = st.number_input("Art Genre: ", value=0.0)

# Prediction function
def predict_class():
    try:
        # Prepare input
        values = [[popularity, acousticness, danceability, instrumentalness, loudness, mode, speechiness, valence, art_genre]]

        # Get model prediction
        prediction = model.predict(values)

        # Check if the prediction is encoded as probabilities
        if hasattr(model, "predict_proba"):
            prediction = np.argmax(model.predict_proba(values), axis=1)

        # Decode the encoded prediction to the original class label

        st.success(f"Predicted genre: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Button to trigger prediction
st.button('Predict Class', on_click=predict_class)
