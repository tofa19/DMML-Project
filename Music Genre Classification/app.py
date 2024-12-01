import joblib
import streamlit as st

st.title("Music genre Classification:")

model = joblib.load('model.pkl')
encoder = joblib.load('le.pkl')

st.write("Enter the values for the features below:")

popularity = st.number_input("popularity: ")
acousticness = st.number_input("acousticness: ")
danceability = st.number_input("danceability: ")
instrumentalness = st.number_input("instrumentalness: ")
loudness = st.number_input("loudness: ")
mode = st.number_input("mode ")
speechiness = st.number_input("speechiness: ")
valence = st.number_input("valence ")
art_genre = st.number_input("art_genre: ")


def predict_class():
    values = [[popularity, acousticness, danceability, instrumentalness, loudness, mode, speechiness, valence, art_genre ]]
    predict = model.predict(values)
    st.write("Predicted class: ", predict)


st.button('Predict Class', on_click=predict_class)
