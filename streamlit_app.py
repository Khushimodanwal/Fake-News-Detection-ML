import streamlit as st
import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detection")

news = st.text_area("Enter News Here")

if st.button("Check News"):

    if news:

        news_vector = vectorizer.transform([news])

        result = model.predict(news_vector)

        st.success("Prediction: " + result[0])