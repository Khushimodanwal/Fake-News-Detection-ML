import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

news = input("Enter News: ")

news_vector = vectorizer.transform([news])

result = model.predict(news_vector)

print("Prediction:", result[0])