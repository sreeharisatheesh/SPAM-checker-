import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv(r'./spam.csv')
df.drop_duplicates(inplace=True)
df['category'] = df['Category'].replace(['spam', 'ham'], ['spam', 'not spam'])
mess = df['Message']
cat = df['category']

# Convert text to numerical data
cv = CountVectorizer(stop_words='english')

# Split dataset
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

# Fit the model
features = cv.fit_transform(mess_train)
model = MultinomialNB()
model.fit(features, cat_train)

# Prediction function
def predict(message):
    return model.predict(cv.transform([message]))[0]

# Streamlit UI
st.header('Spam Detection')
input_message = st.text_input('Enter the Email here!')
if st.button('Validate'):
    prediction = predict(input_message)
    st.markdown(prediction)