import streamlit as st
import string
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

wordnetlemmatizer = WordNetLemmatizer()

def transformation(col):
    new_text = []
    text = col.lower()
    text = word_tokenize(text)
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation:
            new_text.append(i)
    text = new_text[:]
    new_text.clear()
    for i in text:
        new_text.append(wordnetlemmatizer.lemmatize(i))
    return " ".join(new_text)


vectorizer = joblib.load('TfIdf.joblib')
model = joblib.load('BernoulliNB.joblib')

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transformation(input_sms)
    # 2. vectorize
    vector_input = vectorizer.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")