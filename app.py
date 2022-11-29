
# -*- coding: utf-8 -*-
"""
Created on 

@author: 
"""

##Dataset Link: https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data
# Import pandas for data handling
import pandas as pd
import pickle



# NLTK is our Natural-Language-Took-Kit
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Libraries for helping us with strings
import string
# Regular Expression Library
import re

# Import our text vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Import our classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


# Import some ML helper function
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report



# Import our metrics to evaluate our model
from sklearn import metrics


# Library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# You may need to download these from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stopwords = stopwords.words('english')

#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)
pickle_in = open("news_classifier.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def (title):
    
    """Let's identify.
    ---
    parameters:  
      - name: title
        in: query
        type: string
        required: true
     
    responses:
        200:
            description: The output values
        
        """
    vectorizer = TfidfVectorizer()
    new_text_vectorized = vectorizer.transform([title])
    # make a new prediction using our model and vectorized text
    model.predict(new_text_vectorized)
    # This makes your vocab matrix
    vectorizer.fit(X_train,y_train)
    # This transforms your documents into vectors.
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)

    print(X_train.shape, type(X))
    prediction=classifier.predict(title)
    print(prediction)
    return prediction



def main():
    st.title("News Identifier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit News Identifing ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    title = st.text_input("title","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(title)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
