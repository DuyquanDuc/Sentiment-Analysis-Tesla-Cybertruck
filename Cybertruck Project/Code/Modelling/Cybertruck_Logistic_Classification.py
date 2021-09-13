#!/usr/bin/env python
# coding: utf-8

# In[1]:


import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np


# In[7]:


# Pandas read the csv file prepared in Dataiku
data = pd.read_csv(r'C:\Users\DuyQD\Downloads\Final_tweets_Classified.csv')
data['sentiment'] = np.where(data['sentiment'] >=0, 1, 0)
data.head()


# In[8]:


text = " ".join(review for review in data.Text.astype(str))
print ("There are {} words in the combination of all cells in column Text.".format(len(text)))


# In[9]:


spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
#Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))


# In[10]:


# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load("en_core_web_sm")
stopwords = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    #this line cause all data disappear: mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens


# In[11]:


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        # Cleaning Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

# Basic function to clean the text
def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()


# In[12]:


bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))


# In[13]:


tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)


# In[15]:


from sklearn.model_selection import train_test_split

X = data['Text'] # the features we want to analyze
ylabels = data['sentiment'] # the labels, or answers, we want to test against

X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.3)


# In[16]:


# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

# Create pipeline using Bag of Words
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', bow_vector),
                 ('classifier', classifier)])

# model generation
pipe.fit(X_train,y_train)


# In[17]:


from sklearn import metrics
# Predicting with a test dataset
predicted = pipe.predict(X_test)

# Model Accuracy
print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, predicted))
print("Logistic Regression Precision:",metrics.precision_score(y_test, predicted))
print("Logistic Regression Recall:",metrics.recall_score(y_test, predicted))


# In[ ]:




