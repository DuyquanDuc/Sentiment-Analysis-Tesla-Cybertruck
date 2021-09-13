#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy 
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from spacy.tokens import DocBin
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
from collections import Counter
import json


# In[2]:


#read file
data = pd.read_csv(r'C:\Users\DuyQD\Downloads\pos.csv')
data


# In[3]:


#Drop unecessary column
data = data.drop(['col_0'], axis=1)


# In[4]:


# Text of all words in column Text

text = " ".join(review for review in data.Text.astype(str))
print ("There are {} words in the combination of all cells in column Text.".format(len(text)))


# In[5]:


#Prepare spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")
# Word tokenization
from spacy.lang.en import English


# In[6]:


# remove words that we want to exclude

stopwords = set(STOPWORDS)
stopwords.update(["Cybertruck", "elonmusk", "https", "Tesla https", "Tesla","production", 'Semi Roadster', "Elon", "think", "look","week", "video", "will", "one","see","year","truck","need","free", "next","make","model","first"])


# In[7]:


#spacy_stopwords
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# In[8]:


# Generate a word cloud image
wordcloud1 = WordCloud(stopwords=spacy_stopwords, background_color="white", width=800, height=400).generate(text)
wordcloud2 = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)


# In[9]:


#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)


# In[10]:


# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
#print(token_list)


# In[11]:


#Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))

#Printing first ten stop words:
#print('First ten stop words: %s' % list(spacy_stopwords)[:20])


# In[12]:


# filtering stop words
filtered_sent=[]
for word in my_doc:
    if word.is_stop==False:
        filtered_sent.append(word)
#print("Filtered Sentence:",filtered_sent)


# In[13]:


# finding lemma for each word
for word in filtered_sent:
    if word.lemma==False:
         filtered_sent.append(word)
#print("Filtered Sentence:",filtered_sent)


# In[14]:


# Display the generated image:
# the matplotlib way:
plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.show()


# In[15]:


plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.show()


# In[ ]:




