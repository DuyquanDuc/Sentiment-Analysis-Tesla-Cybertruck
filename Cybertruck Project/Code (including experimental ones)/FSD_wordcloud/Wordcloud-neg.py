#!/usr/bin/env python
# coding: utf-8

# In[31]:


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


# In[32]:


#read file
data = pd.read_csv(r'C:\Users\DuyQD\Downloads\neg1.csv')
data


# In[37]:


#Drop unecessary column
#data = data.drop(['col_0'], axis=1)


# In[38]:


# Text of all words in column Text

text = " ".join(review for review in data.Text.astype(str))
print ("There are {} words in the combination of all cells in column Text.".format(len(text)))


# In[39]:


#Prepare spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")
# Word tokenization
from spacy.lang.en import English


# In[40]:


# remove words that we want to exclude

stopwords = set(STOPWORDS)
stopwords.update(["Cybertruck", "elonmusk", "https", "Tesla https", "Tesla","production", 'Semi Roadster', "Elon", "think", "look","week", "video", "will", "one","see","year","truck","need","free", "next","make","model","first"])


# In[41]:


#spacy_stopwords
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


# In[42]:


# Generate a word cloud image
wordcloud1 = WordCloud(stopwords=spacy_stopwords, background_color="white", width=800, height=400).generate(text)
wordcloud2 = WordCloud(stopwords=stopwords, background_color="white", width=800, height=400).generate(text)


# In[43]:


#  "nlp" Object is used to create documents with linguistic annotations.
my_doc = nlp(text)


# In[44]:


# Create list of word tokens
token_list = []
for token in my_doc:
    token_list.append(token.text)
#print(token_list)


# In[45]:


#Printing the total number of stop words:
print('Number of stop words: %d' % len(spacy_stopwords))

#Printing first ten stop words:
#print('First ten stop words: %s' % list(spacy_stopwords)[:20])


# In[46]:


# filtering stop words
filtered_sent=[]
for word in my_doc:
    if word.is_stop==False:
        filtered_sent.append(word)
#print("Filtered Sentence:",filtered_sent)


# In[47]:


# finding lemma for each word
for word in filtered_sent:
    if word.lemma==False:
         filtered_sent.append(word)
#print("Filtered Sentence:",filtered_sent)


# In[48]:


# Display the generated image:
# the matplotlib way:
plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud1, interpolation='bilinear')
plt.show()


# In[49]:


plt.axis("off")
plt.figure( figsize=(40,20))
plt.tight_layout(pad=0)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.show()


# In[ ]:




