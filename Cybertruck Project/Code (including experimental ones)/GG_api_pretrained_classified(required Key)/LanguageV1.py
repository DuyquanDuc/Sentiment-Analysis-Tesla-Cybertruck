#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.cloud import language_v1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def sample_analyze_sentiment(text_content):
    """
    Analyzing Sentiment in a String

    Args:
      text_content The text content to analyze
    """

    client = language_v1.LanguageServiceClient()

    # text_content = 'I am so happy and joyful.'

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    encoding_type = language_v1.EncodingType.UTF8

    response = client.analyze_sentiment(request = {'document': document, 'encoding_type': encoding_type})
    # Get overall sentiment of the input document
    return response.document_sentiment.score
    


# In[3]:


df = pd.read_csv('Final_tweets.csv')


# In[4]:


df['sentiment'] = df['Text'].apply(sample_analyze_sentiment)


# In[5]:


#Sentiment score between -1.0 (negative sentiment) and 1.0 (positive sentiment).
#A non-negative number in the [0, +inf) range, which represents the absolute magnitude of sentiment regardless of score (positive or negative).


# In[6]:


df.head()


# In[7]:


df.to_csv("Final_tweets_classified.csv")


# In[8]:


# Creating histogram
fig, axs = plt.subplots(1, 1,
                        figsize =(10, 7),
                        tight_layout = True)
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    axs.spines[s].set_visible(False)
 
# Remove x, y ticks
axs.xaxis.set_ticks_position('none')
axs.yaxis.set_ticks_position('none')
   
# Add padding between axes and labels
axs.xaxis.set_tick_params(pad = 5)
axs.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
axs.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.6)
 
# Add Text watermark
fig.text(0.3, 0.9, 'Value Frequency',
         fontsize = 18,
         color ='red',
         ha ='right',
         va ='top',
         alpha = 0.7)
 
# Creating histogram
N, bins, patches = axs.hist(df['sentiment'])
 
# Setting color
fracs = ((N**(1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())
 
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

legend = ['distribution']

# Adding extra features   
plt.xlabel("Sentiment-Score")
plt.ylabel("Frequency")
plt.legend(legend)
plt.title('Customized histogram')
 
# Show plot
plt.show()


# In[9]:


df['labeled'] = np.where(df['sentiment']> 0, 'positive', 
         (np.where(df['sentiment'] < 0, 'negative', 'neutral')))
df['labeled'].value_counts()


# In[10]:


data = [344, 434, 222]
label = ['Positive', 'Negative', 'Neutral']
# Creating explode data
explode = (0.1, 0.0, 0.2)
  
# Creating color parameters
colors = ( "darkgreen", "limegreen", "aquamarine")
  
# Wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "black" }
# Creating plot
fig, ax = plt.subplots(figsize =(13, 10))
wedges, autotexts, text = ax.pie(data,
                           autopct='%1.2f%%',
                           explode = explode, 
                           labels = label,
                           shadow = True,
                           colors = colors,
                           startangle = 90,
                           wedgeprops = wp,
                           textprops = dict(color ="indigo"))
  
# Adding legend
ax.legend(wedges, label,
          title ="Sentiment",
          loc ="center left",
          bbox_to_anchor =(1, 0, 0.5, 1))
  
plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Customizing pie chart")
  
# show plot
plt.show()


# In[ ]:




