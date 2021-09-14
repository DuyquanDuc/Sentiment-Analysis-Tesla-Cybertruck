#!/usr/bin/env python
# coding: utf-8

# In[2]:


from google.cloud import language_v1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


df = pd.read_csv(r'C:\Users\DuyQD\Downloads\FDS.csv')
df.head()


# In[12]:


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
N, bins, patches = axs.hist(df['Sentiment_score'])
 
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


# In[13]:


df['labeled'] = np.where(df['Sentiment_score']> 0, 'positive', 
         (np.where(df['Sentiment_score'] < 0, 'negative', 'neutral')))
df['labeled'].value_counts()


# In[14]:


data = [50, 37, 24]
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




