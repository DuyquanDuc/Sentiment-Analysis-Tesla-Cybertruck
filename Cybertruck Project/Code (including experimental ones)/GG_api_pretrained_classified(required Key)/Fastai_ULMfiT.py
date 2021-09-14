#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastai.text.all import *


# In[ ]:


#read file
path = untar_data(URLs.IMDB)
path.ls()


# In[ ]:


(path/'train').ls()


# In[ ]:


dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')


# In[ ]:


dls.show_batch()


# In[ ]:


learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)


# In[ ]:


learn.fine_tune(4, 1e-2)


# In[ ]:


learn.fine_tune(4, 1e-2)


# In[ ]:


learn.show_results()


# In[ ]:


learn.predict("I really liked that movie!")

