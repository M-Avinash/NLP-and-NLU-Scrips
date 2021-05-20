#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import nltk
from nltk import ngrams,FreqDist
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


data = pd.read_csv("Summary.csv" , encoding = 'latin1')
data = data.dropna()
print(data.isnull().sum())
data.info()


# In[6]:


data = data[~data.Summary.str.contains('Monitoring',case=False)]
data = data[~data.Summary.str.contains('Account', case=False)]
data = data[~data.Summary.str.contains('Event','Fault')]
data = data[~data.Summary.str.contains('access', case=False)]
data = data[~data.Summary.str.contains('Outlook',case=False)]
data = data[~data.Summary.str.contains('Password',case=False)]
data = data[~data.Summary.str.contains('Fault', case=False)]
data = data[~data.Summary.str.contains('Unable','connect')]
data = data[~data.Summary.str.contains('Master', case=False)]
data = data[~data.Summary.str.contains('Request',case=False)]
data = data[~data.Summary.str.contains('user',case=False)]


# In[7]:


data.info()


# In[8]:


corpus = data.Summary.tolist()


# In[9]:


with open("corpus.txt", "w") as output:
    output.write(str(corpus))


# In[10]:


words = ' '.join(corpus)


# In[11]:


wordcloud = WordCloud(max_font_size=50, max_words=50, background_color="white",width=600, height=300).generate(words)
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
plt.figure(figsize=(10,5))
plt.imshow(wordcloud)


# In[16]:


nltk.download('stopwords')
nltk.download('punkt')
#stop_words = stopwords.words('english')
stop_words = set(stopwords.words('english')) 
  
word_tokens = word_tokenize(words) 
  
filtered_sentence = [w for w in word_tokens if not w in stop_words] 
  
filtered_sentence = [] 
  
for w in word_tokens: 
    if w not in stop_words: 
        filtered_sentence.append(w) 

#print(filtered_sentence) 


# In[17]:


Counter(filtered_sentence).most_common(1000)


# In[21]:


grams2 =[]
n = 2
text = ngrams(words.split(),n)
for grams in text:
    #print(grams) 
    grams2.append(grams)   
    


# In[22]:


fdist = nltk.FreqDist(grams2)
fdist


# In[23]:


fdist.most_common(200)


# In[ ]:


df= data[data.Summary.str.contains('master incident',case=False)]
df.to_csv('master_incident.csv')


# In[ ]:


data.to_csv('corpus.txt')


# In[ ]:




