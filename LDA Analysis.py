#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)


# In[2]:


import nltk
nltk.download('wordnet')


# In[3]:


'''
Write a function to perform the pre processing steps on the entire dataset
'''
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return text #stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
            
    return result


# In[9]:


data = pd.read_csv("filename".csv", encoding = 'latin1')
data = data.dropna()
print(data.isnull().sum())
data.info()


# In[10]:


#data = data[~data.Summary.str.contains('Monitoring',case=False)]
#data = data[~data.Summary.str.contains('Account', case=False)]
#data = data[~data.Summary.str.contains('Event',case=False)]


# In[11]:


data.head(20)


# In[12]:


corpus = data.Description.tolist()


# In[13]:


corpus


# In[14]:


corpus[12]


# In[15]:


processed_docs = []
for word in corpus:
    processed_docs.append(preprocess(str(word)))


# In[17]:


processed_docs[60]


# In[18]:


'''
Create a dictionary from 'processed_docs' containing the number of times a word appears 
in the training set using gensim.corpora.Dictionary and call it 'dictionary'
'''
dictionary = gensim.corpora.Dictionary(processed_docs)


# In[36]:


print(dictionary)


# In[37]:


'''
Checking dictionary created
'''
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break


# In[38]:


'''
Create the Bag-of-words model for each document i.e for each document we create a dictionary reporting how many
words and how many times those words appear. Save this to 'bow_corpus'
'''
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


# In[39]:


'''
Preview BOW for our sample preprocessed document
'''
document_num = 2
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))


# In[41]:


bow_doc_x[13]


# In[42]:


# LDA mono-core -- fallback code in case LdaMulticore throws an error on your machine
# lda_model = gensim.models.LdaModel(bow_corpus, 
#                                    num_topics = 10, 
#                                    id2word = dictionary,                                    
#                                    passes = 50)

# LDA multicore 
'''
Train your lda model using gensim.models.LdaMulticore and save it to 'lda_model'
'''
# TODO
lda_model =  gensim.models.LdaModel(bow_corpus, 
                                   num_topics = 50, 
                                   id2word = dictionary,                                    
                                   passes = 20,
                                   alpha='auto',
                                   eta = 0.000005)


# In[43]:



'''
For each topic, we will explore the words occuring in that topic and its relative weight
'''
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")


# In[38]:


print(lda_model)


# In[ ]:




