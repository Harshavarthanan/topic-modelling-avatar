#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
dataset = pd.read_csv("/Users/harshavarthanan/Downloads/Avatar2.csv")
display(dataset.head())
print(dataset.shape)


# In[17]:


print(dataset['Comment'][351])
print(" ")


# In[18]:


def clean_text(text):
    text = text.lower().replace("'","").replace('[^\w\s]', ' ').replace(" \d+", " ").strip()
    return text
#test your function works
sample = dataset['Comment'][351]
clean_text(sample)


# In[36]:


dataset['Clean Comment'] = dataset['Comment'].apply(clean_text)
dataset.head()


# In[37]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
#stop_words_list = stopwords.words('english') + ['though', 'game', 'games']


# In[38]:


# initiate stopwords from nltk

stop_words = stopwords.words('english')
print(len(stop_words))
# add additional missing terms

stop_words.extend(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l','m','n','o','p','q','r','s','t', 'u', 'v', 'w', 'x', 'y', 'z', "about", "across", "after", "all", "also", "an", "and", "another", "added",
"any", "are", "as", "at", "basically", "be", "because", 'become', "been", "before", "being", "between","both", "but", "by","came","can","come","could","did","do","does","each","else","every","either","especially", "for","from","get","given","gets",
'give','gives',"got","goes","had","has","have","he","her","here","him","himself","his","how","if","in","into","is","it","its","just","lands","like","make","making", "made", "many","may","me","might","more","most","much","must","my","never","provide", 
"provides", "perhaps","no","now","of","on","only","or","other", "our","out","over","re","said","same","see","should","since","so","some","still","such","seeing", "see", "take","than","that","the","their","them","then","there",
"these","they","this","those","through","to","too","under","up","use","using","used", "underway", "very","want","was","way","we","well","were","what","when","where","which","while","whilst","who","will","with","would","you","your", 
'etc', 'via', 'eg', 'game', 'games' 'like']) 

# remove stopwords
print(len(stop_words))
#df[1] = df[1].apply(lambda x: [item for item in x if item not in stop_words])

#display(df.head(10))


# In[40]:


def superclean(text):
  #tokens = text.split(" ")
  text = text.lower().replace("'","").replace('[^\w\s]', ' ').replace(" \d+", " ").strip()
  tokens = nltk.word_tokenize(text)
  stop_tokens = [item for item in tokens if item not in stop_words]
  new_text = ' '.join(stop_tokens)
  return new_text
print(dataset['Comment'][351])
print("------")
print(superclean(dataset['Comment'][351]))


# In[41]:


dataset['Clean Comment'] = dataset['Comment'].apply(superclean)
dataset.head()
dataset["char_count"] = dataset['Clean Comment'].apply(len)
df = dataset.drop(dataset[dataset['char_count'] < 50].index)
#df = df.drop(df[df.score < 50].index)
#df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
print(df.shape)
print(dataset.shape)
display(df.head())


# In[42]:


from sklearn.feature_extraction.text import CountVectorizer
#Only include those words that appear in less than 80% of the document (max_df=0.8)
#Only include those words that ppear in at least 2 documents
count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
#.values Only the values in the DataFrame will be returned, the axes labels will be removed.
#The astype(‘U’) is telling numpy to convert the data to Unicode (essentially a string in python 3).
doc_term_matrix = count_vect.fit_transform(dataset['Clean Comment'].values.astype('U'))


# In[51]:


from sklearn.decomposition import LatentDirichletAllocation
#n_components = num. of topics
#random_state = Is just random
LDA = LatentDirichletAllocation(n_components=10, random_state=42)
LDA.fit(doc_term_matrix)


# In[52]:


for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[55]:


LDA_Advanced = LatentDirichletAllocation(n_components=5,
max_iter=10, 
learning_method='online',
random_state=100,
batch_size=128,
evaluate_every = -1,
n_jobs = -1 )
LDA_Advanced.fit(doc_term_matrix)


# In[57]:


for i,topic in enumerate(LDA_Advanced.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[53]:


topic_values = LDA_Advanced.transform(doc_term_matrix)
topic_values.shape
dataset['Topic'] = topic_values.argmax(axis=1)
dataset[16000:16010]


# In[54]:


file_name = 'TopicData.xlsx'
  
# saving the excel
dataset.to_excel(file_name)
print('DataFrame is written to Excel File successfully.')


# In[ ]:




