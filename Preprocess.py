#!/usr/bin/env python
# coding: utf-8

# # Import the Dataset

# In[ ]:


import pandas as pd


# In[15]:


df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
#sample_submission = pd.read_csv('dataset/sample_submission.csv')


# In[20]:


df_train.head()


# # Cleaning the Text
# 

# Convert location and text to Lowercase for train and test dataset

# In[25]:


#TRAIN
df_train['location'] = df_train['location'].str.lower()
df_train['text'] = df_train['text'].str.lower()

#TEST
df_test['location'] = df_test['location'].str.lower()
df_test['text'] = df_test['text'].str.lower()


# In[26]:


df_test.head()


# Remove URL's from text column for train and test dataset

# In[35]:

#debug
#print('train before = \n',df_train.head(20))
#print('test before = \n',df_test.head(20))

def clean_data(dataframe):
#replace URL of a text
    dataframe['text'] = dataframe['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')

    # Begin section for removing stopwords
    #
    # Read in list of stopwords from external file
    # List of stopwords is from https://gist.github.com/sebleier/554280
    df_stopwords = pd.read_csv('dataset/NLTKs_list_of_english_stopwords', sep=' ', header=None, names=['stopwords'])

    #debug
    #print('st words = \n', df_stopwords.head())
    
    # Cycle through each of the stopwords and remove any which are found from the specified column in the dataframe
    # uses f-string regular expression to only match whole words
    for s_word in df_stopwords['stopwords']:
        dataframe['text'] = dataframe['text'].str.replace(f'\\b{s_word}\\b', '', regex=True)
    
    #debug
    #print('dataframe = \n',dataframe.head(20))
    
    #
    # End section for removing stopwords


clean_data(df_train)
clean_data(df_test)


# In[38]:


df_test.head(35)
#print(df_train.iloc[33])


# In[ ]:




