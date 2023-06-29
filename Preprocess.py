# coding: utf-8

# # Import the Dataset
import pandas as pd

import re

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
#sample_submission = pd.read_csv('dataset/sample_submission.csv')


# # Cleaning the Text

# Convert location and text to Lowercase for train and test dataset
#TRAIN
df_train['location'] = df_train['location'].str.lower()
df_train['text'] = df_train['text'].str.lower()

#TEST
df_test['location'] = df_test['location'].str.lower()
df_test['text'] = df_test['text'].str.lower()

def clean_data(dataframe):
    # Convert location and text to lowercase
    dataframe['location'] = dataframe['location'].str.lower()
    dataframe['text'] = dataframe['text'].str.lower()

    # Remove URLs, numbers, and non-alphanumeric characters
    dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))
    dataframe['location'] = dataframe['location'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))

    # Load stopwords into a set
    stopwords = set(pd.read_csv('dataset/NLTKs_list_of_english_stopwords', sep=' ', header=None, names=['stopwords'])['stopwords'])
    
    # Remove stopwords
    dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    dataframe['location'] = dataframe['location'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    
# Clean the text
clean_data(df_train)
clean_data(df_test)

sample = df_test.sample(n=5)
print(sample)
