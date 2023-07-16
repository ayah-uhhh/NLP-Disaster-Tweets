# coding: utf-8

# # Import the Dataset
import pandas as pd

import re

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
#sample_submission = pd.read_csv('dataset/sample_submission.csv')


# # Cleaning the Text

def clean_data(dataframe):
    # Convert location and text to lowercase
    dataframe['location'] = dataframe['location'].str.lower()
    dataframe['text'] = dataframe['text'].str.lower()

    # Remove URLs, numbers, and non-alphanumeric characters
    dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))
    dataframe['location'] = dataframe['location'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))
                                                
    # Tokenization
    dataframe['text'] = dataframe['text'].apply(lambda x: word_tokenize(str(x)))

    # Remove stopwords
    stopwords=nltk.corpus.stopwords.words('english')
    dataframe['text'] = dataframe['text'].apply(lambda x: ' '.join([word for word in x if word not in stopwords])) # dataframe['location'] = dataframe['location'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    
    # Lemmatization (normalization)
    lemmatizer=WordNetLemmatizer()
    dataframe['text'] = dataframe['text'].apply(lambda x: lemmatizer.lemmatize(str(x)))
    
    # Remove nan
    dataframe['location'] = dataframe['text'].apply(lambda x: re.sub(r'nan',',', str(x)))

    

# Clean the text
clean_data(df_train)
clean_data(df_test)

sample = df_test.sample(n=5)
print(sample)

# Save the cleaned data as CSV
df_train.to_csv('dataset/cleaned_train.csv', index=False)
df_test.to_csv('dataset/cleaned_test.csv', index=False)