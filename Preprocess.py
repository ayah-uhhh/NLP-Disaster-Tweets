# coding: utf-8

# # Import the Dataset
import pandas as pd

import re

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
#sample_submission = pd.read_csv('dataset/sample_submission.csv')


# # Cleaning the Text

def clean_data(dataframe):
    # Convert location and text to lowercase
    dataframe['location'] = dataframe['location'].str.lower()
    dataframe['text'] = dataframe['text'].str.lower()
    dataframe['keyword'] = dataframe['keyword'].str.lower()
    
    # Remove the extra comma in rows 1-44
    dataframe.loc[1:44, 'keyword'] = dataframe.loc[1:44, 'keyword'].str.rstrip(',')

    # Remove URLs, numbers, and non-alphanumeric characters
    dataframe['text'] = dataframe['text'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))
    dataframe['location'] = dataframe['location'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))
    dataframe['keyword'] = dataframe['keyword'].apply(lambda x: re.sub(r'http[s]?://\S+|[^A-Za-z ]|\d+', ' ', str(x)))

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

# Save the cleaned data as CSV
df_train.to_csv('dataset/cleaned_train.csv', index=False)
df_test.to_csv('dataset/cleaned_test.csv', index=False)