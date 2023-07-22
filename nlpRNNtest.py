
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
from keras import layers
import time
import csv
"""
    Run the RNN model for binary classification. Blind, no labels.
    
    - dataset options: 'test.csv' , 'cleaned_test.csv' , 'cleaned_test_stop.csv'
    
    If you want to change the parameters, you must first run nlpRNN with your prefered settings. 
    This code will load the model from nlpRNN
"""
def nlp_rnn_unlabeled(dataset='test.csv', dropout_rate=0.5, batch_size=64, input_shape=(10,1)):
    start_time = time.time()
    dataset_path = f'dataset/{dataset}'
    train_data = pd.read_csv(dataset_path)

    # Extract X from train_data
    X = train_data['text'].values

    # Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)

    # Pad the sequences
    X_padded = pad_sequences(X_sequences, maxlen=input_shape[0])

    print("Loading pre-trained model...")
    model = tf.keras.models.load_model('saved_rnn_model.h5')

    print("Model summary:")
    model.summary()

    # Classify the data
    predictions = model.predict(X_padded, batch_size=batch_size)

    # Prepare the results for writing to CSV
    results = [(train_data['id'].values[i], int(round(prediction[0]))) for i, prediction in enumerate(predictions)]

    # Write the results to a CSV file
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'target'])
        writer.writerows(results)

    elapsed_time = time.time() - start_time

    return elapsed_time


nlp_rnn_unlabeled()
