
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
    Run the RNN model for binary classification. Blind, no labels. Make sure parameters match the ideal parameters form nlpRNN
    
    - dataset options: 'test.csv' , 'cleaned_test.csv' , 'cleaned_test_stop.csv'
"""
def nlp_rnn_unlabeled(dataset='test.csv', optimizer='adam', units=256, input_shape=(10, 1),
                      dropout_rate=0.5, show_chart=False, save=False, epochs=40, batch_size=64):
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

    print("Building model...")
    start_time = time.time()
    # Define the model
    model = keras.Sequential()

    # Bidirectional LSTM layers
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Bidirectional(layers.LSTM(units)))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer, metrics=['accuracy'])

    print("Built model. {}".format(time.time() - start_time))

    model.summary()

    # Classify the data
    predictions = model.predict(X_padded, batch_size=batch_size)

    # Prepare the results for writing to CSV
    results = [(i, int(round(prediction[0]))) for i, prediction in enumerate(predictions)]

    # Write the results to a CSV file
    with open('results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'target'])
        writer.writerows(results)

    if save:
        print("Saving model...")
        start_time = time.time()
        model.save('ps_rnn_model.h5')
        print("Saved model {}".format(time.time() - start_time))

    elapsed_time = time.time() - start_time

    return [(optimizer, units, input_shape), elapsed_time, model]


nlp_rnn_unlabeled()