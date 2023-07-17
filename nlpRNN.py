import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
from keras import layers
import time
from sklearn.model_selection import train_test_split

#Understand RNN performance
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

#Import Data
# test_data = pd.read_csv('dataset/cleaned_test.csv')
# train_data = pd.read_csv('dataset/cleaned_train.csv')

def nlp_rnn(optimizer='adam', units=128, input_shape=(10,1), show_chart=True, save=False, epochs=200, batch_size=32):
    """Import Data"""
    start_time = time.time()
    train_data = pd.read_csv('dataset/cleaned_train.csv')

    # Extract X and Y from train_data
    X = train_data[ 'text'].values
    Y = train_data['target'].values
# Tokenize the text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_sequences = tokenizer.texts_to_sequences(X)

    # Pad the sequences
    X_padded = pad_sequences(X_sequences, maxlen=input_shape[0])

    print("Splitting data...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded, Y, test_size=0.2, shuffle=False)
    # xtrain shape: (6090, 10)
    # xTest shape: (1523, 10)
    
    print("Building model...")
    start_time = time.time()
    # Define the model
    model = keras.Sequential()

    # Bidirectional LSTM layers
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(units, return_sequences=True), input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Bidirectional(layers.LSTM(units)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=optimizer, metrics=['accuracy'])

    print("Built model. {}".format(time.time()-start_time))

    #model.summary()
    # Train the model
    print("Training model...")
    history = model.fit(X_train, y_train, epochs, batch_size,
                        validation_data=(X_test, y_test))
    print("Model trained.{}".format(time.time()-start_time),"seconds")

    elapsed_time = time.time()-start_time

    if (show_chart):
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    if save:
        print("Saving model...")
        start_time = time.time()
        model.save('ps_rnn_model.h5')
        print("Saved model {}".format(time.time()-start_time))

    print('Accuracy:', (accuracy*100),"%")
    #print("Loss:", loss, "%")

    return [(optimizer, units, input_shape), elapsed_time, loss, accuracy, model]


nlp_rnn()
plot_graphs()
