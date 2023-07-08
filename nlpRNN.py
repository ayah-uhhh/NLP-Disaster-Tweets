import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


#Understand RNN performance
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


#Import Data
test_data = pd.read_csv(cleaned_test.csv)

#bidirectional RNN: this allows context to be gained before and after words

model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5, 10))
)
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))

model.summary()


