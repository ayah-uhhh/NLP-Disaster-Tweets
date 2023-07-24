# -*- coding: utf-8 -*-
"""BERTLarge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kZ-tFMYQXOzWLrHSG6K_XIursUPXC8gM
"""

!pip install transformers

!pip install wordcloud

"""##Import Libraries"""

import os
import re
import nltk
import keras.backend as K
import random
import string
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud, ImageColorGenerator ,STOPWORDS
from transformers import AutoTokenizer , TFAutoModel
from sklearn.metrics import confusion_matrix , classification_report
os.environ["WANDB_DISABLED"] = "true"

import warnings
warnings.filterwarnings('ignore')

"""## Function for cleaning dataset and converting text into tokens"""

def clean_dataset(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '',text)     #Removes Websites
    text = re.sub(r'<.*?>' ,'', text)                   #Removes HTML tags
    text = re.sub(r'\x89\S+' , ' ', text)                #Removes string starting from \x89
    text = re.sub('\w*\d\w*', '', text)                  #Removes numbers
    text = re.sub(r'[^\w\s]','',text)                    #Removes Punctuations
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  #emoticons
                               u"\U0001F300-\U0001F5FF"  #symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  #transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  #flags (iOS)
                               u"\U00002500-\U00002BEF"  #chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"                  #dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text

def Convert(string):
    li = list(string.lower().split(" "))
    return li

"""##Hyperparameters"""

class config:
    TRAIN_PATH = "dataset/train.csv"
    TEST_PATH = "dataset/test.csv"
    MAX_LEN = 36
    LOWER_CASE = True
    RANDOM_STATE = 12
    TEST_SIZE = 0.2
    NUM_LABELS = 1
    BATCH_SIZE = 128
    LEARNING_RATE = 5e-5
    EPOCHS = 10
    WEIGTH_DECAY = 0.01
    #DEVICE = "cuda"

"""##Exploratory Data Analysis"""

df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')
df_train.head(5)

"""## Random Examples from the training dataset"""

# Generates a random integer between 0 and len(df_train)-5
random_index = random.randint(0,len(df_train)-5)

# Loop to ensure that only the 'text' and 'target' columns are selected for processing
for row in df_train[["text","target"]][random_index:random_index +5].itertuples():
  _,text,target = row

  # If 'target' > 0 print in green else for 'target' <= 0 print in red
  print(f"\033[2mTarget :" , f"\033[92m{target} (real disaster)" if target > 0 else f"\033[91m {target} (not a real disaster)")
  print(f"\033[114mText :{text}")
  s = clean_dataset(text)
  print(f"\033[30mCleaned Text: {s}\n")
  print("-------------------------------------------------------\n")

# Apply clean_dataset function to each entry in the "text" column of the "df_train".
df_train["text"] = df_train["text"].map(clean_dataset)

# Apply convert function to each entry in the "text" column of the "df_train" to convert text to tokens.
df_train["tokens"] = df_train["text"].map(Convert)

# Apply clean_dataset and convert fuction to "df_test".
df_test["text"] = df_test["text"].map(clean_dataset)
df_test["tokens"] = df_test["text"].map(Convert)

import nltk
nltk.download('stopwords')

# Returns a list of common English stopwords.
stop_words = nltk.corpus.stopwords.words("english")

# A list containing the "text" values from the rows in "df_train" where the "target" column has a value of 1 (disaster tweets).
disaster_tweets = df_train[df_train["target"]==1]["text"].tolist()

# A list containing the "text" values from the rows in "df_train" where the "target" column has a value of 0 (non-disaster tweets).
non_disaster_tweets = df_train[df_train["target"]==0]["text"].to_list()

# "disaster_tweets_df" DataFrame will have two columns namely "text" and "tokens"
disaster_tweets_df = pd.DataFrame(disaster_tweets , columns = ["text"])
disaster_tweets_df["tokens"] = disaster_tweets_df["text"].map(Convert)
#disaster_tweets_df.head(4)

# "non_disaster_tweets_df" DataFrame will have two columns namely "text" and "tokens"
non_disaster_tweets_df = pd.DataFrame(non_disaster_tweets , columns = ["text"])
non_disaster_tweets_df["tokens"] = non_disaster_tweets_df["text"].map(Convert)
#non_disaster_tweets_df.head(4)

# "disaster_allwords" list will contain all the tokenized words from the "tokens" column of
#  the "disaster_tweets_df" DataFrame, excluding common English stopwords.
disaster_words = disaster_tweets_df["tokens"]
disaster_allwords = []
for wordlist in disaster_words:
    for disaster_word in wordlist:
        if disaster_word not in stop_words:
            disaster_allwords.append(disaster_word)

# "non_disaster_allwords" list will contain all the tokenized words from the "tokens" column of
#  the "non_disaster_tweets_df" DataFrame, excluding common English stopwords.
non_disaster_words = non_disaster_tweets_df["tokens"]
non_disaster_allwords = []
for wordlist in non_disaster_words:
    for non_disaster_word in wordlist:
        if non_disaster_word not in stop_words:
            non_disaster_allwords.append(non_disaster_word)

# FreqDist calculates the word frequencies in the "disaster_allwords" list of the 2500 most common words.
mostcommon = FreqDist(disaster_allwords).most_common(2500)

# wordcloud for visualization of disaster tweets related commom words
wordcloud = WordCloud(width=1800, height=1000, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(20,5), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Disaster Tweets common words', fontsize=35)
plt.tight_layout(pad=1)
plt.show()

# FreqDist calculates the word frequencies in the "non_disaster_allwords" list of the 2500 most common words.
mostcommon = FreqDist(non_disaster_allwords).most_common(2500)

# wordcloud for visualization of non-disaster related common words
wordcloud = WordCloud(width=1800, height=1000, background_color='white').generate(str(mostcommon))
fig = plt.figure(figsize=(20,5), facecolor='white')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title('Non-Disaster Tweets commmon words', fontsize=30)
plt.tight_layout(pad=1)
plt.show()

# Drop columns "id" , "keyword" , "location" , "tokens" from df_train and df_test
test_ids = df_test["id"]
df_train = df_train.drop(["id" , "keyword" , "location" , "tokens"] , axis = 1)
df_test = df_test.drop(["id" , "keyword" , "location" , "tokens"], axis =1 )

# Name of the BERT Model.
# Hugging Face Transformers library to tokenize text data using the BERT model
MODEL_2 = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_2 , do_lower_case = config.LOWER_CASE , max_length = config.MAX_LEN )

x_train = tokenizer(
        text = df_train["text"].tolist(),
        add_special_tokens = True,            #indicates that special tokens like [CLS] (start of sequence) and [SEP] (separator) tokens should be added to the input
        max_length = config.MAX_LEN,          #sets the maximum length of the tokenized sequences
        truncation = True,                    #truncates the sequences to fit within the maximum length
        padding = True,                       #adds padding tokens to the sequences to make them all the same length
        return_tensors = "tf",                #return TensorFlow tensors as output
        return_token_type_ids = False,        #BERT doesn't use token type IDs
        return_attention_mask = True,         #generates attention masks, which indicate the actual tokens vs. padding tokens
        verbose = True                        #displays progress during tokenization
        )

x_test = tokenizer(
        text = df_test["text"].tolist(),
        add_special_tokens = True,
        max_length = config.MAX_LEN,
        truncation = True,
        padding = True,
        return_tensors = "tf",
        return_token_type_ids = False,
        return_attention_mask = True,
        verbose = True
        )

# BERT model loaded as a TensorFlow model
bert_large = TFAutoModel.from_pretrained(MODEL_2)

# Defines the inputs for the tokenized sequences (input_ids) and corresponding attention masks (input_mask)
input_ids = tf.keras.layers.Input(shape = (config.MAX_LEN,) , dtype = tf.int32 , name = "input_ids")
input_mask = tf.keras.layers.Input(shape = (config.MAX_LEN,) , dtype = tf.int32 , name = "attention_mask")

# Tokenized input_ids & their input_mask is loaded on  BERT model
embeddings = bert_large(input_ids , attention_mask = input_mask)[1]             #[1] index extracts the embeddings from the BERT model's output

# Dropout layer to reduce overfitting
x = tf.keras.layers.Dropout(0.3)(embeddings)                                    # 30% of the neuron outputs will be randomly be 0 during training
x = tf.keras.layers.Dense(128 , activation = "relu")(x)                         # First dense layer has 128 units and ReLU activation
x = tf.keras.layers.Dropout(0.2)(x)                                             # Dropout layer to avoid overfitting
x = tf.keras.layers.Dense(32 , activation = "relu")(x)                          # Second dense layer has 32 units and ReLU activation
output = tf.keras.layers.Dense(config.NUM_LABELS , activation = "sigmoid")(x)   # Output Layer uses Sigmoid as binary classification

model_2 = tf.keras.Model(inputs = [input_ids , input_mask] , outputs = output)

print("Transformer Layer Unfreezed!!")
model_2.layers[2].trainable = True
model_2.summary()

# Check if the directory "./weights/bert_large_weights" exists or not using the os.path.isdir() function
# If not, creates it using os.makedirs() to store the BERT model's weights
if  os.path.isdir("./weights/bert_large_weights") is None:
          os.makedirs("./weights/bert_large_weights")

checkpoint_filepath_bert_large  = "./weights/bert_large_weights"
checkpoint_callback_bert_large = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_filepath_bert_large,                                             #directory path where weights will be saved
    save_weights_only=True,                                                     #only the model's weights will be saved
    monitor='val_accuracy',                                                     #save the best weights i.e. the validation accuracy
    mode='auto',                                                                #direction of improvement for the monitored metric
    save_best_only=True)                                                        #weights corresponding to the best validation performance will be saved

model_2.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
             optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = config.LEARNING_RATE , epsilon = 1e-8 , decay  =config.WEIGTH_DECAY , clipnorm = 1.0),
             metrics = ["accuracy"])

bert_large_history  = model_2.fit(x = {"input_ids": x_train["input_ids"] , "attention_mask" : x_train["attention_mask"]},
                y = df_train["target"] ,
                epochs = config.EPOCHS ,
                validation_split = 0.2,
                batch_size = 32 , callbacks = [checkpoint_callback_bert_large])

# Saving the model
model_2.save('/content/drive/MyDrive/DataMiningModelLarge')

import os
saved_model_path = '/content/drive/MyDrive/DataMiningModelLarge'
os.listdir(saved_model_path)

model_2.load_weights(checkpoint_filepath_bert_large)

bert_large_hist_df = pd.DataFrame(bert_large_history.history , columns = ['loss', 'accuracy', 'val_loss', 'val_accuracy'])

fig = plt.figure(figsize = (5,4))
plt.plot(np.arange(len(bert_large_hist_df["accuracy"]))+1,bert_large_hist_df["accuracy"],'r-',np.arange(len(bert_large_hist_df["val_accuracy"]))+1,bert_large_hist_df["val_accuracy"],'b-',linewidth=2)
plt.legend(["Accuracy" , "Validation Accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
#plt.title("BERT-Large accuracy plot")

fig = plt.figure(figsize = (5,4))
plt.plot(np.arange(len(bert_large_hist_df["loss"]))+1,bert_large_hist_df["loss"],'r-',np.arange(len(bert_large_hist_df["val_loss"]))+1,bert_large_hist_df["val_loss"],'b-',linewidth=2)
plt.legend(["Loss" , "Validation Loss"])
plt.xlabel("Epochs")
plt.ylabel("Loss")
#plt.title("BERT-Large Loss plot")

y_pred = model_2.predict({"input_ids" : x_train["input_ids"] ,"attention_mask" : x_train["attention_mask"]})
y_pred = np.where(y_pred > 0.5 , 1,0)
y_test = df_train["target"]
CLASS_LABELS = ["Disaster Tweet" , "Non Disaster Tweet"]
cm_data = confusion_matrix(y_test , y_pred)
cm = pd.DataFrame(cm_data , columns = CLASS_LABELS , index = CLASS_LABELS)
fig = px.imshow(img = cm_data ,
                x = CLASS_LABELS,
                y = CLASS_LABELS,
                aspect="auto" ,
                color_continuous_scale = "Sunset")
fig.update_xaxes(title="Predicted")
fig.update_yaxes(title = "Actual")
fig.update_layout(title = "Confusion Matrix",
                  template = "plotly_white",
                  title_x = 0.5)
fig.update_layout(
    yaxis_tickangle=-90  # Set the desired rotation angle (e.g., -45 degrees)
)
fig.show()

print(classification_report(y_test , y_pred))

model_2_pred_probs = model_2.predict({"input_ids" : x_test["input_ids"] ,"attention_mask" : x_test["attention_mask"]})
y_pred_2 = np.where(model_2_pred_probs > 0.5 , 1,0)
y_pred_2

bert_large_df=pd.DataFrame()
bert_large_df['id'] = test_ids
bert_large_df['target'] = y_pred_2

bert_large_df.to_csv('bert_large.csv',index = False)
bert_large_df["target"].value_counts()

result2 = {}
result2['id'] = list(test_ids)
result2['target'] = list(np.squeeze(y_pred_2))
print (result2)
pdObj = pd.DataFrame.from_dict(result2)
pdObj.to_csv("submission_bert_large.csv",index=False)

