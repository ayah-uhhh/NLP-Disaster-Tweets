import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Import the dataset
df_test = pd.read_csv('dataset/cleaned_test.csv')

# Distribution of Target Variable
target_counts = df_test['target'].value_counts()
plt.bar(target_counts.index, target_counts.values)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Distribution of Target Variable')
plt.show()

# Word Frequency Analysis
text = ' '.join(df_test['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Frequency Analysis')
plt.show()