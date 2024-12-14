from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('../Data/raw_analyst_ratings.csv')


required_columns = ['headline', 'url', 'publisher', 'date', 'stock']
if not all(col in data.columns for col in required_columns):
    raise ValueError("Dataset is missing required columns")


def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['sentiment'] = data['headline'].apply(analyze_sentiment)


sentiment_counts = data['sentiment'].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)


plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Headlines')
plt.xlabel('Sentiment')
plt.ylabel('Number of Articles')
plt.xticks(rotation=0)
plt.show()


vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
word_count_matrix = vectorizer.fit_transform(data['headline'])


keywords = vectorizer.get_feature_names_out()
word_freq = word_count_matrix.sum(axis=0).A1
keyword_freq = pd.DataFrame({'keyword': keywords, 'frequency': word_freq})
keyword_freq = keyword_freq.sort_values(by='frequency', ascending=False)

print("Top Keywords:")
print(keyword_freq.head(10))


wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
    dict(zip(keyword_freq['keyword'], keyword_freq['frequency']))
)


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Headlines')
plt.show()
