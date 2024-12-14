import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data



data = load_data('../Data/raw_analyst_ratings.csv')


data['headline_length'] = data['headline'].str.len()


headline_stats = data['headline_length'].describe()
print("Headline Length Statistics:")
print(headline_stats)


plt.figure(figsize=(10, 6))
sns.histplot(data['headline_length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Headline Lengths')
plt.xlabel('Headline Length (characters)')
plt.ylabel('Frequency')
plt.show()


publisher_counts = data['publisher'].value_counts()
print("Articles per Publisher:")
print(publisher_counts)


plt.figure(figsize=(12, 6))
publisher_counts.head(10).plot(kind='bar', color='green')
plt.title('Top 10 Publishers by Number of Articles')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()




data['publication_date'] = data['date'].dt.date


articles_by_day = data['publication_date'].value_counts().sort_index()


plt.figure(figsize=(14, 6))
articles_by_day.plot(kind='line', color='purple')
plt.title('Articles Published Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.show()


data['day_of_week'] = data['date'].dt.day_name()
day_of_week_counts = data['day_of_week'].value_counts()


plt.figure(figsize=(10, 6))
day_of_week_counts.plot(kind='bar', color='orange')
plt.title('Articles Published by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.show()


data['hour'] = data['date'].dt.hour
hourly_counts = data['hour'].value_counts().sort_index()


plt.figure(figsize=(10, 6))
hourly_counts.plot(kind='bar', color='cyan')
plt.title('Articles Published by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Articles')
plt.xticks(rotation=0)
plt.show()




publication_frequency = data.groupby(data['publication_date']).size()


plt.figure(figsize=(14, 6))
plt.plot(publication_frequency, marker='o', color='blue')
plt.title('Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()


threshold = publication_frequency.mean() + 2 * publication_frequency.std()
spikes = publication_frequency[publication_frequency > threshold]
print("Significant Spikes in Publication Frequency:")
print(spikes)


hourly_distribution = data.groupby('hour').size()


plt.figure(figsize=(10, 6))
hourly_distribution.plot(kind='line', marker='o', color='purple')
plt.title('Publishing Trends by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()


print("Analysis Complete. Visualizations and statistics generated.")
