import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('../Data/raw_analyst_ratings.csv')


data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


data.dropna(subset=['date'], inplace=True)


data['publication_date'] = data['date'].dt.date


publication_trends = data.groupby('publication_date').size()
print("Publication Frequency Over Time:")
print(publication_trends)


plt.figure(figsize=(12, 6))
publication_trends.plot(kind='line', color='blue', linewidth=2)
plt.title('Publication Frequency Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.grid(True)
plt.show()


mean_publications = publication_trends.mean()
std_publications = publication_trends.std()
spike_threshold = mean_publications + 2 * std_publications


spikes = publication_trends[publication_trends > spike_threshold]
print("Significant Spikes in Publication Frequency:")
print(spikes)


plt.figure(figsize=(12, 6))
plt.plot(publication_trends.index, publication_trends, label='Publication Frequency', color='blue')
plt.scatter(spikes.index, spikes, color='red', label='Spikes', zorder=5)
plt.axhline(spike_threshold, color='orange', linestyle='--', label=f'Spike Threshold ({spike_threshold:.2f})')
plt.title('Publication Frequency with Spikes Highlighted')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.legend()
plt.grid(True)
plt.show()


data['hour'] = data['date'].dt.hour
hourly_distribution = data.groupby('hour').size()
print("Publishing Frequency by Hour:")
print(hourly_distribution)


plt.figure(figsize=(10, 6))
hourly_distribution.plot(kind='bar', color='purple')
plt.title('Publishing Frequency by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Articles')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()
