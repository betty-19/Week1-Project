import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_data


data = load_data('../Data/raw_analyst_ratings.csv')





publisher_counts = data['publisher'].value_counts()
print("Top Publishers by Article Count:")
print(publisher_counts.head(10))


plt.figure(figsize=(10, 6))
publisher_counts.head(10).plot(kind='bar', color='skyblue')
plt.title('Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()


def extract_domain(email):
    if "@" in email:
        return email.split('@')[1]
    return "Unknown"


data['publisher_domain'] = data['publisher'].apply(extract_domain)


domain_counts = data['publisher_domain'].value_counts()
print("Top Domains by Article Count:")
print(domain_counts.head(10))


plt.figure(figsize=(10, 6))
domain_counts.head(10).plot(kind='bar', color='lightgreen')
plt.title('Top 10 Publisher Domains by Article Count')
plt.xlabel('Domain')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.show()


if 'headline' in data.columns:
    data['type_of_news'] = data['headline'].apply(
        lambda x: 'Earnings' if 'earnings' in x.lower() else
                  'Price Target' if 'price target' in x.lower() else
                  'Other'
    )

    type_counts = data['type_of_news'].value_counts()
    print("Type of News Reported:")
    print(type_counts)

  
    plt.figure(figsize=(8, 6))
    type_counts.plot(kind='bar', color=['orange', 'blue', 'gray'])
    plt.title('Type of News Reported')
    plt.xlabel('Type of News')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.grid(axis='y')
    plt.show()
else:
    print("No 'headline' column found in the dataset for type analysis.")
