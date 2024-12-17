import pandas as pd
import os
from textblob import TextBlob
from scipy.stats import pearsonr


news_data_path = '../Data/raw_analyst_ratings.csv'
stock_data_folder = '../Data/yfinance_data'


news_data = pd.read_csv(news_data_path)


news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce').dt.date # Convert to date only


def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity 


news_data['sentiment_score'] = news_data['headline'].apply(analyze_sentiment)


correlation_results = []


for stock_file in os.listdir(stock_data_folder):
    if stock_file.endswith('.csv'):
        stock_name = stock_file.split('_')[0] 
        stock_path = os.path.join(stock_data_folder, stock_file)

       
        stock_data = pd.read_csv(stock_path)
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

      
        stock_data['daily_return'] = stock_data['Adj Close'].pct_change()

       
        avg_sentiments = news_data.groupby('date')['sentiment_score'].mean().reset_index()
        avg_sentiments.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)

     
        merged_data = pd.merge(stock_data, avg_sentiments, left_on='Date', right_on='date', how='left')
        merged_data.drop(columns=['date'], inplace=True)

     
        merged_data.dropna(subset=['daily_return', 'avg_sentiment_score'], inplace=True)

    
        if not merged_data.empty:
            correlation, _ = pearsonr(merged_data['avg_sentiment_score'], merged_data['daily_return'])
            correlation_results.append((stock_name, correlation))
            print(f"Correlation between Sentiment and Daily Returns for {stock_name}: {correlation:.4f}")
        else:
            print(f"No data available for {stock_name} after merging.")


correlation_df = pd.DataFrame(correlation_results, columns=['Stock', 'Correlation'])
print("\nOverall Correlation Results:")
print(correlation_df)


correlation_df.to_csv('./sentiment_stock_correlation_results.csv', index=False)
