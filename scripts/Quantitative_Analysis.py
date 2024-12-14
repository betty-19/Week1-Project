import pandas as pd
import talib
import matplotlib.pyplot as plt
import numpy as np
import os


def sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)


def sortino_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_deviation = np.std(downside_returns)
    return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else np.nan


data_folder = '../Data/yfinance_data'


all_data = []


for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        filepath = os.path.join(data_folder, file)
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df['Stock'] = file.split('.')[0]  
        all_data.append(df)


data = pd.concat(all_data, ignore_index=True)
data.sort_values(['Stock', 'Date'], inplace=True)


selected_stock = data[data['Stock'] == data['Stock'].unique()[0]].copy()


selected_stock.set_index('Date', inplace=True)


required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
if not all(col in selected_stock.columns for col in required_columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")


selected_stock.loc[:, 'SMA_50'] = talib.SMA(selected_stock['Close'], timeperiod=50)
selected_stock.loc[:, 'SMA_200'] = talib.SMA(selected_stock['Close'], timeperiod=200)
selected_stock.loc[:, 'RSI'] = talib.RSI(selected_stock['Close'], timeperiod=14)
selected_stock.loc[:, 'MACD'], selected_stock.loc[:, 'MACD_Signal'], _ = talib.MACD(
    selected_stock['Close'], fastperiod=12, slowperiod=26, signalperiod=9
)


returns = selected_stock['Close'].pct_change().dropna()
sharpe_ratio_value = sharpe_ratio(returns, risk_free_rate=0.01)
sortino_ratio_value = sortino_ratio(returns, risk_free_rate=0.01)

print(f"Sharpe Ratio: {sharpe_ratio_value:.2f}")
print(f"Sortino Ratio: {sortino_ratio_value:.2f}")


plt.figure(figsize=(12, 8))


plt.subplot(3, 1, 1)
plt.plot(selected_stock['Close'], label='Close Price', color='blue')
plt.plot(selected_stock['SMA_50'], label='50-day SMA', color='orange')
plt.plot(selected_stock['SMA_200'], label='200-day SMA', color='green')
plt.title(f"Stock Price and Moving Averages ({selected_stock['Stock'].iloc[0]})")
plt.legend()


plt.subplot(3, 1, 2)
plt.plot(selected_stock['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red', alpha=0.7)
plt.axhline(30, linestyle='--', color='green', alpha=0.7)
plt.title("Relative Strength Index (RSI)")
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(selected_stock['MACD'], label='MACD', color='red')
plt.plot(selected_stock['MACD_Signal'], label='Signal Line', color='black')
plt.title("MACD and Signal Line")
plt.legend()

plt.tight_layout()
plt.show()
