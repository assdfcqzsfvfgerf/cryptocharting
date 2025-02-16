import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt', quiet=True)

def fetch_crypto_data(symbol, start_date, end_date):
    data = yf.download(f"{symbol}-USD", start=start_date, end=end_date)
    return data

def calculate_indicators(data):
    # Basic indicators
    data['SMA_20'] = SMAIndicator(data['Close'], window=20).sma_indicator()
    data['EMA_20'] = EMAIndicator(data['Close'], window=20).ema_indicator()
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    data['VWAP'] = VolumeWeightedAveragePrice(data['High'], data['Low'], data['Close'], data['Volume']).volume_weighted_average_price()

    # Advanced indicators
    bb = BollingerBands(data['Close'])
    data['BB_upper'] = bb.bollinger_hband()
    data['BB_lower'] = bb.bollinger_lband()
    
    # MACD
    data['EMA_12'] = EMAIndicator(data['Close'], window=12).ema_indicator()
    data['EMA_26'] = EMAIndicator(data['Close'], window=26).ema_indicator()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_signal'] = EMAIndicator(data['MACD'], window=9).ema_indicator()
    
    # ATR (Average True Range)
    data['TR'] = np.maximum(data['High'] - data['Low'], 
                            np.maximum(abs(data['High'] - data['Close'].shift()), 
                                       abs(data['Low'] - data['Close'].shift())))
    data['ATR'] = data['TR'].rolling(window=14).mean()
    
    # OBV (On-Balance Volume)
    data['OBV'] = OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()

    return data

def detect_patterns_ml(data, n_clusters=5):
    # Prepare features for clustering
    features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(scaled_features)

    # Identify pattern changes
    data['Pattern_Change'] = data['Cluster'].diff() != 0

    return data

def simple_pattern_detection(data, window=20):
    # Detect local maxima and minima
    data['Local_Max'] = data['High'].rolling(window=window, center=True).max() == data['High']
    data['Local_Min'] = data['Low'].rolling(window=window, center=True).min() == data['Low']
    
    # Combine with cluster-based pattern changes
    data['Pattern_Change'] = data['Pattern_Change'] | data['Local_Max'] | data['Local_Min']
    
    return data

def backtest_strategy(data, strategy_func):
    # Implement a simple backtesting framework
    data['Signal'] = strategy_func(data)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Signal'].shift(1) * data['Returns']
    
    # Calculate performance metrics
    cumulative_returns = (1 + data['Strategy_Returns']).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe_ratio = np.sqrt(252) * data['Strategy_Returns'].mean() / data['Strategy_Returns'].std()
    
    return {
        'Cumulative_Returns': cumulative_returns,
        'Total_Return': total_return,
        'Sharpe_Ratio': sharpe_ratio
    }

def simple_moving_average_crossover(data):
    # Example strategy: Buy when short-term SMA crosses above long-term SMA, sell when it crosses below
    data['SMA_short'] = data['Close'].rolling(window=10).mean()
    data['SMA_long'] = data['Close'].rolling(window=50).mean()
    return np.where(data['SMA_short'] > data['SMA_long'], 1, 0)

def fetch_news_sentiment(symbol, num_articles=5):
    # Fetch recent news articles
    url = f"https://cryptonews.com/news/bitcoin-news/"  # Replace with an appropriate news source
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('div', class_='cn-tile article')

    sentiments = []
    for article in articles[:num_articles]:
        title = article.find('h4', class_='cn-tile-header').text
        blob = TextBlob(title)
        sentiment = blob.sentiment.polarity
        sentiments.append(sentiment)

    return np.mean(sentiments)

def calculate_portfolio_value(holdings, current_prices):
    return sum(holdings[symbol] * current_prices[symbol] for symbol in holdings)

def create_candlestick_chart(data, symbol, indicators, volume_profile, patterns):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ), row=1, col=1)

    # Add indicators
    for indicator in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data[indicator],
            name=indicator,
            line=dict(color=next(fig._color_cycle))
        ), row=1, col=1)

    # Add volume profile
    if volume_profile is not None:
        fig.add_trace(go.Bar(
            x=volume_profile.values,
            y=volume_profile.index.mid,
            orientation='h',
            name='Volume Profile',
            marker=dict(color='rgba(0,0,255,0.2)'),
            showlegend=False
        ), row=1, col=1)

    # Add detected patterns
    pattern_changes = data[data['Pattern_Change']]
    fig.add_trace(go.Scatter(
        x=pattern_changes.index,
        y=pattern_changes['Close'],
        mode='markers',
        marker=dict(symbol='star', size=10, color='red'),
        name='Pattern Change'
    ), row=1, col=1)

    # Add volume bars
    colors = ['green' if data['Close'][i] > data['Open'][i] else 'red' for i in range(len(data))]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker=dict(color=colors)
    ), row=2, col=1)

    fig.update_layout(
        title=f"{symbol} Price Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False
    )

    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig

def main():
    st.set_page_config(page_title="Advanced Crypto Analysis App", layout="wide")
    st.title("Advanced Cryptocurrency Analysis App")

    # Sidebar for user input
    st.sidebar.header("Settings")
    crypto_symbols = st.sidebar.multiselect("Select Cryptocurrencies", ["BTC", "ETH", "XRP", "LTC", "ADA"], default=["BTC"])
    
    # Date range selector
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", datetime.now() - timedelta(days=365))
    end_date = col2.date_input("End Date", datetime.now())

    # Indicator selection
    st.sidebar.subheader("Indicators")
    selected_indicators = st.sidebar.multiselect("Select Indicators", 
                                                 ['SMA_20', 'EMA_20', 'RSI', 'VWAP', 'BB_upper', 'BB_lower', 'MACD', 'ATR', 'OBV'],
                                                 default=['SMA_20', 'EMA_20', 'RSI'])

    # Portfolio tracking
    st.sidebar.subheader("Portfolio Tracking")
    portfolio = {}
    for symbol in crypto_symbols:
        portfolio[symbol] = st.sidebar.number_input(f"{symbol} Holdings", min_value=0.0, value=0.0, step=0.1)

    # Fetch data and calculate indicators for all selected cryptocurrencies
    all_data = {}
    for symbol in crypto_symbols:
        data = fetch_crypto_data(symbol, start_date, end_date)
        data = calculate_indicators(data)
        data = detect_patterns_ml(data)
        data = simple_pattern_detection(data)
        all_data[symbol] = data

    # Create tabs
    tabs = st.tabs(["Price Analysis"] + crypto_symbols + ["Portfolio", "Backtesting", "Sentiment Analysis"])

    with tabs[0]:
        st.subheader("Multi-Currency Price Analysis")
        fig = go.Figure()
        for symbol, data in all_data.items():
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=f"{symbol} Close Price"))
        fig.update_layout(title="Comparative Price Analysis", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

    # Individual cryptocurrency tabs
    for i, symbol in enumerate(crypto_symbols, start=1):
        with tabs[i]:
            st.subheader(f"{symbol} Analysis")
            data = all_data[symbol]
            chart = create_candlestick_chart(data, symbol, selected_indicators, None, None)
            st.plotly_chart(chart, use_container_width=True)

            # Display additional analysis
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Technical Indicators")
                st.write(data[selected_indicators].tail())
            with col2:
                st.subheader("Pattern Recognition")
                st.write(f"Number of detected patterns: {data['Pattern_Change'].sum()}")

    # Portfolio tab
    with tabs[-3]:
        st.subheader("Portfolio Analysis")
        current_prices = {symbol: data['Close'].iloc[-1] for symbol, data in all_data.items()}
        portfolio_value = calculate_portfolio_value(portfolio, current_prices)
        st.write(f"Total Portfolio Value: ${portfolio_value:.2f}")

        # Portfolio composition pie chart
        fig = go.Figure(data=[go.Pie(labels=list(portfolio.keys()), 
                                     values=[portfolio[symbol] * current_prices[symbol] for symbol in portfolio])])
        fig.update_layout(title="Portfolio Composition")
        st.plotly_chart(fig, use_container_width=True)

    # Backtesting tab
    with tabs[-2]:
        st.subheader("Strategy Backtesting")
        selected_symbol = st.selectbox("Select Cryptocurrency for Backtesting", crypto_symbols)
        backtest_results = backtest_strategy(all_data[selected_symbol], simple_moving_average_crossover)

        st.write(f"Total Return: {backtest_results['Total_Return']:.2%}")
        st.write(f"Sharpe Ratio: {backtest_results['Sharpe_Ratio']:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=backtest_results['Cumulative_Returns'].index, 
                                 y=backtest_results['Cumulative_Returns'], 
                                 name="Strategy Returns"))
        fig.update_layout(title="Backtesting Results", xaxis_title="Date", yaxis_title="Cumulative Returns")
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment Analysis tab
    with tabs[-1]:
        st.subheader("Sentiment Analysis")
        for symbol in crypto_symbols:
            sentiment = fetch_news_sentiment(symbol)
            st.write(f"{symbol} Sentiment Score: {sentiment:.2f}")

if __name__ == "__main__":
    main()
