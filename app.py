import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import json
import os
import requests
from textblob import TextBlob
import ccxt
from concurrent.futures import ThreadPoolExecutor

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for ML model"""
        features = pd.DataFrame()
        
        # Price-based features
        features['price_change'] = df['Close'].pct_change()
        features['price_volatility'] = df['Close'].rolling(window=20).std()
        
        # Volume features
        features['volume_change'] = df['Volume'].pct_change()
        features['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Technical indicators
        features['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        features['macd'] = ta.trend.MACD(df['Close']).macd()
        
        # Remove NaN values
        features = features.dropna()
        
        return features
        
    def prepare_labels(self, df):
        """Prepare labels for ML model (1 for price increase, 0 for decrease)"""
        return (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]
        
    def train(self, df):
        """Train the ML model"""
        features = self.prepare_features(df)
        labels = self.prepare_labels(df)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(scaled_features[:-1], labels)
        
    def predict(self, df):
        """Make predictions"""
        features = self.prepare_features(df)
        scaled_features = self.scaler.transform(features)
        return self.model.predict_proba(scaled_features)

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        
    def simple_moving_average_strategy(self, df, short_window=20, long_window=50):
        """Simple moving average crossover strategy"""
        df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        df['Signal'] = 0
        df.loc[df['SMA_short'] > df['SMA_long'], 'Signal'] = 1
        df.loc[df['SMA_short'] < df['SMA_long'], 'Signal'] = -1
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        
        # Calculate cumulative returns
        df['Cumulative_Market_Returns'] = (1 + df['Returns']).cumprod()
        df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
        
        return df
        
    def calculate_metrics(self, df):
        """Calculate backtest metrics"""
        strategy_returns = df['Strategy_Returns'].dropna()
        
        metrics = {
            'Total Return': f"{(df['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 100:.2f}%",
            'Annual Return': f"{(df['Cumulative_Strategy_Returns'].iloc[-1] - 1) * 365/len(df) * 100:.2f}%",
            'Sharpe Ratio': f"{np.sqrt(252) * strategy_returns.mean() / strategy_returns.std():.2f}",
            'Max Drawdown': f"{(df['Cumulative_Strategy_Returns'].cummax() - df['Cumulative_Strategy_Returns']).max() * 100:.2f}%"
        }
        
        return metrics

class SentimentAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('NEWS_API_KEY', '')
        
    def fetch_news(self, crypto):
        """Fetch news articles about cryptocurrency"""
        url = f"https://newsapi.org/v2/everything?q={crypto}&apiKey={self.api_key}"
        try:
            response = requests.get(url)
            return response.json()['articles']
        except:
            return []
            
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        return TextBlob(text).sentiment.polarity
        
    def get_overall_sentiment(self, crypto):
        """Get overall sentiment from news articles"""
        articles = self.fetch_news(crypto)
        if not articles:
            return 0
            
        sentiments = [self.analyze_sentiment(article['title'] + ' ' + article['description'])
                     for article in articles]
        return np.mean(sentiments)

class PortfolioTracker:
    def __init__(self):
        self.exchange = ccxt.binance()
        
    def load_portfolio(self):
        """Load portfolio from storage"""
        if os.path.exists('portfolio.json'):
            with open('portfolio.json', 'r') as f:
                return json.load(f)
        return {}
        
    def save_portfolio(self, portfolio):
        """Save portfolio to storage"""
        with open('portfolio.json', 'w') as f:
            json.dump(portfolio, f)
            
    def calculate_portfolio_value(self, portfolio):
        """Calculate current portfolio value"""
        total_value = 0
        for crypto, amount in portfolio.items():
            try:
                ticker = self.exchange.fetch_ticker(f"{crypto}/USDT")
                price = ticker['last']
                value = float(amount) * price
                total_value += value
            except:
                continue
        return total_value
        
    def get_portfolio_stats(self, portfolio):
        """Get portfolio statistics"""
        stats = {}
        total_value = self.calculate_portfolio_value(portfolio)
        
        for crypto, amount in portfolio.items():
            try:
                ticker = self.exchange.fetch_ticker(f"{crypto}/USDT")
                price = ticker['last']
                value = float(amount) * price
                stats[crypto] = {
                    'amount': amount,
                    'value': value,
                    'percentage': (value / total_value) * 100 if total_value > 0 else 0
                }
            except:
                continue
                
        return stats

def main():
    st.set_page_config(layout="wide")
    st.title("Advanced Cryptocurrency Trading Dashboard")

    # Initialize components
    ml_predictor = MLPredictor()
    backtester = Backtester()
    sentiment_analyzer = SentimentAnalyzer()
    portfolio_tracker = PortfolioTracker()

    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market Analysis", 
        "ML Predictions", 
        "Backtesting", 
        "Sentiment Analysis",
        "Portfolio Tracking"
    ])

    # Market Analysis Tab
    with tab1:
        crypto_symbols = ['BTC', 'ETH', 'DOGE', 'ADA', 'DOT', 'SOL', 'MATIC', 'LINK']
        selected_crypto = st.selectbox("Select Cryptocurrency", crypto_symbols)
        
        timeframe = st.selectbox(
            "Select Timeframe",
            ["1 Day", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "YTD"]
        )

        # Fetch data
        end_date = datetime.now()
        start_date = end_date - timedelta(days={
            "1 Day": 1, "1 Week": 7, "1 Month": 30, "3 Months": 90,
            "6 Months": 180, "1 Year": 365, "YTD": (end_date - datetime(end_date.year, 1, 1)).days
        }[timeframe])

        df = yf.download(f"{selected_crypto}-USD", start=start_date, end=end_date)

        # Display main chart
        fig = go.Figure(data=[go.Candlestick(x=df.index,
                                            open=df['Open'],
                                            high=df['High'],
                                            low=df['Low'],
                                            close=df['Close'])])
        st.plotly_chart(fig, use_container_width=True)

    # ML Predictions Tab
    with tab2:
        st.subheader("Machine Learning Predictions")
        
        if st.button("Train Model"):
            with st.spinner("Training ML model..."):
                ml_predictor.train(df)
            
            predictions = ml_predictor.predict(df)
            
            st.write("Probability of price increase:", f"{predictions[-1][1]:.2%}")
            st.write("Probability of price decrease:", f"{predictions[-1][0]:.2%}")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': ml_predictor.prepare_features(df).columns,
                'Importance': ml_predictor.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.write("Feature Importance:")
            st.bar_chart(feature_importance.set_index('Feature'))

    # Backtesting Tab
    with tab3:
        st.subheader("Strategy Backtesting")
        
        col1, col2 = st.columns(2)
        with col1:
            short_window = st.slider("Short MA Window", 5, 50, 20)
        with col2:
            long_window = st.slider("Long MA Window", 20, 200, 50)
            
        backtest_results = backtester.simple_moving_average_strategy(
            df.copy(), 
            short_window=short_window, 
            long_window=long_window
        )
        
        metrics = backtester.calculate_metrics(backtest_results)
        
        # Display metrics
        for metric, value in metrics.items():
            st.metric(metric, value)
            
        # Plot strategy returns
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['Cumulative_Market_Returns'],
            name="Market Returns"
        ))
        fig.add_trace(go.Scatter(
            x=backtest_results.index,
            y=backtest_results['Cumulative_Strategy_Returns'],
            name="Strategy Returns"
        ))
        st.plotly_chart(fig, use_container_width=True)

    # Sentiment Analysis Tab
    with tab4:
        st.subheader("Market Sentiment Analysis")
        
        sentiment = sentiment_analyzer.get_overall_sentiment(selected_crypto)
        
        # Display sentiment gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-1, 1]},
                'steps': [
                    {'range': [-1, -0.3], 'color': "red"},
                    {'range': [-0.3, 0.3], 'color': "gray"},
                    {'range': [0.3, 1], 'color': "green"}
                ]
            }
        ))
        st.plotly_chart(fig)

    # Portfolio Tracking Tab
    with tab5:
        st.subheader("Portfolio Tracker")
        
        portfolio = portfolio_tracker.load_portfolio()
        
        # Add new position
        col1, col2 = st.columns(2)
        with col1:
            new_crypto = st.selectbox("Cryptocurrency", crypto_symbols)
            new_amount = st.number_input("Amount", min_value=0.0)
            
        if st.button("Add Position"):
            portfolio[new_crypto] = new_amount
            portfolio_tracker.save_portfolio(portfolio)
            
        # Display portfolio
        if portfolio:
            stats = portfolio_tracker.get_portfolio_stats(portfolio)
            
            # Portfolio pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(stats.keys()),
                values=[stat['value'] for stat in stats.values()]
            )])
            st.plotly_chart(fig)
            
            # Portfolio table
            st.write("Portfolio Breakdown:")
            portfolio_df = pd.DataFrame.from_dict(stats, orient='index')
            st.dataframe(portfolio_df)

if __name__ == "__main__":
    main()
