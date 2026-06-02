#!/usr/bin/env python3
"""
Configuration File - AI Crypto Trading Bot
Author: kawser parvez
Email: kawserparvez7878@gmail.com
WARNING: Replace placeholder API keys with your actual keys before running!
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# BINANCE API CONFIGURATION
# ============================================================
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY_HERE')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'YOUR_BINANCE_SECRET_KEY_HERE')

# Use testnet for paper trading (recommended for testing)
USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
BINANCE_TESTNET_URL = 'https://testnet.binance.vision/api'

# ============================================================
# TELEGRAM CONFIGURATION
# ============================================================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8723812816:AAFov0BuNE3bJdmhSjN2wPH4FYa-MaYFLyI')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '@CodeReceiveBot')
TELEGRAM_BOT_USERNAME = '@CodeReceiveBot'

# ============================================================
# TRADING PAIRS TO MONITOR
# ============================================================
TRADING_PAIRS = [
    'BTCUSDT',   # Bitcoin
    'ETHUSDT',   # Ethereum
    'BNBUSDT',   # Binance Coin
    'SOLUSDT',   # Solana
    'ADAUSDT',   # Cardano
    'XRPUSDT',   # Ripple
    'DOGEUSDT',  # Dogecoin
    'MATICUSDT', # Polygon
    'DOTUSDT',   # Polkadot
    'AVAXUSDT',  # Avalanche
]

# ============================================================
# TRADING PARAMETERS
# ============================================================
# Percentage of available balance to use per trade (0.1 = 10%)
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', '0.1'))

# Stop-loss percentage (2% below entry price)
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))

# Take-profit percentage (4% above entry price)
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', '4.0'))

# Interval between market checks (in seconds)
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '300'))  # 5 minutes

# Maximum number of simultaneous open positions
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))

# Minimum trade value in USDT
MIN_TRADE_VALUE = float(os.getenv('MIN_TRADE_VALUE', '10.0'))

# ============================================================
# TECHNICAL ANALYSIS PARAMETERS
# ============================================================
# Moving Average periods
MA_SHORT_PERIOD = int(os.getenv('MA_SHORT_PERIOD', '10'))
MA_LONG_PERIOD = int(os.getenv('MA_LONG_PERIOD', '30'))

# RSI settings
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30.0'))

# MACD settings
MACD_FAST = int(os.getenv('MACD_FAST', '12'))
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))

# Bollinger Bands settings
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD_DEV = float(os.getenv('BB_STD_DEV', '2.0'))

# ============================================================
# MACHINE LEARNING MODEL SETTINGS
# ============================================================
# Number of historical candles to use as input features
ML_LOOKBACK = int(os.getenv('ML_LOOKBACK', '60'))

# Training epochs
ML_EPOCHS = int(os.getenv('ML_EPOCHS', '100'))

# Batch size for training
ML_BATCH_SIZE = int(os.getenv('ML_BATCH_SIZE', '32'))

# Train/test split ratio
ML_TRAIN_SPLIT = float(os.getenv('ML_TRAIN_SPLIT', '0.8'))

# Minimum price change % to generate BUY/SELL signal
ML_SIGNAL_THRESHOLD = float(os.getenv('ML_SIGNAL_THRESHOLD', '1.5'))

# Path to save/load trained model
MODEL_PATH = os.getenv('MODEL_PATH', 'models/lstm_model.h5')

# Retrain model every N hours
MODEL_RETRAIN_HOURS = int(os.getenv('MODEL_RETRAIN_HOURS', '24'))

# ============================================================
# DATA SETTINGS
# ============================================================
# Candlestick interval for analysis
CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')  # 1m, 5m, 15m, 1h, 4h, 1d

# Number of historical candles to fetch
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '500'))

# ============================================================
# LOGGING SETTINGS
# ============================================================
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

# ============================================================
# RISK MANAGEMENT
# ============================================================
# Maximum daily loss percentage before stopping bot
MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '5.0'))

# Maximum drawdown percentage
MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '10.0'))

# Enable/disable live trading (False = paper trading mode)
LIVE_TRADING = os.getenv('LIVE_TRADING', 'False').lower() == 'true'

if __name__ == '__main__':
    print("=" * 50)
    print("AI Crypto Trading Bot - Configuration")
    print("=" * 50)
    print(f"Trading Pairs: {len(TRADING_PAIRS)} pairs")
    print(f"Trade Quantity: {TRADE_QUANTITY * 100}% per trade")
    print(f"Stop Loss: {STOP_LOSS_PERCENT}%")
    print(f"Take Profit: {TAKE_PROFIT_PERCENT}%")
    print(f"Check Interval: {CHECK_INTERVAL}s")
    print(f"ML Lookback: {ML_LOOKBACK} candles")
    print(f"Live Trading: {LIVE_TRADING}")
    print(f"Testnet Mode: {USE_TESTNET}")
    print("=" * 50)
