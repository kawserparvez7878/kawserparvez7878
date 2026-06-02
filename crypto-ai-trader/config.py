#!/usr/bin/env python3
"""
Configuration File - API Keys, Trading Parameters & Settings
Author: kawser parvez | kawserparvez7878@gmail.com
Description: Central configuration for the AI Crypto Trading Bot.
             Replace placeholder values with your actual API keys.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────────────────────
# BINANCE API CONFIGURATION
# ─────────────────────────────────────────────────────────────
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'YOUR_BINANCE_API_KEY_HERE')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', 'YOUR_BINANCE_SECRET_KEY_HERE')

# Use Testnet for paper trading (recommended for testing)
USE_TESTNET = os.getenv('USE_TESTNET', 'True').lower() == 'true'
BINANCE_TESTNET_URL = 'https://testnet.binance.vision/api'

# ─────────────────────────────────────────────────────────────
# TELEGRAM CONFIGURATION
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8723812816:AAFov0BuNE3bJdmhSjN2wPH4FYa-MaYFLyI')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'YOUR_TELEGRAM_CHAT_ID')
TELEGRAM_BOT_USERNAME = '@CodeReceiveBot'

# ─────────────────────────────────────────────────────────────
# TRADING PAIRS TO MONITOR
# ─────────────────────────────────────────────────────────────
TRADING_PAIRS = [
    'BTCUSDT',   # Bitcoin
    'ETHUSDT',   # Ethereum
    'BNBUSDT',   # Binance Coin
    'SOLUSDT',   # Solana
    'ADAUSDT',   # Cardano
    'XRPUSDT',   # Ripple
    'DOGEUSDT',  # Dogecoin
    'DOTUSDT',   # Polkadot
]

# ─────────────────────────────────────────────────────────────
# TRADING PARAMETERS
# ─────────────────────────────────────────────────────────────
# Trade quantity in base asset (e.g., 0.001 BTC)
TRADE_QUANTITY = float(os.getenv('TRADE_QUANTITY', '0.001'))

# How often to check the market (in seconds)
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '3600'))  # 1 hour default

# Risk management
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', '2.0'))    # 2% stop loss
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', '5.0'))  # 5% take profit

# Maximum number of simultaneous open trades
MAX_OPEN_TRADES = int(os.getenv('MAX_OPEN_TRADES', '3'))

# Minimum confidence threshold for ML predictions (0.0 - 1.0)
MIN_ML_CONFIDENCE = float(os.getenv('MIN_ML_CONFIDENCE', '0.6'))

# ─────────────────────────────────────────────────────────────
# TECHNICAL ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────
# Moving Averages
MA_SHORT = int(os.getenv('MA_SHORT', '9'))    # Short EMA period
MA_LONG = int(os.getenv('MA_LONG', '21'))     # Long EMA period

# RSI
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70.0'))
RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30.0'))

# MACD
MACD_FAST = int(os.getenv('MACD_FAST', '12'))
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))

# Bollinger Bands
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
BB_STD_DEV = float(os.getenv('BB_STD_DEV', '2.0'))

# ─────────────────────────────────────────────────────────────
# ML MODEL SETTINGS
# ─────────────────────────────────────────────────────────────
# Number of historical candles to look back for LSTM input
ML_LOOKBACK = int(os.getenv('ML_LOOKBACK', '60'))

# Training epochs
ML_EPOCHS = int(os.getenv('ML_EPOCHS', '100'))

# Batch size for training
ML_BATCH_SIZE = int(os.getenv('ML_BATCH_SIZE', '32'))

# Path to save/load the trained model
MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models/lstm_model.h5')

# Retrain model every N cycles
RETRAIN_INTERVAL = int(os.getenv('RETRAIN_INTERVAL', '168'))  # ~1 week of hourly cycles

# ─────────────────────────────────────────────────────────────
# DATA SETTINGS
# ─────────────────────────────────────────────────────────────
# Candlestick interval for analysis
CANDLE_INTERVAL = os.getenv('CANDLE_INTERVAL', '1h')

# Number of historical candles to fetch
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '500'))

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

# ─────────────────────────────────────────────────────────────
# GITHUB CONFIGURATION (optional - for auto-pushing reports)
# ─────────────────────────────────────────────────────────────
GITHUB_REPO = os.getenv('GITHUB_REPO', 'kawserparvez7878/kawserparvez7878')
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN_HERE')
