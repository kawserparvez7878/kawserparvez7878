# 🤖 AI Crypto Trading Bot

> **24/7 AI-powered cryptocurrency trading bot** with Binance API integration, LSTM neural network price prediction, advanced technical analysis, and real-time Telegram notifications.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Binance](https://img.shields.io/badge/Binance-API-yellow.svg)](https://binance.com)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://t.me/CodeReceiveBot)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Technical Analysis](#-technical-analysis)
- [ML Model](#-ml-model)
- [Telegram Notifications](#-telegram-notifications)
- [Risk Management](#-risk-management)
- [Author](#-author)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔄 **24/7 Monitoring** | Continuously monitors crypto markets around the clock |
| 📡 **Binance API** | Real-time market data, order placement, balance tracking |
| 🧠 **LSTM Neural Network** | Deep learning model for price direction prediction |
| 📈 **Technical Analysis** | MA, RSI, MACD, Bollinger Bands, Volume analysis |
| 🎯 **Signal Generation** | STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL |
| 📲 **Telegram Alerts** | Real-time trade signals sent to @CodeReceiveBot |
| 🛡️ **Risk Management** | Configurable stop-loss and take-profit levels |
| 📊 **Performance Reports** | Automated daily P&L and win-rate reports |
| 🔧 **Multi-pair Support** | Monitor BTC, ETH, BNB, SOL, ADA, XRP, DOGE, DOT |

---

## 📁 Project Structure

```
crypto-ai-trader/
├── trading_bot.py          # 🚀 Main trading engine (entry point)
├── ml_model.py             # 🧠 LSTM neural network for price prediction
├── technical_analysis.py   # 📈 TA indicators: MA, RSI, MACD, BB, Volume
├── telegram_handler.py     # 📲 Telegram notification system
├── config.py               # ⚙️  All configuration & API keys
├── requirements.txt        # 📦 Python dependencies
├── README.md               # 📖 This file
└── models/
    └── lstm_model.h5       # 💾 Saved trained model (auto-generated)
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/kawserparvez7878/kawserparvez7878.git
cd kawserparvez7878/crypto-ai-trader
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file
```bash
cp .env.example .env
# Then edit .env with your actual API keys
```

---

## ⚙️ Configuration

Create a `.env` file in the `crypto-ai-trader/` directory:

```env
# Binance API (get from https://www.binance.com/en/my/settings/api-management)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
USE_TESTNET=True   # Set False for live trading

# Telegram (get token from @BotFather)
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Parameters
TRADE_QUANTITY=0.001
CHECK_INTERVAL=3600
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=5.0
MAX_OPEN_TRADES=3

# ML Settings
ML_LOOKBACK=60
ML_EPOCHS=100
ML_BATCH_SIZE=32
```

---

## ▶️ Usage

### Start the bot
```bash
python trading_bot.py
```

### The bot will:
1. ✅ Connect to Binance API
2. ✅ Initialize the LSTM model (train if no saved model exists)
3. ✅ Send startup notification to Telegram
4. ✅ Begin 24/7 market monitoring loop
5. ✅ Generate and execute trade signals
6. ✅ Send real-time alerts to @CodeReceiveBot
7. ✅ Send daily performance reports

---

## 📈 Technical Analysis

The bot uses **5 indicators** to generate trading signals:

### Moving Averages (MA)
- **Short EMA** (9-period) vs **Long EMA** (21-period)
- Golden Cross → **BUY** | Death Cross → **SELL**
- Price vs SMA-50 for trend confirmation

### RSI (Relative Strength Index)
- Period: 14 | Overbought: 70 | Oversold: 30
- RSI < 30 → **BUY** | RSI > 70 → **SELL**
- Divergence detection for early signals

### MACD
- Fast: 12 | Slow: 26 | Signal: 9
- Bullish crossover → **BUY** | Bearish crossover → **SELL**
- Histogram momentum confirmation

### Bollinger Bands
- Period: 20 | Std Dev: 2.0
- Price at lower band → **BUY** | Price at upper band → **SELL**

### Volume Analysis
- Volume ratio vs 20-period average
- High volume + price up → **BUY** | High volume + price down → **SELL**

---

## 🧠 ML Model

### Architecture: LSTM Neural Network
```
Input (60 timesteps × 17 features)
    ↓
LSTM(128) → BatchNorm → Dropout(0.2)
    ↓
LSTM(64)  → BatchNorm → Dropout(0.2)
    ↓
LSTM(32)  → BatchNorm → Dropout(0.2)
    ↓
Dense(64, relu) → Dropout(0.1)
    ↓
Dense(32, relu)
    ↓
Dense(1, linear) → Predicted Price
```

### Features (17 total)
- OHLCV (5) + Price ratios (3) + Rolling MAs (3) + Std devs (2) + Momentum (2) + Volume (2)

### Training
- 80/20 train/validation split
- Early stopping with patience=15
- ReduceLROnPlateau for adaptive learning rate
- Model saved to `models/lstm_model.h5`
- Auto-retrains every 168 cycles (~1 week)

---

## 📲 Telegram Notifications

All alerts are sent to **@CodeReceiveBot**.

### Notification Types:
| Type | Trigger |
|---|---|
| 🚀 Startup | Bot starts |
| 🟢 BUY Signal | Buy order placed |
| 🔴 SELL Signal | Sell order placed |
| 🛑 Stop-Loss | Stop-loss triggered |
| ✅ Take-Profit | Take-profit triggered |
| 📊 Daily Report | Every 24 analysis cycles |
| 🚨 Error Alert | Any critical error |

---

## 🛡️ Risk Management

| Parameter | Default | Description |
|---|---|---|
| Stop Loss | 2% | Auto-sell if price drops 2% from entry |
| Take Profit | 5% | Auto-sell if price rises 5% from entry |
| Max Open Trades | 3 | Maximum simultaneous positions |
| Min ML Confidence | 60% | Minimum ML confidence to act on signal |
| Min Balance | $10 | Minimum USDT balance to place a trade |

> ⚠️ **DISCLAIMER**: This bot is for educational purposes. Cryptocurrency trading involves significant risk. Always test on Binance Testnet before using real funds. Past performance does not guarantee future results.

---

## 👤 Author

**kawser parvez**
- 📧 Email: kawserparvez7878@gmail.com
- 🤖 Telegram: @CodeReceiveBot
- 🐙 GitHub: [kawserparvez7878](https://github.com/kawserparvez7878)

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ using Python, TensorFlow, Binance API & Telegram Bot API*
