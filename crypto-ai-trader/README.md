# 🤖 AI Crypto Trading Bot

> **Fully automated, AI-powered cryptocurrency trading bot** with LSTM neural network predictions, multi-indicator technical analysis, Binance API integration, and real-time Telegram notifications.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📈 **Live Market Data** | Real-time OHLCV data via Binance API |
| 🧠 **LSTM Neural Network** | Deep learning price direction prediction |
| 📊 **Technical Analysis** | MA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic |
| 🤖 **Signal Generation** | Weighted multi-indicator BUY/SELL/HOLD signals |
| 📱 **Telegram Alerts** | Real-time notifications to @CodeReceiveBot |
| 🛡️ **Risk Management** | Stop Loss, Take Profit, daily loss limits |
| 🔄 **24/7 Monitoring** | Continuous market scanning across multiple pairs |
| 📉 **Multi-Pair Support** | BTC, ETH, BNB, SOL, ADA, XRP, DOGE, DOT |
| 🔁 **Auto Retraining** | ML model retrains periodically on fresh data |
| 📋 **Performance Reports** | Daily P&L, win rate, trade history via Telegram |

---

## 📁 Project Structure

```
crypto-ai-trader/
├── trading_bot.py          # 🏗️  Main trading engine & orchestration
├── ml_model.py             # 🧠  LSTM neural network for price prediction
├── technical_analysis.py   # 📊  TA indicators & signal generation
├── telegram_handler.py     # 📱  Telegram notification system
├── config.py               # ⚙️   Configuration & API keys
├── requirements.txt        # 📦  Python dependencies
├── README.md               # 📖  This file
├── models/                 # 💾  Saved ML models (auto-created)
└── logs/                   # 📝  Log files (auto-created)
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/kawserparvez7878/kawserparvez7878.git
cd kawserparvez7878/crypto-ai-trader
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the `crypto-ai-trader/` directory:
```env
# Binance API (get from https://www.binance.com/en/my/settings/api-management)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
USE_TESTNET=True   # Set False for live trading

# Telegram (get chat ID by messaging @userinfobot)
TELEGRAM_TOKEN=8723812816:AAFov0BuNE3bJdmhSjN2wPH4FYa-MaYFLyI
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Parameters
TRADE_QUANTITY=0.02
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=5.0
MAX_OPEN_TRADES=3
CHECK_INTERVAL=3600
```

---

## ▶️ Usage

### Start the Bot
```bash
python trading_bot.py
```

### Test Telegram Connection
```python
from telegram_handler import TelegramHandler
th = TelegramHandler()
th.test_connection()
th.send_alert("Bot test message!")
```

### Train ML Model Manually
```python
from ml_model import CryptoPricePredictor
from trading_bot import CryptoTradingBot

bot = CryptoTradingBot()
df = bot.get_historical_data('BTCUSDT', limit=1000)

predictor = CryptoPricePredictor()
result = predictor.train(df, 'BTCUSDT')
print(f"Accuracy: {result['accuracy']:.2%}")
```

### Run Technical Analysis
```python
from technical_analysis import TechnicalAnalysis

ta = TechnicalAnalysis()
signal, details = ta.generate_signal(df)
print(f"Signal: {signal}")
print(f"RSI: {details['rsi']['rsi']}")
print(f"MACD: {details['macd']['histogram']}")
```

---

## 🧠 ML Model Architecture

```
Input (60 timesteps × 15 features)
    ↓
LSTM(128) → BatchNorm → Dropout(0.3)
    ↓
LSTM(64)  → BatchNorm → Dropout(0.2)
    ↓
LSTM(32)  → BatchNorm → Dropout(0.2)
    ↓
Dense(32, ReLU) → Dropout(0.1)
    ↓
Dense(16, ReLU)
    ↓
Dense(1, Sigmoid)  →  P(price goes UP)
```

**Features used:** Open, High, Low, Close, Volume, Returns, Log Returns, Volatility, Price Range, Body Size, MA7/21/50 ratios, RSI, MACD, Volume Ratio

---

## 📊 Technical Indicators

| Indicator | Parameters | Signal Logic |
|---|---|---|
| **EMA Crossover** | 9 / 21 / 50 | Golden/Death cross + trend filter |
| **RSI** | Period 14 | Oversold (<30) = BUY, Overbought (>70) = SELL |
| **MACD** | 12/26/9 | Bullish/Bearish crossover + histogram |
| **Bollinger Bands** | 20 period, 2σ | Price at lower/upper band |
| **Stochastic** | %K=14, %D=3 | Crossover in oversold/overbought zones |
| **Volume** | 20-period MA | High volume confirmation |
| **ATR** | Period 14 | Volatility context |

**Signal Weights:** MA=2, MACD=2, RSI=1.5, BB=1, Stochastic=1, Volume=1

---

## 📱 Telegram Notifications

The bot sends the following messages to **@CodeReceiveBot**:

- 🟢 **BUY Signal** — Symbol, price, quantity, SL, TP, confidence, TA & ML details
- 🔴 **SELL Signal** — Same as above for sell signals
- 💰 **Trade Closed** — Entry/exit price, P&L in USDT and %
- 📊 **Daily Report** — Balance, win rate, total P&L, open trades
- ⚠️ **Alerts** — Stop loss hits, daily loss limit, errors
- 🌍 **Market Summary** — All pairs with price, 24h change, signal

---

## 🛡️ Risk Management

- **Stop Loss:** 2% below entry price (configurable)
- **Take Profit:** 5% above entry price (configurable)
- **Max Open Trades:** 3 simultaneous positions
- **Daily Loss Limit:** Bot pauses if daily loss exceeds threshold
- **Signal Cooldown:** 60-minute cooldown per pair after a signal
- **Trade Size:** 2% of available balance per trade

---

## ⚠️ Disclaimer

> **This bot is for educational purposes only.**
> Cryptocurrency trading involves significant financial risk.
> Always start with the **testnet** (`USE_TESTNET=True`) before live trading.
> Never invest more than you can afford to lose.
> Past performance does not guarantee future results.

---

## 👤 Author

**kawser parvez**
- 📧 Email: kawserparvez7878@gmail.com
- 🐙 GitHub: [@kawserparvez7878](https://github.com/kawserparvez7878)
- 📱 Telegram: @CodeReceiveBot

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

*Built with ❤️ using Python, TensorFlow, Binance API & Telegram Bot API*
