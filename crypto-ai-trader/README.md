# 🤖 AI-Powered Cryptocurrency Trading Bot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![Binance](https://img.shields.io/badge/Binance-API-yellow.svg)](https://binance.com)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://telegram.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A fully automated, AI-powered cryptocurrency trading bot that monitors markets 24/7, uses LSTM neural networks for price prediction, performs technical analysis, and sends real-time trading signals via Telegram.

---

## 🚀 Features

- **24/7 Market Monitoring** — Continuously scans multiple crypto pairs
- **Binance API Integration** — Real-time market data and order execution
- **LSTM Neural Network** — Deep learning model for price prediction
- **Technical Analysis** — MA, RSI, MACD, Bollinger Bands, Stochastic, ATR
- **Smart Signal Generation** — Combines ML + TA for BUY/SELL signals
- **Telegram Notifications** — Real-time alerts to @CodeReceiveBot
- **Risk Management** — Stop-loss, take-profit, max drawdown protection
- **Performance Reports** — Hourly P&L and trade statistics
- **Paper Trading Mode** — Test safely before going live
- **Multi-Pair Support** — Monitor 10+ cryptocurrency pairs simultaneously

---

## 📁 Project Structure

```
crypto-ai-trader/
├── trading_bot.py          # Main trading engine & orchestrator
├── ml_model.py             # LSTM neural network for price prediction
├── technical_analysis.py   # TA indicators (MA, RSI, MACD, BB, etc.)
├── telegram_handler.py     # Telegram notification system
├── config.py               # Configuration & API keys
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/                 # Saved ML models (auto-created)
│   └── lstm_model.h5
└── trading_bot.log         # Log file (auto-created)
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
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Create Models Directory
```bash
mkdir models
```

---

## 🔑 Configuration

### Option A: Environment Variables (Recommended)
Create a `.env` file in the project root:
```env
# Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# Telegram
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=@YourTelegramUsername

# Trading Settings
TRADE_QUANTITY=0.1
STOP_LOSS_PERCENT=2.0
TAKE_PROFIT_PERCENT=4.0
CHECK_INTERVAL=300

# Safety
USE_TESTNET=True
LIVE_TRADING=False
```

### Option B: Edit config.py directly
Open `config.py` and replace the placeholder values with your actual API keys.

### Getting API Keys

**Binance API:**
1. Go to [Binance](https://www.binance.com) → Account → API Management
2. Create a new API key
3. Enable "Read Info" and "Enable Spot & Margin Trading"
4. Copy API Key and Secret Key

**Telegram Bot:**
1. Open Telegram → Search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the bot token
4. Get your chat ID by messaging `@userinfobot`

---

## 🎯 Usage

### Start the Bot
```bash
python trading_bot.py
```

### Test Configuration
```bash
python config.py
```

### Run Technical Analysis Only
```python
from technical_analysis import TechnicalAnalysis
import pandas as pd

ta = TechnicalAnalysis()
# Load your data
df = pd.read_csv('your_data.csv')
signals = ta.generate_signals(df)
print(signals)
```

### Train ML Model Manually
```python
from ml_model import MLModel
import pandas as pd

model = MLModel()
df = pd.read_csv('historical_data.csv')
metrics = model.train(df)
print(f"Model accuracy: {metrics['direction_accuracy']:.1f}%")
```

---

## 📊 Trading Strategy

The bot uses a **multi-signal consensus approach**:

| Signal Source | Weight | Description |
|--------------|--------|-------------|
| Moving Average | 1x | Golden/Death cross crossover |
| RSI | 1x | Overbought (>70) / Oversold (<30) |
| MACD | 1x | Bullish/Bearish crossover |
| Bollinger Bands | 1x | Price at band extremes |
| LSTM ML Model | 2x | Predicted price direction |

**Signal Thresholds:**
- `STRONG_BUY` → 3+ buy signals
- `BUY` → More buy than sell signals
- `HOLD` → Equal signals
- `SELL` → More sell than buy signals
- `STRONG_SELL` → 3+ sell signals

---

## 🛡️ Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Stop Loss | 2% | Auto-sell if price drops 2% |
| Take Profit | 4% | Auto-sell if price rises 4% |
| Trade Size | 10% | Use 10% of balance per trade |
| Max Positions | 3 | Max simultaneous open trades |
| Max Daily Loss | 5% | Stop bot if daily loss exceeds 5% |

---

## 📱 Telegram Notifications

The bot sends the following alerts to **@CodeReceiveBot**:

- 🟢 **BUY Signal** — When a buy opportunity is detected
- 🔴 **SELL Signal** — When a sell opportunity is detected
- 💸 **Trade Executed** — When an order is placed
- 🛑 **Stop-Loss** — When stop-loss is triggered
- ✅ **Take-Profit** — When take-profit is triggered
- 📊 **Performance Report** — Hourly P&L summary
- ❌ **Error Alert** — If any error occurs

---

## 🔧 Supported Trading Pairs

| Pair | Asset |
|------|-------|
| BTCUSDT | Bitcoin |
| ETHUSDT | Ethereum |
| BNBUSDT | Binance Coin |
| SOLUSDT | Solana |
| ADAUSDT | Cardano |
| XRPUSDT | Ripple |
| DOGEUSDT | Dogecoin |
| MATICUSDT | Polygon |
| DOTUSDT | Polkadot |
| AVAXUSDT | Avalanche |

---

## 🧠 ML Model Architecture

```
Input Layer: (60 timesteps × 4 features)
    ↓
LSTM Layer 1: 128 units + BatchNorm + Dropout(0.2)
    ↓
LSTM Layer 2: 64 units + BatchNorm + Dropout(0.2)
    ↓
LSTM Layer 3: 32 units + BatchNorm + Dropout(0.2)
    ↓
Dense Layer: 64 units (ReLU) + Dropout(0.1)
    ↓
Dense Layer: 32 units (ReLU)
    ↓
Output Layer: 1 unit (Linear) → Predicted Price
```

**Features used:** Close Price, Volume, High, Low

---

## ⚠️ Disclaimer

> **IMPORTANT:** This bot is for educational purposes only. Cryptocurrency trading involves significant financial risk. Never trade with money you cannot afford to lose. Always test with paper trading (testnet) before using real funds. The authors are not responsible for any financial losses.

---

## 👤 Author

**kawser parvez**
- Email: kawserparvez7878@gmail.com
- Telegram: @CodeReceiveBot

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

*Built with ❤️ using Python, TensorFlow, Binance API & Telegram*
