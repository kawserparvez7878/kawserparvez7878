#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Main Trading Engine
Author: kawser parvez | kawserparvez7878@gmail.com
Description: 24/7 automated cryptocurrency trading bot with ML predictions,
             technical analysis, and real-time Telegram notifications.
"""

import time
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException

import config
from technical_analysis import TechnicalAnalysis
from ml_model import CryptoPricePredictor
from telegram_handler import TelegramHandler

os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('TradingBot')


class CryptoTradingBot:
    """Main AI-powered cryptocurrency trading bot."""

    def __init__(self):
        logger.info("Initializing AI Crypto Trading Bot...")
        self.client = Client(
            api_key=config.BINANCE_API_KEY,
            api_secret=config.BINANCE_SECRET_KEY,
            testnet=config.USE_TESTNET
        )
        self.ta = TechnicalAnalysis()
        self.predictor = CryptoPricePredictor()
        self.telegram = TelegramHandler()
        self.open_trades: Dict[str, dict] = {}
        self.trade_history: List[dict] = []
        self.daily_pnl: float = 0.0
        self.start_balance: float = 0.0
        self.signal_cooldowns: Dict[str, datetime] = {}
        self.cycle_count: int = 0
        self.last_report_time: datetime = datetime.now()
        logger.info("Bot initialized successfully!")

    def get_historical_data(self, symbol: str, interval: str = None, limit: int = None) -> pd.DataFrame:
        """Fetch OHLCV candlestick data from Binance."""
        interval = interval or config.CANDLE_INTERVAL
        limit = limit or config.HISTORY_LIMIT
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API error fetching {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0

    def get_account_balance(self, asset: str = 'USDT') -> float:
        """Get available balance for a given asset."""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free']) if balance else 0.0
        except BinanceAPIException as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def generate_signal(self, symbol: str) -> Tuple[str, float, dict]:
        """Generate trading signal combining TA and ML predictions."""
        df = self.get_historical_data(symbol)
        if df.empty or len(df) < 100:
            return 'HOLD', 0.0, {}

        ta_signal, ta_details = self.ta.generate_signal(df)
        ml_signal, ml_confidence, ml_details = self.predictor.predict(df, symbol)

        signal_scores = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        signal_scores[ta_signal] += 1.0
        signal_scores[ml_signal] += ml_confidence

        final_signal = max(signal_scores, key=signal_scores.get)
        total = sum(signal_scores.values())
        confidence = signal_scores[final_signal] / total if total > 0 else 0.0

        details = {
            'ta_signal': ta_signal, 'ml_signal': ml_signal,
            'ml_confidence': ml_confidence, 'ta_details': ta_details,
            'ml_details': ml_details, 'current_price': df['close'].iloc[-1]
        }
        logger.info(f"{symbol} | Signal: {final_signal} | Confidence: {confidence:.2%} | TA: {ta_signal} | ML: {ml_signal}")
        return final_signal, confidence, details

    def is_signal_on_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is in signal cooldown period."""
        if symbol in self.signal_cooldowns:
            elapsed = (datetime.now() - self.signal_cooldowns[symbol]).total_seconds() / 60
            return elapsed < config.SIGNAL_COOLDOWN_MINUTES
        return False

    def calculate_trade_quantity(self, symbol: str, price: float) -> float:
        """Calculate trade quantity based on risk management rules."""
        balance = self.get_account_balance('USDT')
        trade_value = balance * config.TRADE_QUANTITY
        trade_value = max(10.0, min(trade_value, 1000.0))
        return round(trade_value / price, 6)

    def execute_buy(self, symbol: str, price: float, confidence: float, details: dict) -> bool:
        """Execute a BUY market order."""
        if len(self.open_trades) >= config.MAX_OPEN_TRADES:
            logger.warning(f"Max open trades reached. Skipping BUY for {symbol}.")
            return False
        quantity = self.calculate_trade_quantity(symbol, price)
        if quantity <= 0:
            return False
        try:
            if config.USE_TESTNET:
                order = {'orderId': f'TEST_{int(time.time())}', 'status': 'FILLED'}
            else:
                order = self.client.order_market_buy(symbol=symbol, quantity=quantity)

            stop_loss = price * (1 - config.STOP_LOSS_PERCENT / 100)
            take_profit = price * (1 + config.TAKE_PROFIT_PERCENT / 100)
            self.open_trades[symbol] = {
                'symbol': symbol, 'side': 'BUY', 'entry_price': price,
                'quantity': quantity, 'stop_loss': stop_loss,
                'take_profit': take_profit, 'order_id': order.get('orderId'),
                'timestamp': datetime.now(), 'confidence': confidence
            }
            self.signal_cooldowns[symbol] = datetime.now()
            logger.info(f"BUY: {symbol} @ {price:.4f} | Qty: {quantity} | SL: {stop_loss:.4f} | TP: {take_profit:.4f}")
            self.telegram.send_trade_signal(
                signal='BUY', symbol=symbol, price=price,
                quantity=quantity, stop_loss=stop_loss,
                take_profit=take_profit, confidence=confidence, details=details
            )
            return True
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"BUY order failed for {symbol}: {e}")
            return False

    def execute_sell(self, symbol: str, price: float, reason: str = 'SIGNAL') -> bool:
        """Execute a SELL market order."""
        if symbol not in self.open_trades:
            return False
        trade = self.open_trades[symbol]
        try:
            if config.USE_TESTNET:
                order = {'orderId': f'TEST_{int(time.time())}', 'status': 'FILLED'}
            else:
                order = self.client.order_market_sell(symbol=symbol, quantity=trade['quantity'])

            pnl = (price - trade['entry_price']) * trade['quantity']
            pnl_pct = (price - trade['entry_price']) / trade['entry_price'] * 100
            self.daily_pnl += pnl
            self.trade_history.append({**trade, 'exit_price': price,
                'exit_time': datetime.now(), 'pnl': pnl, 'pnl_pct': pnl_pct, 'exit_reason': reason})
            del self.open_trades[symbol]
            self.signal_cooldowns[symbol] = datetime.now()
            logger.info(f"SELL: {symbol} @ {price:.4f} | PnL: {pnl:.4f} USDT ({pnl_pct:.2f}%) | Reason: {reason}")
            self.telegram.send_trade_closed(
                symbol=symbol, entry_price=trade['entry_price'],
                exit_price=price, pnl=pnl, pnl_pct=pnl_pct, reason=reason
            )
            return True
        except (BinanceAPIException, BinanceOrderException) as e:
            logger.error(f"SELL order failed for {symbol}: {e}")
            return False

    def check_stop_loss_take_profit(self):
        """Monitor open trades for SL/TP triggers."""
        for symbol, trade in list(self.open_trades.items()):
            price = self.get_current_price(symbol)
            if price <= 0:
                continue
            if price <= trade['stop_loss']:
                logger.warning(f"Stop Loss triggered: {symbol} @ {price:.4f}")
                self.execute_sell(symbol, price, reason='STOP_LOSS')
            elif price >= trade['take_profit']:
                logger.info(f"Take Profit triggered: {symbol} @ {price:.4f}")
                self.execute_sell(symbol, price, reason='TAKE_PROFIT')

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been breached."""
        if self.start_balance > 0:
            daily_loss_pct = abs(min(0, self.daily_pnl)) / self.start_balance
            if daily_loss_pct >= config.STOP_LOSS_PERCENT / 100 * 5:
                logger.critical(f"Daily loss limit reached: {daily_loss_pct:.2%}. Stopping trading.")
                self.telegram.send_alert(f"\u26a0\ufe0f Daily loss limit reached ({daily_loss_pct:.2%}). Bot paused.")
                return True
        return False

    def send_performance_report(self):
        """Generate and send a performance report via Telegram."""
        balance = self.get_account_balance('USDT')
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
        self.telegram.send_performance_report(
            balance=balance, total_trades=total_trades,
            win_rate=win_rate, total_pnl=total_pnl,
            daily_pnl=self.daily_pnl, open_trades=len(self.open_trades),
            cycle_count=self.cycle_count
        )

    def run_cycle(self):
        """Execute one full trading cycle across all pairs."""
        self.cycle_count += 1
        logger.info(f"--- Cycle #{self.cycle_count} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
        self.check_stop_loss_take_profit()
        if self.check_daily_loss_limit():
            return
        if self.cycle_count % config.RETRAIN_INTERVAL == 0:
            logger.info("Retraining ML model...")
            for symbol in config.TRADING_PAIRS[:2]:
                df = self.get_historical_data(symbol, limit=1000)
                if not df.empty:
                    self.predictor.train(df, symbol)
        for symbol in config.TRADING_PAIRS:
            try:
                if self.is_signal_on_cooldown(symbol):
                    continue
                signal, confidence, details = self.generate_signal(symbol)
                price = details.get('current_price', self.get_current_price(symbol))
                if signal == 'BUY' and confidence >= config.MIN_ML_CONFIDENCE:
                    if symbol not in self.open_trades:
                        self.execute_buy(symbol, price, confidence, details)
                elif signal == 'SELL' and symbol in self.open_trades:
                    self.execute_sell(symbol, price, reason='SIGNAL')
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        hours_since_report = (datetime.now() - self.last_report_time).total_seconds() / 3600
        if hours_since_report >= 24:
            self.send_performance_report()
            self.last_report_time = datetime.now()
            self.daily_pnl = 0.0

    def start(self):
        """Start the trading bot main loop."""
        logger.info("=" * 60)
        logger.info("  AI CRYPTO TRADING BOT STARTED")
        logger.info(f"  Pairs: {', '.join(config.TRADING_PAIRS)}")
        logger.info(f"  Interval: {config.CANDLE_INTERVAL} | Testnet: {config.USE_TESTNET}")
        logger.info("=" * 60)
        self.start_balance = self.get_account_balance('USDT')
        logger.info(f"Starting balance: {self.start_balance:.2f} USDT")
        logger.info("Training initial ML models...")
        for symbol in config.TRADING_PAIRS[:3]:
            df = self.get_historical_data(symbol, limit=1000)
            if not df.empty:
                self.predictor.train(df, symbol)
        self.telegram.send_startup_message(
            pairs=config.TRADING_PAIRS,
            balance=self.start_balance,
            interval=config.CANDLE_INTERVAL
        )
        while True:
            try:
                self.run_cycle()
                logger.info(f"Sleeping {config.CHECK_INTERVAL}s until next cycle...")
                time.sleep(config.CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("Bot stopped by user.")
                self.telegram.send_alert("\U0001f6d1 Trading bot stopped by user.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                self.telegram.send_alert(f"\u26a0\ufe0f Bot error: {str(e)[:200]}")
                time.sleep(60)


if __name__ == '__main__':
    bot = CryptoTradingBot()
    bot.start()
