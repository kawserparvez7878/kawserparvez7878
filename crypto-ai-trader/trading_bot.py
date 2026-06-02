#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Main Engine
Author: kawser parvez | kawserparvez7878@gmail.com
Description: 24/7 AI-powered cryptocurrency trading bot with Binance API,
             technical analysis, ML predictions, and Telegram notifications.
"""

import time
import logging
import asyncio
from datetime import datetime
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY,
    TRADING_PAIRS, TRADE_QUANTITY, CHECK_INTERVAL,
    STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
)
from technical_analysis import TechnicalAnalysis
from ml_model import MLModel
from telegram_handler import TelegramHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('trading_bot.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class CryptoTradingBot:
    """Main AI-powered cryptocurrency trading bot."""

    def __init__(self):
        logger.info("Initializing AI Crypto Trading Bot...")
        self.client = Client(BINANCE_API_KEY, BINANCE_SECRET_KEY)
        self.ta = TechnicalAnalysis()
        self.ml_model = MLModel()
        self.telegram = TelegramHandler()
        self.active_trades = {}
        self.trade_history = []
        self.is_running = False
        logger.info("Bot initialized successfully!")

    def get_historical_data(self, symbol, interval='1h', limit=500):
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp','open','high','low','close','volume',
                'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            df.set_index('timestamp', inplace=True)
            return df
        except BinanceAPIException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def get_account_balance(self, asset='USDT'):
        try:
            return float(self.client.get_asset_balance(asset=asset)['free'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def place_order(self, symbol, side, quantity):
        try:
            order = self.client.order_market(symbol=symbol, side=side, quantity=quantity)
            logger.info(f"Order placed: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None

    def analyze_market(self, symbol):
        logger.info(f"Analyzing market for {symbol}...")
        df = self.get_historical_data(symbol, interval='1h', limit=200)
        if df is None or len(df) < 50:
            return None
        ta_signals = self.ta.generate_signals(df)
        ml_prediction = self.ml_model.predict(df)
        current_price = self.get_current_price(symbol)
        final_signal = self.combine_signals(ta_signals, ml_prediction)
        return {
            'symbol': symbol, 'current_price': current_price,
            'ta_signals': ta_signals, 'ml_prediction': ml_prediction,
            'final_signal': final_signal, 'timestamp': datetime.now().isoformat()
        }

    def combine_signals(self, ta_signals, ml_prediction):
        score = 0
        if ta_signals.get('ma_signal') == 'BUY': score += 1
        elif ta_signals.get('ma_signal') == 'SELL': score -= 1
        if ta_signals.get('rsi_signal') == 'BUY': score += 1
        elif ta_signals.get('rsi_signal') == 'SELL': score -= 1
        if ta_signals.get('macd_signal') == 'BUY': score += 1
        elif ta_signals.get('macd_signal') == 'SELL': score -= 1
        if ml_prediction.get('direction') == 'UP' and ml_prediction.get('confidence', 0) > 0.6: score += 2
        elif ml_prediction.get('direction') == 'DOWN' and ml_prediction.get('confidence', 0) > 0.6: score -= 2
        if score >= 3: return 'STRONG_BUY'
        elif score >= 1: return 'BUY'
        elif score <= -3: return 'STRONG_SELL'
        elif score <= -1: return 'SELL'
        return 'HOLD'

    def execute_trade(self, analysis):
        symbol, signal, price = analysis['symbol'], analysis['final_signal'], analysis['current_price']
        if signal in ['BUY','STRONG_BUY'] and symbol not in self.active_trades:
            if self.get_account_balance('USDT') > 10:
                order = self.place_order(symbol, 'BUY', TRADE_QUANTITY)
                if order:
                    self.active_trades[symbol] = {'entry_price': price, 'quantity': TRADE_QUANTITY,
                        'order_id': order['orderId'], 'timestamp': datetime.now().isoformat()}
                    asyncio.run(self.telegram.send_trade_signal(symbol, 'BUY', price, TRADE_QUANTITY, analysis))
        elif signal in ['SELL','STRONG_SELL'] and symbol in self.active_trades:
            trade = self.active_trades[symbol]
            order = self.place_order(symbol, 'SELL', trade['quantity'])
            if order:
                profit = (price - trade['entry_price']) * trade['quantity']
                self.trade_history.append({'symbol': symbol, 'entry_price': trade['entry_price'],
                    'exit_price': price, 'profit': profit, 'timestamp': datetime.now().isoformat()})
                del self.active_trades[symbol]
                asyncio.run(self.telegram.send_trade_signal(symbol, 'SELL', price, trade['quantity'], analysis))

    def check_stop_loss_take_profit(self):
        for symbol, trade in list(self.active_trades.items()):
            price = self.get_current_price(symbol)
            if not price: continue
            change = (price - trade['entry_price']) / trade['entry_price'] * 100
            if change <= -STOP_LOSS_PERCENT:
                if self.place_order(symbol, 'SELL', trade['quantity']):
                    del self.active_trades[symbol]
                    asyncio.run(self.telegram.send_alert(f"STOP-LOSS: {symbol} | Loss: {change:.2f}%"))
            elif change >= TAKE_PROFIT_PERCENT:
                if self.place_order(symbol, 'SELL', trade['quantity']):
                    del self.active_trades[symbol]
                    asyncio.run(self.telegram.send_alert(f"TAKE-PROFIT: {symbol} | Profit: {change:.2f}%"))

    def generate_performance_report(self):
        if not self.trade_history:
            return {'message': 'No completed trades yet.'}
        total_profit = sum(t['profit'] for t in self.trade_history)
        winning = [t for t in self.trade_history if t['profit'] > 0]
        win_rate = len(winning) / len(self.trade_history) * 100
        return {
            'total_trades': len(self.trade_history), 'winning_trades': len(winning),
            'losing_trades': len(self.trade_history) - len(winning),
            'win_rate': f"{win_rate:.2f}%", 'total_profit': f"${total_profit:.2f}",
            'active_trades': len(self.active_trades)
        }

    def run(self):
        self.is_running = True
        logger.info("Starting AI Crypto Trading Bot - 24/7 monitoring active!")
        asyncio.run(self.telegram.send_alert(
            f"AI Crypto Trading Bot Started!\nMonitoring: {', '.join(TRADING_PAIRS)}\n24/7 active."
        ))
        cycle = 0
        while self.is_running:
            try:
                cycle += 1
                logger.info(f"--- Cycle #{cycle} ---")
                for symbol in TRADING_PAIRS:
                    analysis = self.analyze_market(symbol)
                    if analysis: self.execute_trade(analysis)
                self.check_stop_loss_take_profit()
                if cycle % 24 == 0:
                    asyncio.run(self.telegram.send_performance_report(self.generate_performance_report()))
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                self.is_running = False
                asyncio.run(self.telegram.send_alert("Bot stopped."))
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)


if __name__ == '__main__':
    CryptoTradingBot().run()
