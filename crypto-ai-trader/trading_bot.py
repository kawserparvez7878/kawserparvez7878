#!/usr/bin/env python3
"""
AI-Powered Cryptocurrency Trading Bot
Author: kawser parvez
Email: kawserparvez7878@gmail.com
Description: 24/7 automated crypto trading bot with ML predictions and Telegram alerts
"""

import time
import logging
import asyncio
from datetime import datetime
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
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CryptoTradingBot:
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
            import pandas as pd
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
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None

    def get_account_balance(self, asset='USDT'):
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except BinanceAPIException as e:
            logger.error(f"Error fetching balance: {e}")
            return 0.0

    def place_order(self, symbol, side, quantity):
        try:
            order = self.client.order_market(symbol=symbol, side=side, quantity=quantity)
            logger.info(f"Order placed: {side} {quantity} {symbol}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Order error for {symbol}: {e}")
            return None

    def analyze_market(self, symbol):
        logger.info(f"Analyzing {symbol}...")
        df = self.get_historical_data(symbol)
        if df is None or len(df) < 50:
            return None
        ta_signals = self.ta.generate_signals(df)
        ml_prediction = self.ml_model.predict(df)
        current_price = self.get_current_price(symbol)
        recommendation = self._generate_recommendation(ta_signals, ml_prediction)
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'ta_signals': ta_signals,
            'ml_prediction': ml_prediction,
            'recommendation': recommendation
        }

    def _generate_recommendation(self, ta_signals, ml_prediction):
        buy_signals = sell_signals = 0
        if ta_signals:
            for key in ['ma_signal','rsi_signal','macd_signal']:
                if ta_signals.get(key) == 'BUY': buy_signals += 1
                elif ta_signals.get(key) == 'SELL': sell_signals += 1
        if ml_prediction:
            if ml_prediction.get('signal') == 'BUY': buy_signals += 2
            elif ml_prediction.get('signal') == 'SELL': sell_signals += 2
        if buy_signals > sell_signals and buy_signals >= 3: return 'STRONG_BUY'
        elif buy_signals > sell_signals: return 'BUY'
        elif sell_signals > buy_signals and sell_signals >= 3: return 'STRONG_SELL'
        elif sell_signals > buy_signals: return 'SELL'
        else: return 'HOLD'

    def execute_trade(self, symbol, recommendation, current_price):
        if recommendation in ['BUY','STRONG_BUY']:
            balance = self.get_account_balance('USDT')
            if balance > 10:
                quantity = round((balance * TRADE_QUANTITY) / current_price, 6)
                order = self.place_order(symbol, 'BUY', quantity)
                if order:
                    self.active_trades[symbol] = {
                        'entry_price': current_price, 'quantity': quantity,
                        'stop_loss': current_price * (1 - STOP_LOSS_PERCENT/100),
                        'take_profit': current_price * (1 + TAKE_PROFIT_PERCENT/100),
                        'order_id': order['orderId'], 'timestamp': datetime.now().isoformat()
                    }
                    return order
        elif recommendation in ['SELL','STRONG_SELL']:
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                order = self.place_order(symbol, 'SELL', trade['quantity'])
                if order:
                    profit = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                    self.trade_history.append({'symbol': symbol, 'entry_price': trade['entry_price'],
                        'exit_price': current_price, 'profit_percent': profit,
                        'timestamp': datetime.now().isoformat()})
                    del self.active_trades[symbol]
                    return order
        return None

    def check_stop_loss_take_profit(self, symbol, current_price):
        if symbol not in self.active_trades: return
        trade = self.active_trades[symbol]
        if current_price <= trade['stop_loss']:
            order = self.place_order(symbol, 'SELL', trade['quantity'])
            if order:
                loss = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                asyncio.run(self.telegram.send_alert(f"STOP-LOSS triggered for {symbol}\nLoss: {loss:.2f}%"))
                del self.active_trades[symbol]
        elif current_price >= trade['take_profit']:
            order = self.place_order(symbol, 'SELL', trade['quantity'])
            if order:
                profit = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                asyncio.run(self.telegram.send_alert(f"TAKE-PROFIT triggered for {symbol}\nProfit: +{profit:.2f}%"))
                del self.active_trades[symbol]

    def generate_performance_report(self):
        if not self.trade_history: return "No completed trades yet."
        total = len(self.trade_history)
        profitable = sum(1 for t in self.trade_history if t['profit_percent'] > 0)
        total_profit = sum(t['profit_percent'] for t in self.trade_history)
        return (f"Performance Report\nTotal Trades: {total}\nProfitable: {profitable}\n"
                f"Win Rate: {profitable/total*100:.1f}%\nTotal P&L: {total_profit:.2f}%\n"
                f"Active Positions: {len(self.active_trades)}")

    def run(self):
        self.is_running = True
        logger.info("Starting AI Crypto Trading Bot - 24/7 monitoring active")
        asyncio.run(self.telegram.send_alert(f"AI Crypto Trading Bot Started! Monitoring {len(TRADING_PAIRS)} pairs."))
        report_counter = 0
        while self.is_running:
            try:
                for symbol in TRADING_PAIRS:
                    analysis = self.analyze_market(symbol)
                    if analysis is None: continue
                    current_price = analysis['current_price']
                    recommendation = analysis['recommendation']
                    logger.info(f"{symbol}: ${current_price:.4f} -> {recommendation}")
                    self.check_stop_loss_take_profit(symbol, current_price)
                    if recommendation in ['BUY','STRONG_BUY','SELL','STRONG_SELL']:
                        ta = analysis['ta_signals']
                        ml = analysis['ml_prediction']
                        msg = (f"{recommendation} Signal: {symbol}\nPrice: ${current_price:.4f}\n"
                               f"MA: {ta.get('ma_signal','N/A')} | RSI: {ta.get('rsi_value',0):.1f}\n"
                               f"MACD: {ta.get('macd_signal','N/A')} | ML: {ml.get('signal','N/A')} ({ml.get('confidence',0):.1f}%)")
                        asyncio.run(self.telegram.send_alert(msg))
                        self.execute_trade(symbol, recommendation, current_price)
                    time.sleep(1)
                report_counter += 1
                if report_counter >= (3600 / CHECK_INTERVAL):
                    asyncio.run(self.telegram.send_alert(self.generate_performance_report()))
                    report_counter = 0
                time.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                self.is_running = False
                asyncio.run(self.telegram.send_alert("Trading bot stopped."))
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)


if __name__ == '__main__':
    bot = CryptoTradingBot()
    bot.run()
