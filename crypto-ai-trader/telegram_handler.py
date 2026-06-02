#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Telegram Notification Handler
Author: kawser parvez | kawserparvez7878@gmail.com
Description: Sends real-time trading signals, performance reports,
             and alerts to Telegram bot @CodeReceiveBot.
"""

import logging
import requests
from datetime import datetime
from typing import Optional

import config

logger = logging.getLogger('TelegramHandler')


class TelegramHandler:
    """
    Handles all Telegram notifications for the trading bot.
    Sends messages to @CodeReceiveBot via Telegram Bot API.
    """

    def __init__(self):
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.enabled = config.TELEGRAM_NOTIFICATIONS_ENABLED if hasattr(config, 'TELEGRAM_NOTIFICATIONS_ENABLED') else True
        logger.info(f"TelegramHandler initialized. Chat: {self.chat_id}")

    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to the configured Telegram chat."""
        if not self.enabled:
            logger.debug(f"Telegram disabled. Message: {text[:100]}")
            return False
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.debug("Telegram message sent successfully.")
                return True
            else:
                logger.warning(f"Telegram API error {response.status_code}: {response.text[:200]}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram send error: {e}")
            return False

    def send_startup_message(self, pairs: list, balance: float, interval: str) -> bool:
        """Send bot startup notification."""
        pairs_str = ' | '.join(pairs)
        msg = (
            f"\U0001f916 <b>AI Crypto Trading Bot Started!</b>\n"
            f"{'=' * 35}\n"
            f"\U0001f4c5 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            f"\U0001f4b0 <b>Balance:</b> {balance:.2f} USDT\n"
            f"\u23f1 <b>Interval:</b> {interval}\n"
            f"\U0001f4ca <b>Pairs:</b>\n{pairs_str}\n"
            f"{'=' * 35}\n"
            f"\u2705 Bot is now monitoring markets 24/7!\n"
            f"\U0001f4e1 Signals will be sent to this chat."
        )
        return self.send_message(msg)

    def send_trade_signal(self, signal: str, symbol: str, price: float,
                          quantity: float, stop_loss: float, take_profit: float,
                          confidence: float, details: dict) -> bool:
        """Send a BUY/SELL trading signal notification."""
        emoji = "\U0001f7e2" if signal == 'BUY' else "\U0001f534"
        action = "\U0001f4c8 LONG" if signal == 'BUY' else "\U0001f4c9 SHORT"

        ta_details = details.get('ta_details', {})
        rsi_info = ta_details.get('rsi', {}).get('rsi', 'N/A')
        macd_info = ta_details.get('macd', {}).get('histogram', 'N/A')
        ta_signal = details.get('ta_signal', 'N/A')
        ml_signal = details.get('ml_signal', 'N/A')
        ml_conf = details.get('ml_confidence', 0)

        msg = (
            f"{emoji} <b>{signal} SIGNAL - {symbol}</b> {emoji}\n"
            f"{'=' * 35}\n"
            f"\U0001f4b1 <b>Action:</b> {action}\n"
            f"\U0001f4b5 <b>Price:</b> ${price:,.4f}\n"
            f"\U0001f522 <b>Quantity:</b> {quantity:.6f}\n"
            f"\U0001f6d1 <b>Stop Loss:</b> ${stop_loss:,.4f}\n"
            f"\U0001f3af <b>Take Profit:</b> ${take_profit:,.4f}\n"
            f"\U0001f9e0 <b>Confidence:</b> {confidence:.1%}\n"
            f"{'=' * 35}\n"
            f"\U0001f4ca <b>Technical Analysis:</b> {ta_signal}\n"
            f"   RSI: {rsi_info} | MACD Hist: {macd_info}\n"
            f"\U0001f916 <b>ML Prediction:</b> {ml_signal} ({ml_conf:.1%})\n"
            f"{'=' * 35}\n"
            f"\u23f0 {datetime.now().strftime('%H:%M:%S')} UTC"
        )
        return self.send_message(msg)

    def send_trade_closed(self, symbol: str, entry_price: float, exit_price: float,
                          pnl: float, pnl_pct: float, reason: str) -> bool:
        """Send trade closed notification with P&L."""
        if pnl >= 0:
            emoji = "\U0001f4b0"
            result = "PROFIT"
        else:
            emoji = "\U0001f4b8"
            result = "LOSS"

        reason_map = {
            'STOP_LOSS': '\U0001f6d1 Stop Loss Hit',
            'TAKE_PROFIT': '\U0001f3af Take Profit Hit',
            'SIGNAL': '\U0001f4ca Signal Exit'
        }
        reason_str = reason_map.get(reason, reason)

        msg = (
            f"{emoji} <b>TRADE CLOSED - {symbol}</b>\n"
            f"{'=' * 35}\n"
            f"\U0001f4cc <b>Result:</b> {result}\n"
            f"\U0001f4b5 <b>Entry:</b> ${entry_price:,.4f}\n"
            f"\U0001f4b5 <b>Exit:</b> ${exit_price:,.4f}\n"
            f"\U0001f4b0 <b>P&L:</b> {'+' if pnl >= 0 else ''}{pnl:.4f} USDT ({'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%)\n"
            f"\U0001f4cb <b>Reason:</b> {reason_str}\n"
            f"{'=' * 35}\n"
            f"\u23f0 {datetime.now().strftime('%H:%M:%S')} UTC"
        )
        return self.send_message(msg)

    def send_performance_report(self, balance: float, total_trades: int,
                                 win_rate: float, total_pnl: float,
                                 daily_pnl: float, open_trades: int,
                                 cycle_count: int) -> bool:
        """Send periodic performance report."""
        pnl_emoji = "\U0001f4c8" if total_pnl >= 0 else "\U0001f4c9"
        daily_emoji = "\U0001f7e2" if daily_pnl >= 0 else "\U0001f534"

        msg = (
            f"\U0001f4ca <b>PERFORMANCE REPORT</b>\n"
            f"{'=' * 35}\n"
            f"\U0001f4c5 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n"
            f"\U0001f4b0 <b>Balance:</b> {balance:.2f} USDT\n"
            f"{pnl_emoji} <b>Total P&L:</b> {'+' if total_pnl >= 0 else ''}{total_pnl:.4f} USDT\n"
            f"{daily_emoji} <b>Daily P&L:</b> {'+' if daily_pnl >= 0 else ''}{daily_pnl:.4f} USDT\n"
            f"\U0001f3af <b>Win Rate:</b> {win_rate:.1f}%\n"
            f"\U0001f522 <b>Total Trades:</b> {total_trades}\n"
            f"\U0001f504 <b>Open Trades:</b> {open_trades}\n"
            f"\U0001f501 <b>Cycles Run:</b> {cycle_count}\n"
            f"{'=' * 35}\n"
            f"\U0001f916 AI Crypto Bot | @CodeReceiveBot"
        )
        return self.send_message(msg)

    def send_alert(self, message: str) -> bool:
        """Send a general alert message."""
        msg = (
            f"\u26a0\ufe0f <b>BOT ALERT</b>\n"
            f"{'=' * 35}\n"
            f"{message}\n"
            f"{'=' * 35}\n"
            f"\u23f0 {datetime.now().strftime('%H:%M:%S')} UTC"
        )
        return self.send_message(msg)

    def send_market_summary(self, summaries: list) -> bool:
        """Send a market summary for all monitored pairs."""
        lines = ["\U0001f30d <b>MARKET SUMMARY</b>", "=" * 35]
        for s in summaries:
            symbol = s.get('symbol', '')
            price = s.get('price', 0)
            change = s.get('change_24h', 0)
            signal = s.get('signal', 'HOLD')
            sig_emoji = "\U0001f7e2" if signal == 'BUY' else ("\U0001f534" if signal == 'SELL' else "\u26aa")
            chg_emoji = "\U0001f4c8" if change >= 0 else "\U0001f4c9"
            lines.append(
                f"{sig_emoji} <b>{symbol}</b>: ${price:,.4f} "
                f"{chg_emoji} {'+' if change >= 0 else ''}{change:.2f}% | {signal}"
            )
        lines.append("=" * 35)
        lines.append(f"\u23f0 {datetime.now().strftime('%H:%M:%S')} UTC")
        return self.send_message('\n'.join(lines))

    def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                bot_info = response.json().get('result', {})
                logger.info(f"Telegram connected: @{bot_info.get('username', 'unknown')}")
                return True
            return False
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
