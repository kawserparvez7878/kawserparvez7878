#!/usr/bin/env python3
"""
Telegram Handler - Notification System
Author: kawser parvez | kawserparvez7878@gmail.com
Description: Sends trading signals, alerts, and performance reports
             to Telegram bot @CodeReceiveBot.
"""

import logging
from datetime import datetime
from telegram import Bot
from telegram.constants import ParseMode
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramHandler:
    """Handles all Telegram notifications for the trading bot."""

    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        logger.info("TelegramHandler initialized.")

    async def send_message(self, text: str, parse_mode: str = ParseMode.MARKDOWN):
        """Send a raw message to Telegram."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode,
                disable_web_page_preview=True
            )
            logger.info("Telegram message sent.")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")

    async def send_alert(self, message: str):
        """Send a general alert message."""
        text = f"⚠️ *ALERT* | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n{message}"
        await self.send_message(text)

    async def send_trade_signal(self, symbol: str, side: str, price: float,
                                quantity: float, analysis: dict):
        """Send a formatted trade signal notification."""
        emoji = "🟢" if side == 'BUY' else "🔴"
        final_signal = analysis.get('final_signal', side)
        ta = analysis.get('ta_signals', {})
        ml = analysis.get('ml_prediction', {})

        text = (
            f"{emoji} *{side} SIGNAL* - {symbol}\n"
            f"────────────────────\n"
            f"💰 *Price:* ${price:.4f}\n"
            f"📊 *Quantity:* {quantity}\n"
            f"🎯 *Signal:* {final_signal}\n"
            f"⏰ *Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"📈 *Technical Analysis:*\n"
            f"  MA: {ta.get('ma_signal', 'N/A')}\n"
            f"  RSI: {ta.get('rsi_details', {}).get('rsi', 'N/A'):.1f} ({ta.get('rsi_signal', 'N/A')})\n"
            f"  MACD: {ta.get('macd_signal', 'N/A')}\n\n"
            f"🤖 *ML Prediction:*\n"
            f"  Direction: {ml.get('direction', 'N/A')}\n"
            f"  Confidence: {ml.get('confidence', 0)*100:.1f}%\n"
            f"  Predicted Price: ${ml.get('predicted_price', 0):.4f}\n"
        )
        await self.send_message(text)

    async def send_performance_report(self, report: dict):
        """Send a performance report."""
        if 'message' in report:
            await self.send_alert(report['message'])
            return

        text = (
            f"📊 *PERFORMANCE REPORT*\n"
            f"────────────────────\n"
            f"📅 *Date:* {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"💼 *Total Trades:* {report.get('total_trades', 0)}\n"
            f"✅ *Winning:* {report.get('winning_trades', 0)}\n"
            f"❌ *Losing:* {report.get('losing_trades', 0)}\n"
            f"🎯 *Win Rate:* {report.get('win_rate', '0%')}\n"
            f"💵 *Total Profit:* {report.get('total_profit', '$0.00')}\n"
            f"🔄 *Active Trades:* {report.get('active_trades', 0)}\n"
        )
        await self.send_message(text)

    async def send_market_update(self, symbol: str, price: float, change_24h: float,
                                  volume: float, signals: dict):
        """Send a market update for a symbol."""
        trend = "📈" if change_24h >= 0 else "📉"
        text = (
            f"{trend} *MARKET UPDATE* - {symbol}\n"
            f"────────────────────\n"
            f"💰 *Price:* ${price:.4f}\n"
            f"📉 *24h Change:* {change_24h:+.2f}%\n"
            f"📊 *Volume:* {volume:,.0f}\n\n"
            f"📌 *Signals:*\n"
            f"  MA: {signals.get('ma_signal', 'N/A')}\n"
            f"  RSI: {signals.get('rsi_signal', 'N/A')}\n"
            f"  MACD: {signals.get('macd_signal', 'N/A')}\n"
            f"  BB: {signals.get('bb_signal', 'N/A')}\n"
        )
        await self.send_message(text)

    async def send_error(self, error_msg: str):
        """Send an error notification."""
        text = (
            f"🚨 *ERROR ALERT*\n"
            f"────────────────────\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🔴 {error_msg}"
        )
        await self.send_message(text)

    async def send_startup_message(self, trading_pairs: list):
        """Send bot startup notification."""
        pairs_str = ', '.join(trading_pairs)
        text = (
            f"🚀 *AI CRYPTO TRADING BOT STARTED*\n"
            f"────────────────────\n"
            f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"📊 *Monitoring Pairs:* {pairs_str}\n"
            f"🤖 *ML Model:* LSTM Neural Network\n"
            f"📈 *Indicators:* MA, RSI, MACD, BB\n"
            f"✅ *Status:* 24/7 Active\n"
        )
        await self.send_message(text)
