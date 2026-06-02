#!/usr/bin/env python3
"""
Telegram Handler - Notification System for AI Crypto Trading Bot
Author: kawser parvez
Bot: @CodeReceiveBot
"""

import logging
import asyncio
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramHandler:
    """Handles all Telegram notifications for the trading bot."""

    def __init__(self):
        self.bot = Bot(token=TELEGRAM_TOKEN)
        self.chat_id = TELEGRAM_CHAT_ID
        logger.info("Telegram handler initialized.")

    async def send_message(self, text: str, parse_mode: str = 'Markdown'):
        """Send a message via Telegram bot."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            logger.info(f"Telegram message sent: {text[:50]}...")
            return True
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending Telegram message: {e}")
            return False

    async def send_alert(self, message: str):
        """Send a trading alert."""
        return await self.send_message(message)

    async def send_trade_signal(self, symbol: str, signal: str, price: float,
                                 ta_signals: dict, ml_prediction: dict):
        """Send a formatted trading signal notification."""
        emoji = "\U0001F7E2" if 'BUY' in signal else "\U0001F534"
        confidence = ml_prediction.get('confidence', 0)
        predicted_price = ml_prediction.get('predicted_price', price)
        change_pct = ml_prediction.get('price_change_pct', 0)

        message = (
            f"{emoji} *{signal} SIGNAL DETECTED*\n"
            f"{'='*30}\n"
            f"*Pair:* `{symbol}`\n"
            f"*Current Price:* `${price:.4f}`\n"
            f"*ML Predicted:* `${predicted_price:.4f}` ({change_pct:+.2f}%)\n"
            f"*ML Confidence:* `{confidence:.1f}%`\n"
            f"\n"
            f"*Technical Indicators:*\n"
            f"  MA Signal: `{ta_signals.get('ma_signal', 'N/A')}`\n"
            f"  RSI: `{ta_signals.get('rsi_value', 0):.1f}` ({ta_signals.get('rsi_signal', 'N/A')})\n"
            f"  MACD: `{ta_signals.get('macd_signal', 'N/A')}`\n"
            f"  Bollinger: `{ta_signals.get('bb_signal', 'N/A')}`\n"
            f"  Volume Ratio: `{ta_signals.get('volume_ratio', 1):.2f}x`\n"
            f"\n"
            f"*Signal Strength:* `{ta_signals.get('buy_count', 0)} BUY / {ta_signals.get('sell_count', 0)} SELL`\n"
            f"*Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}`"
        )
        return await self.send_message(message)

    async def send_trade_executed(self, symbol: str, side: str, quantity: float,
                                   price: float, order_id: str):
        """Send notification when a trade is executed."""
        emoji = "\U0001F4B8" if side == 'BUY' else "\U0001F4B0"
        message = (
            f"{emoji} *TRADE EXECUTED*\n"
            f"*Action:* `{side}`\n"
            f"*Pair:* `{symbol}`\n"
            f"*Quantity:* `{quantity}`\n"
            f"*Price:* `${price:.4f}`\n"
            f"*Value:* `${quantity * price:.2f}`\n"
            f"*Order ID:* `{order_id}`\n"
            f"*Time:* `{datetime.now().strftime('%H:%M:%S UTC')}`"
        )
        return await self.send_message(message)

    async def send_stop_loss(self, symbol: str, entry_price: float,
                              exit_price: float, loss_pct: float):
        """Send stop-loss notification."""
        message = (
            f"\U0001F6D1 *STOP-LOSS TRIGGERED*\n"
            f"*Pair:* `{symbol}`\n"
            f"*Entry:* `${entry_price:.4f}`\n"
            f"*Exit:* `${exit_price:.4f}`\n"
            f"*Loss:* `{loss_pct:.2f}%`\n"
            f"*Time:* `{datetime.now().strftime('%H:%M:%S UTC')}`"
        )
        return await self.send_message(message)

    async def send_take_profit(self, symbol: str, entry_price: float,
                                exit_price: float, profit_pct: float):
        """Send take-profit notification."""
        message = (
            f"\U00002705 *TAKE-PROFIT TRIGGERED*\n"
            f"*Pair:* `{symbol}`\n"
            f"*Entry:* `${entry_price:.4f}`\n"
            f"*Exit:* `${exit_price:.4f}`\n"
            f"*Profit:* `+{profit_pct:.2f}%`\n"
            f"*Time:* `{datetime.now().strftime('%H:%M:%S UTC')}`"
        )
        return await self.send_message(message)

    async def send_performance_report(self, report_data: dict):
        """Send a detailed performance report."""
        total = report_data.get('total_trades', 0)
        profitable = report_data.get('profitable_trades', 0)
        win_rate = report_data.get('win_rate', 0)
        total_pnl = report_data.get('total_pnl', 0)
        avg_pnl = report_data.get('avg_pnl', 0)
        active = report_data.get('active_positions', 0)
        best_trade = report_data.get('best_trade', 0)
        worst_trade = report_data.get('worst_trade', 0)

        pnl_emoji = "\U0001F4C8" if total_pnl >= 0 else "\U0001F4C9"
        message = (
            f"\U0001F4CA *PERFORMANCE REPORT*\n"
            f"{'='*30}\n"
            f"*Period:* `{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}`\n"
            f"\n"
            f"*Trade Statistics:*\n"
            f"  Total Trades: `{total}`\n"
            f"  Profitable: `{profitable}`\n"
            f"  Win Rate: `{win_rate:.1f}%`\n"
            f"  Active Positions: `{active}`\n"
            f"\n"
            f"*P&L Summary:* {pnl_emoji}\n"
            f"  Total P&L: `{total_pnl:+.2f}%`\n"
            f"  Avg per Trade: `{avg_pnl:+.2f}%`\n"
            f"  Best Trade: `+{best_trade:.2f}%`\n"
            f"  Worst Trade: `{worst_trade:.2f}%`\n"
        )
        return await self.send_message(message)

    async def send_bot_status(self, status: str, pairs: list, uptime: str):
        """Send bot status update."""
        status_emoji = "\U0001F7E2" if status == 'RUNNING' else "\U0001F534"
        pairs_str = ', '.join(pairs[:5]) + ('...' if len(pairs) > 5 else '')
        message = (
            f"{status_emoji} *BOT STATUS: {status}*\n"
            f"*Monitoring:* `{pairs_str}`\n"
            f"*Pairs Count:* `{len(pairs)}`\n"
            f"*Uptime:* `{uptime}`\n"
            f"*Time:* `{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}`"
        )
        return await self.send_message(message)

    async def send_error_alert(self, error_msg: str):
        """Send error alert."""
        message = (
            f"\U0000274C *BOT ERROR*\n"
            f"*Error:* `{error_msg[:200]}`\n"
            f"*Time:* `{datetime.now().strftime('%H:%M:%S UTC')}`"
        )
        return await self.send_message(message)
