#!/usr/bin/env python3
"""
Technical Analysis - Indicators & Signal Generation
Author: kawser parvez | kawserparvez7878@gmail.com
Description: Moving Average, RSI, MACD calculations and trading signal logic.
"""

import numpy as np
import pandas as pd
import logging
from config import (
    MA_SHORT, MA_LONG, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL
)

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Technical analysis indicators and signal generation."""

    # ─────────────────────────────────────────────
    # Moving Averages
    # ─────────────────────────────────────────────

    def calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    def calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_wma(self, series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

    def ma_signal(self, df: pd.DataFrame) -> dict:
        """Generate MA crossover signal."""
        close = df['close']
        short_ma = self.calculate_ema(close, MA_SHORT)
        long_ma = self.calculate_ema(close, MA_LONG)
        sma_50 = self.calculate_sma(close, 50)
        sma_200 = self.calculate_sma(close, 200)

        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        current_price = close.iloc[-1]

        # Golden/Death cross detection
        golden_cross = prev_short <= prev_long and current_short > current_long
        death_cross = prev_short >= prev_long and current_short < current_long

        if golden_cross or (current_short > current_long and current_price > sma_50.iloc[-1]):
            signal = 'BUY'
        elif death_cross or (current_short < current_long and current_price < sma_50.iloc[-1]):
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'short_ma': float(current_short),
            'long_ma': float(current_long),
            'sma_50': float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else None,
            'sma_200': float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else None,
            'golden_cross': golden_cross,
            'death_cross': death_cross
        }

    # ─────────────────────────────────────────────
    # RSI
    # ─────────────────────────────────────────────

    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def rsi_signal(self, df: pd.DataFrame) -> dict:
        """Generate RSI-based trading signal."""
        rsi = self.calculate_rsi(df['close'], RSI_PERIOD)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if current_rsi < RSI_OVERSOLD:
            signal = 'BUY'
        elif current_rsi > RSI_OVERBOUGHT:
            signal = 'SELL'
        elif prev_rsi < RSI_OVERSOLD and current_rsi >= RSI_OVERSOLD:
            signal = 'BUY'   # RSI recovering from oversold
        elif prev_rsi > RSI_OVERBOUGHT and current_rsi <= RSI_OVERBOUGHT:
            signal = 'SELL'  # RSI falling from overbought
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'rsi': float(current_rsi),
            'overbought': RSI_OVERBOUGHT,
            'oversold': RSI_OVERSOLD
        }

    # ─────────────────────────────────────────────
    # MACD
    # ─────────────────────────────────────────────

    def calculate_macd(self, series: pd.Series):
        """MACD line, signal line, and histogram."""
        ema_fast = self.calculate_ema(series, MACD_FAST)
        ema_slow = self.calculate_ema(series, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, MACD_SIGNAL)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def macd_signal(self, df: pd.DataFrame) -> dict:
        """Generate MACD-based trading signal."""
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])

        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        curr_hist = histogram.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        prev_signal = signal_line.iloc[-2]
        prev_hist = histogram.iloc[-2]

        # Bullish crossover
        bullish_cross = prev_macd <= prev_signal and curr_macd > curr_signal
        # Bearish crossover
        bearish_cross = prev_macd >= prev_signal and curr_macd < curr_signal
        # Histogram momentum
        hist_increasing = curr_hist > prev_hist

        if bullish_cross or (curr_macd > curr_signal and hist_increasing and curr_macd > 0):
            signal = 'BUY'
        elif bearish_cross or (curr_macd < curr_signal and not hist_increasing and curr_macd < 0):
            signal = 'SELL'
        else:
            signal = 'HOLD'

        return {
            'signal': signal,
            'macd': float(curr_macd),
            'signal_line': float(curr_signal),
            'histogram': float(curr_hist),
            'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross
        }

    # ─────────────────────────────────────────────
    # Bollinger Bands
    # ─────────────────────────────────────────────

    def calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands: upper, middle, lower."""
        middle = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def bollinger_signal(self, df: pd.DataFrame) -> dict:
        """Generate Bollinger Bands signal."""
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        price = df['close'].iloc[-1]
        if price <= lower.iloc[-1]:
            signal = 'BUY'
        elif price >= upper.iloc[-1]:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        return {
            'signal': signal,
            'upper': float(upper.iloc[-1]),
            'middle': float(middle.iloc[-1]),
            'lower': float(lower.iloc[-1]),
            'price': float(price)
        }

    # ─────────────────────────────────────────────
    # Volume Analysis
    # ─────────────────────────────────────────────

    def volume_analysis(self, df: pd.DataFrame) -> dict:
        """Analyse volume trends."""
        vol = df['volume']
        vol_ma = vol.rolling(20).mean()
        vol_ratio = vol.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
        price_change = df['close'].pct_change().iloc[-1]
        if vol_ratio > 1.5 and price_change > 0:
            signal = 'BUY'
        elif vol_ratio > 1.5 and price_change < 0:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        return {'signal': signal, 'volume_ratio': float(vol_ratio),
                'current_volume': float(vol.iloc[-1]), 'avg_volume': float(vol_ma.iloc[-1])}

    # ─────────────────────────────────────────────
    # Aggregate Signal Generator
    # ─────────────────────────────────────────────

    def generate_signals(self, df: pd.DataFrame) -> dict:
        """Generate all technical analysis signals."""
        try:
            ma = self.ma_signal(df)
            rsi = self.rsi_signal(df)
            macd = self.macd_signal(df)
            bb = self.bollinger_signal(df)
            vol = self.volume_analysis(df)

            return {
                'ma_signal': ma['signal'],
                'rsi_signal': rsi['signal'],
                'macd_signal': macd['signal'],
                'bb_signal': bb['signal'],
                'volume_signal': vol['signal'],
                'ma_details': ma,
                'rsi_details': rsi,
                'macd_details': macd,
                'bb_details': bb,
                'volume_details': vol
            }
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'ma_signal': 'HOLD', 'rsi_signal': 'HOLD',
                'macd_signal': 'HOLD', 'bb_signal': 'HOLD', 'volume_signal': 'HOLD'
            }

    def get_market_summary(self, df: pd.DataFrame) -> str:
        """Return a human-readable market summary."""
        signals = self.generate_signals(df)
        rsi_val = signals['rsi_details'].get('rsi', 'N/A')
        macd_val = signals['macd_details'].get('macd', 'N/A')
        price = df['close'].iloc[-1]
        return (
            f"Price: ${price:.4f} | "
            f"MA: {signals['ma_signal']} | "
            f"RSI: {rsi_val:.1f} ({signals['rsi_signal']}) | "
            f"MACD: {macd_val:.4f} ({signals['macd_signal']}) | "
            f"BB: {signals['bb_signal']} | "
            f"Vol: {signals['volume_signal']}"
        )
