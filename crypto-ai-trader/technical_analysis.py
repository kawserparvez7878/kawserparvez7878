#!/usr/bin/env python3
"""
Technical Analysis Module
Author: kawser parvez
Indicators: Moving Average, RSI, MACD, Bollinger Bands, Stochastic
"""

import numpy as np
import pandas as pd
import logging
from config import (
    MA_SHORT_PERIOD, MA_LONG_PERIOD,
    RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL
)

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Technical analysis indicators and signal generation."""

    def calculate_moving_averages(self, df):
        """Calculate Simple and Exponential Moving Averages."""
        df = df.copy()
        df['SMA_short'] = df['close'].rolling(window=MA_SHORT_PERIOD).mean()
        df['SMA_long'] = df['close'].rolling(window=MA_LONG_PERIOD).mean()
        df['EMA_short'] = df['close'].ewm(span=MA_SHORT_PERIOD, adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=MA_LONG_PERIOD, adjust=False).mean()
        return df

    def calculate_rsi(self, df, period=None):
        """Calculate Relative Strength Index."""
        if period is None:
            period = RSI_PERIOD
        df = df.copy()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(self, df):
        """Calculate MACD (Moving Average Convergence Divergence)."""
        df = df.copy()
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        return df

    def calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """Calculate Bollinger Bands."""
        df = df.copy()
        df['BB_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * std_dev)
        df['BB_lower'] = df['BB_middle'] - (bb_std * std_dev)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        return df

    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator."""
        df = df.copy()
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['Stoch_K'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=d_period).mean()
        return df

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (volatility indicator)."""
        df = df.copy()
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=period).mean()
        return df

    def calculate_volume_indicators(self, df):
        """Calculate volume-based indicators."""
        df = df.copy()
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['volume'] / df['Volume_MA']
        # On-Balance Volume
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return df

    def get_ma_signal(self, df):
        """Generate Moving Average crossover signal."""
        if len(df) < MA_LONG_PERIOD + 2:
            return 'HOLD', None, None
        last = df.iloc[-1]
        prev = df.iloc[-2]
        sma_short = last.get('SMA_short', np.nan)
        sma_long = last.get('SMA_long', np.nan)
        prev_sma_short = prev.get('SMA_short', np.nan)
        prev_sma_long = prev.get('SMA_long', np.nan)
        if pd.isna(sma_short) or pd.isna(sma_long):
            return 'HOLD', sma_short, sma_long
        # Golden cross: short MA crosses above long MA
        if sma_short > sma_long and prev_sma_short <= prev_sma_long:
            return 'BUY', sma_short, sma_long
        # Death cross: short MA crosses below long MA
        elif sma_short < sma_long and prev_sma_short >= prev_sma_long:
            return 'SELL', sma_short, sma_long
        elif sma_short > sma_long:
            return 'BUY', sma_short, sma_long
        elif sma_short < sma_long:
            return 'SELL', sma_short, sma_long
        return 'HOLD', sma_short, sma_long

    def get_rsi_signal(self, df):
        """Generate RSI-based signal."""
        if 'RSI' not in df.columns or df['RSI'].isna().all():
            return 'HOLD', 50
        rsi_value = df['RSI'].iloc[-1]
        if pd.isna(rsi_value):
            return 'HOLD', 50
        if rsi_value < RSI_OVERSOLD:
            return 'BUY', rsi_value
        elif rsi_value > RSI_OVERBOUGHT:
            return 'SELL', rsi_value
        return 'HOLD', rsi_value

    def get_macd_signal(self, df):
        """Generate MACD-based signal."""
        if 'MACD' not in df.columns or len(df) < 2:
            return 'HOLD', 0, 0
        last = df.iloc[-1]
        prev = df.iloc[-2]
        macd = last.get('MACD', np.nan)
        macd_sig = last.get('MACD_signal', np.nan)
        prev_macd = prev.get('MACD', np.nan)
        prev_sig = prev.get('MACD_signal', np.nan)
        if pd.isna(macd) or pd.isna(macd_sig):
            return 'HOLD', 0, 0
        # Bullish crossover
        if macd > macd_sig and prev_macd <= prev_sig:
            return 'BUY', macd, macd_sig
        # Bearish crossover
        elif macd < macd_sig and prev_macd >= prev_sig:
            return 'SELL', macd, macd_sig
        elif macd > macd_sig:
            return 'BUY', macd, macd_sig
        elif macd < macd_sig:
            return 'SELL', macd, macd_sig
        return 'HOLD', macd, macd_sig

    def get_bollinger_signal(self, df):
        """Generate Bollinger Bands signal."""
        if 'BB_position' not in df.columns:
            return 'HOLD'
        bb_pos = df['BB_position'].iloc[-1]
        if pd.isna(bb_pos):
            return 'HOLD'
        if bb_pos < 0.1:
            return 'BUY'   # Price near lower band - oversold
        elif bb_pos > 0.9:
            return 'SELL'  # Price near upper band - overbought
        return 'HOLD'

    def generate_signals(self, df):
        """Generate comprehensive trading signals from all indicators."""
        try:
            # Calculate all indicators
            df = self.calculate_moving_averages(df)
            df = self.calculate_rsi(df)
            df = self.calculate_macd(df)
            df = self.calculate_bollinger_bands(df)
            df = self.calculate_stochastic(df)
            df = self.calculate_atr(df)
            df = self.calculate_volume_indicators(df)

            # Get individual signals
            ma_signal, sma_short, sma_long = self.get_ma_signal(df)
            rsi_signal, rsi_value = self.get_rsi_signal(df)
            macd_signal, macd_val, macd_sig_val = self.get_macd_signal(df)
            bb_signal = self.get_bollinger_signal(df)

            # Get latest values
            last = df.iloc[-1]

            signals = {
                'ma_signal': ma_signal,
                'sma_short': float(sma_short) if sma_short and not pd.isna(sma_short) else None,
                'sma_long': float(sma_long) if sma_long and not pd.isna(sma_long) else None,
                'rsi_signal': rsi_signal,
                'rsi_value': float(rsi_value),
                'macd_signal': macd_signal,
                'macd_value': float(macd_val),
                'macd_signal_line': float(macd_sig_val),
                'macd_histogram': float(last.get('MACD_histogram', 0) or 0),
                'bb_signal': bb_signal,
                'bb_upper': float(last.get('BB_upper', 0) or 0),
                'bb_lower': float(last.get('BB_lower', 0) or 0),
                'bb_position': float(last.get('BB_position', 0.5) or 0.5),
                'atr': float(last.get('ATR', 0) or 0),
                'volume_ratio': float(last.get('Volume_ratio', 1) or 1),
                'stoch_k': float(last.get('Stoch_K', 50) or 50),
                'stoch_d': float(last.get('Stoch_D', 50) or 50),
            }

            # Overall signal strength
            buy_count = sum(1 for s in [ma_signal, rsi_signal, macd_signal, bb_signal] if s == 'BUY')
            sell_count = sum(1 for s in [ma_signal, rsi_signal, macd_signal, bb_signal] if s == 'SELL')
            signals['buy_count'] = buy_count
            signals['sell_count'] = sell_count
            signals['signal_strength'] = (buy_count - sell_count) / 4 * 100

            logger.debug(f"Signals generated: MA={ma_signal}, RSI={rsi_signal}, MACD={macd_signal}, BB={bb_signal}")
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return {
                'ma_signal': 'HOLD', 'rsi_signal': 'HOLD',
                'macd_signal': 'HOLD', 'bb_signal': 'HOLD',
                'rsi_value': 50, 'macd_value': 0, 'macd_signal_line': 0,
                'macd_histogram': 0, 'atr': 0, 'volume_ratio': 1,
                'buy_count': 0, 'sell_count': 0, 'signal_strength': 0
            }
