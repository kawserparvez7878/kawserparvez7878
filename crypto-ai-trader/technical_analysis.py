#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Technical Analysis Module
Author: kawser parvez | kawserparvez7878@gmail.com
Description: Calculates technical indicators (MA, RSI, MACD, Bollinger Bands)
             and generates BUY/SELL/HOLD signals based on combined indicator logic.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict

import config

logger = logging.getLogger('TechnicalAnalysis')


class TechnicalAnalysis:
    """
    Computes technical indicators and generates trading signals.
    Indicators: EMA, SMA, RSI, MACD, Bollinger Bands, ATR, Stochastic.
    """

    def __init__(self):
        logger.info("TechnicalAnalysis engine initialized.")

    # ----------------------------------------------------------
    # MOVING AVERAGES
    # ----------------------------------------------------------

    def sma(self, series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    def calculate_moving_averages(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate short, medium, and long moving averages."""
        close = df['close']
        return {
            'sma_9':  self.sma(close, config.MA_SHORT),
            'sma_21': self.sma(close, config.MA_LONG),
            'ema_9':  self.ema(close, config.MA_SHORT),
            'ema_21': self.ema(close, config.MA_LONG),
            'ema_50': self.ema(close, 50),
            'ema_200': self.ema(close, 200),
        }

    def ma_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Generate signal from Moving Average crossover."""
        mas = self.calculate_moving_averages(df)
        ema_fast = mas['ema_9'].iloc[-1]
        ema_slow = mas['ema_21'].iloc[-1]
        ema_fast_prev = mas['ema_9'].iloc[-2]
        ema_slow_prev = mas['ema_21'].iloc[-2]
        ema_50 = mas['ema_50'].iloc[-1]
        close = df['close'].iloc[-1]

        # Golden cross / Death cross
        golden_cross = ema_fast_prev <= ema_slow_prev and ema_fast > ema_slow
        death_cross = ema_fast_prev >= ema_slow_prev and ema_fast < ema_slow

        # Trend filter: price above/below EMA50
        uptrend = close > ema_50
        downtrend = close < ema_50

        details = {
            'ema_fast': round(ema_fast, 4), 'ema_slow': round(ema_slow, 4),
            'ema_50': round(ema_50, 4), 'golden_cross': golden_cross,
            'death_cross': death_cross, 'uptrend': uptrend
        }

        if golden_cross and uptrend:
            return 'BUY', details
        elif death_cross and downtrend:
            return 'SELL', details
        elif ema_fast > ema_slow and uptrend:
            return 'BUY', details
        elif ema_fast < ema_slow and downtrend:
            return 'SELL', details
        return 'HOLD', details

    # ----------------------------------------------------------
    # RSI
    # ----------------------------------------------------------

    def calculate_rsi(self, series: pd.Series, period: int = None) -> pd.Series:
        """Relative Strength Index."""
        period = period or config.RSI_PERIOD
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def rsi_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Generate signal from RSI."""
        rsi = self.calculate_rsi(df['close'])
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        details = {
            'rsi': round(current_rsi, 2),
            'overbought': config.RSI_OVERBOUGHT,
            'oversold': config.RSI_OVERSOLD
        }

        # RSI crossing out of oversold = BUY
        if prev_rsi <= config.RSI_OVERSOLD and current_rsi > config.RSI_OVERSOLD:
            return 'BUY', details
        # RSI crossing out of overbought = SELL
        elif prev_rsi >= config.RSI_OVERBOUGHT and current_rsi < config.RSI_OVERBOUGHT:
            return 'SELL', details
        elif current_rsi < config.RSI_OVERSOLD:
            return 'BUY', details
        elif current_rsi > config.RSI_OVERBOUGHT:
            return 'SELL', details
        return 'HOLD', details

    # ----------------------------------------------------------
    # MACD
    # ----------------------------------------------------------

    def calculate_macd(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, Signal line, and Histogram."""
        ema_fast = self.ema(series, config.MACD_FAST)
        ema_slow = self.ema(series, config.MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, config.MACD_SIGNAL)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def macd_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Generate signal from MACD crossover."""
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        macd_curr = macd_line.iloc[-1]
        signal_curr = signal_line.iloc[-1]
        macd_prev = macd_line.iloc[-2]
        signal_prev = signal_line.iloc[-2]
        hist_curr = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]

        # Bullish crossover
        bullish_cross = macd_prev <= signal_prev and macd_curr > signal_curr
        # Bearish crossover
        bearish_cross = macd_prev >= signal_prev and macd_curr < signal_curr
        # Histogram momentum
        hist_rising = hist_curr > hist_prev

        details = {
            'macd': round(macd_curr, 6), 'signal': round(signal_curr, 6),
            'histogram': round(hist_curr, 6), 'bullish_cross': bullish_cross,
            'bearish_cross': bearish_cross
        }

        if bullish_cross and macd_curr < 0:
            return 'BUY', details  # Crossover below zero = strong buy
        elif bearish_cross and macd_curr > 0:
            return 'SELL', details  # Crossover above zero = strong sell
        elif bullish_cross:
            return 'BUY', details
        elif bearish_cross:
            return 'SELL', details
        elif macd_curr > signal_curr and hist_rising:
            return 'BUY', details
        elif macd_curr < signal_curr and not hist_rising:
            return 'SELL', details
        return 'HOLD', details

    # ----------------------------------------------------------
    # BOLLINGER BANDS
    # ----------------------------------------------------------

    def calculate_bollinger_bands(self, series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Upper, Middle, and Lower Bollinger Bands."""
        middle = self.sma(series, config.BB_PERIOD)
        std = series.rolling(config.BB_PERIOD).std()
        upper = middle + (config.BB_STD_DEV * std)
        lower = middle - (config.BB_STD_DEV * std)
        return upper, middle, lower

    def bollinger_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Generate signal from Bollinger Bands."""
        upper, middle, lower = self.calculate_bollinger_bands(df['close'])
        close = df['close'].iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        middle_val = middle.iloc[-1]
        bandwidth = (upper_val - lower_val) / middle_val

        details = {
            'upper': round(upper_val, 4), 'middle': round(middle_val, 4),
            'lower': round(lower_val, 4), 'bandwidth': round(bandwidth, 4),
            'close': round(close, 4)
        }

        if close <= lower_val:
            return 'BUY', details   # Price at lower band = oversold
        elif close >= upper_val:
            return 'SELL', details  # Price at upper band = overbought
        elif close < middle_val:
            return 'BUY', details
        elif close > middle_val:
            return 'SELL', details
        return 'HOLD', details

    # ----------------------------------------------------------
    # ATR (Average True Range)
    # ----------------------------------------------------------

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range for volatility measurement."""
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    # ----------------------------------------------------------
    # STOCHASTIC OSCILLATOR
    # ----------------------------------------------------------

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic %K and %D."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d

    def stochastic_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Generate signal from Stochastic Oscillator."""
        k, d = self.calculate_stochastic(df)
        k_curr, d_curr = k.iloc[-1], d.iloc[-1]
        k_prev, d_prev = k.iloc[-2], d.iloc[-2]

        details = {'stoch_k': round(k_curr, 2), 'stoch_d': round(d_curr, 2)}

        if k_prev <= d_prev and k_curr > d_curr and k_curr < 30:
            return 'BUY', details
        elif k_prev >= d_prev and k_curr < d_curr and k_curr > 70:
            return 'SELL', details
        elif k_curr < 20:
            return 'BUY', details
        elif k_curr > 80:
            return 'SELL', details
        return 'HOLD', details

    # ----------------------------------------------------------
    # VOLUME ANALYSIS
    # ----------------------------------------------------------

    def volume_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Analyze volume for confirmation."""
        volume = df['volume']
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[-2]

        details = {'volume_ratio': round(vol_ratio, 2), 'price_change': round(price_change, 4)}

        if vol_ratio > 1.5 and price_change > 0:
            return 'BUY', details   # High volume + price up = bullish
        elif vol_ratio > 1.5 and price_change < 0:
            return 'SELL', details  # High volume + price down = bearish
        return 'HOLD', details

    # ----------------------------------------------------------
    # COMBINED SIGNAL GENERATION
    # ----------------------------------------------------------

    def generate_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """
        Combine all indicators to generate a final trading signal.
        Uses a weighted voting system across all indicators.
        Returns: (signal, details_dict)
        """
        if len(df) < 200:
            return 'HOLD', {'reason': 'Insufficient data'}

        # Get individual signals
        ma_sig, ma_det = self.ma_signal(df)
        rsi_sig, rsi_det = self.rsi_signal(df)
        macd_sig, macd_det = self.macd_signal(df)
        bb_sig, bb_det = self.bollinger_signal(df)
        stoch_sig, stoch_det = self.stochastic_signal(df)
        vol_sig, vol_det = self.volume_signal(df)

        # Weighted voting (MA and MACD have higher weight)
        weights = {'MA': 2, 'RSI': 1.5, 'MACD': 2, 'BB': 1, 'STOCH': 1, 'VOL': 1}
        signals = {
            'MA': ma_sig, 'RSI': rsi_sig, 'MACD': macd_sig,
            'BB': bb_sig, 'STOCH': stoch_sig, 'VOL': vol_sig
        }

        buy_score = sum(weights[k] for k, v in signals.items() if v == 'BUY')
        sell_score = sum(weights[k] for k, v in signals.items() if v == 'SELL')
        total_weight = sum(weights.values())

        buy_pct = buy_score / total_weight
        sell_pct = sell_score / total_weight

        # Determine final signal
        if buy_pct >= 0.45 and buy_pct > sell_pct:
            final_signal = 'BUY'
        elif sell_pct >= 0.45 and sell_pct > buy_pct:
            final_signal = 'SELL'
        else:
            final_signal = 'HOLD'

        # ATR for volatility context
        atr = self.calculate_atr(df)
        atr_val = atr.iloc[-1]

        details = {
            'final_signal': final_signal,
            'buy_score': round(buy_pct, 3),
            'sell_score': round(sell_pct, 3),
            'signals': signals,
            'atr': round(atr_val, 4),
            'ma': ma_det, 'rsi': rsi_det, 'macd': macd_det,
            'bb': bb_det, 'stoch': stoch_det, 'volume': vol_det
        }

        logger.debug(f"TA Signal: {final_signal} | BUY: {buy_pct:.2%} | SELL: {sell_pct:.2%} | Indicators: {signals}")
        return final_signal, details

    def get_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[float, float]:
        """Calculate basic support and resistance levels."""
        recent = df.tail(window)
        support = recent['low'].min()
        resistance = recent['high'].max()
        return support, resistance

    def get_trend(self, df: pd.DataFrame) -> str:
        """Determine overall market trend."""
        close = df['close']
        ema_50 = self.ema(close, 50).iloc[-1]
        ema_200 = self.ema(close, 200).iloc[-1]
        current_price = close.iloc[-1]
        if current_price > ema_50 > ema_200:
            return 'STRONG_UPTREND'
        elif current_price > ema_50:
            return 'UPTREND'
        elif current_price < ema_50 < ema_200:
            return 'STRONG_DOWNTREND'
        elif current_price < ema_50:
            return 'DOWNTREND'
        return 'SIDEWAYS'
