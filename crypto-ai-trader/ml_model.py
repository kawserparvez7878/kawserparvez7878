#!/usr/bin/env python3
"""
AI Crypto Trading Bot - Machine Learning Model
Author: kawser parvez | kawserparvez7878@gmail.com
Description: LSTM Neural Network for cryptocurrency price prediction.
             Trains on historical OHLCV data and predicts future price direction.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import config

logger = logging.getLogger('MLModel')
os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)


class CryptoPricePredictor:
    """
    LSTM-based price predictor for cryptocurrency trading signals.
    Uses historical OHLCV + technical indicator features to predict
    whether the next candle will be bullish or bearish.
    """

    def __init__(self):
        self.models = {}          # symbol -> keras model
        self.scalers = {}         # symbol -> MinMaxScaler
        self.is_trained = {}      # symbol -> bool
        self.lookback = config.ML_LOOKBACK
        self.features = ['open', 'high', 'low', 'close', 'volume',
                         'returns', 'volatility', 'price_range']
        logger.info("CryptoPricePredictor initialized.")

    # ----------------------------------------------------------
    # FEATURE ENGINEERING
    # ----------------------------------------------------------

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw OHLCV data."""
        data = df[['open', 'high', 'low', 'close', 'volume']].copy()

        # Price-based features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(10).std()
        data['price_range'] = (data['high'] - data['low']) / data['close']
        data['body_size'] = abs(data['close'] - data['open']) / data['close']

        # Moving averages
        data['ma_7'] = data['close'].rolling(7).mean() / data['close']
        data['ma_21'] = data['close'].rolling(21).mean() / data['close']
        data['ma_50'] = data['close'].rolling(50).mean() / data['close']

        # RSI
        delta = data['close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        data['rsi'] = 100 - (100 / (1 + rs))
        data['rsi'] = data['rsi'] / 100  # Normalize to [0,1]

        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = (ema12 - ema26) / data['close']

        # Volume features
        data['volume_ma'] = data['volume'].rolling(20).mean()
        data['volume_ratio'] = data['volume'] / (data['volume_ma'] + 1e-10)

        # Target: 1 if next close > current close, else 0
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)

        data.dropna(inplace=True)
        return data

    def create_sequences(self, data: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM input."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    # ----------------------------------------------------------
    # MODEL ARCHITECTURE
    # ----------------------------------------------------------

    def build_model(self, input_shape: Tuple[int, int]) -> 'tf.keras.Model':
        """Build LSTM neural network architecture."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),

            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),

            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),

            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary: BUY(1) or SELL(0)
        ])
        model.compile(
            optimizer=Adam(learning_rate=config.ML_LEARNING_RATE if hasattr(config, 'ML_LEARNING_RATE') else 0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    # ----------------------------------------------------------
    # TRAINING
    # ----------------------------------------------------------

    def train(self, df: pd.DataFrame, symbol: str) -> dict:
        """Train the LSTM model on historical data for a given symbol."""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Using rule-based fallback.")
            self.is_trained[symbol] = False
            return {}

        logger.info(f"Training ML model for {symbol}...")
        try:
            data = self.prepare_features(df)
            if len(data) < self.lookback + 50:
                logger.warning(f"Insufficient data for {symbol}: {len(data)} rows")
                return {}

            feature_cols = [c for c in data.columns if c != 'target']
            X_raw = data[feature_cols].values
            y_raw = data['target'].values

            # Scale features
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_scaled = scaler.fit_transform(X_raw)
            self.scalers[symbol] = scaler

            # Create sequences
            X, y = self.create_sequences(X_scaled, y_raw)

            # Train/test split
            split = int(len(X) * config.ML_TRAIN_SPLIT if hasattr(config, 'ML_TRAIN_SPLIT') else 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            # Build model
            model = self.build_model((X_train.shape[1], X_train.shape[2]))

            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
                ModelCheckpoint(
                    filepath=os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_model.h5'),
                    save_best_only=True, monitor='val_loss'
                )
            ]

            # Train
            history = model.fit(
                X_train, y_train,
                epochs=config.ML_EPOCHS,
                batch_size=config.ML_BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=0
            )

            # Evaluate
            y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            accuracy = np.mean(y_pred == y_test)
            logger.info(f"{symbol} model trained | Accuracy: {accuracy:.2%} | Epochs: {len(history.history['loss'])}")

            self.models[symbol] = model
            self.is_trained[symbol] = True

            # Save scaler
            joblib.dump(scaler, os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_scaler.pkl'))

            return {'accuracy': accuracy, 'epochs': len(history.history['loss'])}

        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            self.is_trained[symbol] = False
            return {}

    # ----------------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------------

    def predict(self, df: pd.DataFrame, symbol: str) -> Tuple[str, float, dict]:
        """
        Predict trading signal for the given symbol.
        Returns: (signal, confidence, details)
        signal: 'BUY', 'SELL', or 'HOLD'
        confidence: float between 0 and 1
        """
        # Load model from disk if not in memory
        if symbol not in self.models or not self.is_trained.get(symbol, False):
            model_path = os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_model.h5')
            scaler_path = os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_scaler.pkl')
            if os.path.exists(model_path) and os.path.exists(scaler_path) and TF_AVAILABLE:
                try:
                    self.models[symbol] = load_model(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    self.is_trained[symbol] = True
                    logger.info(f"Loaded saved model for {symbol}")
                except Exception as e:
                    logger.warning(f"Could not load model for {symbol}: {e}")
                    return self._rule_based_fallback(df)
            else:
                logger.info(f"No trained model for {symbol}. Using rule-based fallback.")
                return self._rule_based_fallback(df)

        try:
            data = self.prepare_features(df)
            if len(data) < self.lookback:
                return self._rule_based_fallback(df)

            feature_cols = [c for c in data.columns if c != 'target']
            X_raw = data[feature_cols].values[-self.lookback:]
            X_scaled = self.scalers[symbol].transform(X_raw)
            X_input = X_scaled.reshape(1, self.lookback, X_scaled.shape[1])

            prob = float(self.models[symbol].predict(X_input, verbose=0)[0][0])

            if prob >= 0.65:
                signal, confidence = 'BUY', prob
            elif prob <= 0.35:
                signal, confidence = 'SELL', 1 - prob
            else:
                signal, confidence = 'HOLD', 0.5

            details = {'probability': prob, 'model': 'LSTM', 'lookback': self.lookback}
            logger.debug(f"{symbol} ML prediction: {signal} | prob={prob:.4f}")
            return signal, confidence, details

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}")
            return self._rule_based_fallback(df)

    def _rule_based_fallback(self, df: pd.DataFrame) -> Tuple[str, float, dict]:
        """Simple rule-based fallback when ML model is unavailable."""
        try:
            close = df['close'].values
            if len(close) < 21:
                return 'HOLD', 0.5, {'model': 'fallback'}
            ma_short = close[-9:].mean()
            ma_long = close[-21:].mean()
            if ma_short > ma_long * 1.005:
                return 'BUY', 0.6, {'model': 'MA_fallback'}
            elif ma_short < ma_long * 0.995:
                return 'SELL', 0.6, {'model': 'MA_fallback'}
            return 'HOLD', 0.5, {'model': 'MA_fallback'}
        except Exception:
            return 'HOLD', 0.5, {'model': 'fallback'}

    def load_model_for_symbol(self, symbol: str) -> bool:
        """Attempt to load a pre-trained model for a symbol."""
        model_path = os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_model.h5')
        scaler_path = os.path.join(config.MODEL_SAVE_PATH, f'{symbol}_scaler.pkl')
        if os.path.exists(model_path) and os.path.exists(scaler_path) and TF_AVAILABLE:
            try:
                self.models[symbol] = load_model(model_path)
                self.scalers[symbol] = joblib.load(scaler_path)
                self.is_trained[symbol] = True
                logger.info(f"Model loaded for {symbol}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model for {symbol}: {e}")
        return False
