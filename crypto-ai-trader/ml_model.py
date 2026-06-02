#!/usr/bin/env python3
"""
ML Model - LSTM Neural Network for Crypto Price Prediction
Author: kawser parvez | kawserparvez7878@gmail.com
Description: LSTM-based deep learning model for predicting cryptocurrency
             price movements with training, evaluation, and prediction.
"""

import numpy as np
import pandas as pd
import logging
import os
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from config import ML_LOOKBACK, ML_EPOCHS, ML_BATCH_SIZE, MODEL_SAVE_PATH

logger = logging.getLogger(__name__)


class MLModel:
    """LSTM Neural Network for cryptocurrency price prediction."""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = ML_LOOKBACK
        self.is_trained = False
        self.model_path = MODEL_SAVE_PATH
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                self.is_trained = True
                logger.info(f"Pre-trained model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Will train new model.")

    def build_model(self, input_shape):
        """Build the LSTM neural network architecture."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        logger.info(f"LSTM model built: {model.count_params()} parameters")
        return model

    def prepare_features(self, df):
        """Prepare feature set from OHLCV data."""
        f = pd.DataFrame(index=df.index)
        f['close'] = df['close']
        f['open'] = df['open']
        f['high'] = df['high']
        f['low'] = df['low']
        f['volume'] = df['volume']
        f['price_change'] = df['close'].pct_change()
        f['high_low_ratio'] = df['high'] / df['low']
        f['close_open_ratio'] = df['close'] / df['open']
        f['ma_7'] = df['close'].rolling(7).mean()
        f['ma_21'] = df['close'].rolling(21).mean()
        f['ma_50'] = df['close'].rolling(50).mean()
        f['std_7'] = df['close'].rolling(7).std()
        f['std_21'] = df['close'].rolling(21).std()
        f['momentum_7'] = df['close'] / df['close'].shift(7) - 1
        f['momentum_14'] = df['close'] / df['close'].shift(14) - 1
        f['volume_ma_7'] = df['volume'].rolling(7).mean()
        f['volume_ratio'] = df['volume'] / f['volume_ma_7']
        f.dropna(inplace=True)
        return f

    def create_sequences(self, data, target):
        """Create time-series sequences for LSTM training."""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def train(self, df):
        """Train the LSTM model on historical data."""
        logger.info("Starting model training...")
        features = self.prepare_features(df)
        if len(features) < self.lookback + 50:
            return {'error': 'Insufficient data'}

        feature_cols = [c for c in features.columns if c != 'close']
        X_scaled = self.scaler.fit_transform(features[feature_cols].values)
        y_scaled = self.y_scaler.fit_transform(features['close'].values.reshape(-1, 1)).flatten()

        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        split = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split], X_seq[split:]
        y_train, y_val = y_seq[:split], y_seq[split:]

        self.model = self.build_model((self.lookback, X_scaled.shape[1]))
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ModelCheckpoint(self.model_path, save_best_only=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6)
        ]
        history = self.model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=ML_EPOCHS, batch_size=ML_BATCH_SIZE, callbacks=callbacks, verbose=1
        )
        y_pred = self.y_scaler.inverse_transform(self.model.predict(X_val))
        y_actual = self.y_scaler.inverse_transform(y_val.reshape(-1, 1))
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        self.is_trained = True
        logger.info(f"Training complete. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        return {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse),
                'train_loss': float(history.history['loss'][-1]),
                'val_loss': float(history.history['val_loss'][-1])}

    def predict(self, df):
        """Predict price direction and confidence."""
        if not self.is_trained or self.model is None:
            logger.info("Model not trained. Training now...")
            self.train(df)
        try:
            features = self.prepare_features(df)
            if len(features) < self.lookback:
                return {'direction': 'UNKNOWN', 'confidence': 0.0, 'predicted_price': None}
            feature_cols = [c for c in features.columns if c != 'close']
            X_raw = features[feature_cols].values[-self.lookback:]
            X_input = self.scaler.transform(X_raw).reshape(1, self.lookback, -1)
            pred_price = self.y_scaler.inverse_transform(self.model.predict(X_input, verbose=0))[0][0]
            current_price = features['close'].iloc[-1]
            pct = (pred_price - current_price) / current_price * 100
            if pct > 0.5:
                direction, confidence = 'UP', min(0.5 + abs(pct) / 10, 0.95)
            elif pct < -0.5:
                direction, confidence = 'DOWN', min(0.5 + abs(pct) / 10, 0.95)
            else:
                direction, confidence = 'NEUTRAL', 0.5
            return {'direction': direction, 'confidence': float(confidence),
                    'predicted_price': float(pred_price), 'current_price': float(current_price),
                    'price_change_pct': float(pct)}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'direction': 'UNKNOWN', 'confidence': 0.0, 'predicted_price': None}

    def retrain(self, df):
        """Retrain the model with new data."""
        logger.info("Retraining model...")
        self.is_trained = False
        return self.train(df)

    def get_model_summary(self):
        """Get model architecture summary."""
        if self.model:
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "Model not built yet."
