#!/usr/bin/env python3
"""
Machine Learning Model - LSTM Neural Network for Crypto Price Prediction
Author: kawser parvez
"""

import numpy as np
import pandas as pd
import logging
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config import ML_LOOKBACK, ML_EPOCHS, ML_BATCH_SIZE, ML_TRAIN_SPLIT, MODEL_PATH

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


class MLModel:
    """LSTM-based ML model for crypto price prediction."""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback = ML_LOOKBACK
        self.is_trained = False
        self.model_path = MODEL_PATH
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        if TF_AVAILABLE and os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info(f"Pre-trained model loaded from {self.model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def prepare_data(self, df):
        features = [f for f in ['close','volume','high','low'] if f in df.columns]
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i - self.lookback:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        split = int(len(X) * ML_TRAIN_SPLIT)
        return X[:split], X[split:], y[:split], y[split:], scaled_data

    def build_model(self, input_shape):
        if not TF_AVAILABLE:
            return None
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(), Dropout(0.2),
            LSTM(64, return_sequences=True),
            BatchNormalization(), Dropout(0.2),
            LSTM(32, return_sequences=False),
            BatchNormalization(), Dropout(0.2),
            Dense(64, activation='relu'), Dropout(0.1),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        logger.info(f"LSTM model built: input_shape={input_shape}")
        return model

    def train(self, df):
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available.")
            return {}
        logger.info("Training LSTM model...")
        X_train, X_test, y_train, y_test, _ = self.prepare_data(df)
        if len(X_train) < 10:
            logger.warning("Insufficient training data.")
            return {}
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            ModelCheckpoint(self.model_path, save_best_only=True)
        ]
        self.model.fit(X_train, y_train, epochs=ML_EPOCHS, batch_size=ML_BATCH_SIZE,
                       validation_data=(X_test, y_test), callbacks=callbacks, verbose=1)
        metrics = self.evaluate(X_test, y_test)
        self.is_trained = True
        scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    def evaluate(self, X_test, y_test):
        if self.model is None:
            return {}
        predictions = self.model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        actual_dir = np.diff(y_test) > 0
        pred_dir = np.diff(predictions.flatten()) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
        return {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse), 'direction_accuracy': float(dir_acc)}

    def predict(self, df):
        if not TF_AVAILABLE:
            return self._fallback_prediction(df)
        if not self.is_trained or self.model is None:
            logger.info("Model not trained. Training now...")
            self.train(df)
        if self.model is None:
            return self._fallback_prediction(df)
        try:
            features = [f for f in ['close','volume','high','low'] if f in df.columns]
            data = df[features].values
            scaled_data = self.scaler.transform(data)
            X = np.array([scaled_data[-self.lookback:]])
            pred_scaled = self.model.predict(X, verbose=0)[0][0]
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = pred_scaled
            predicted_price = self.scaler.inverse_transform(dummy)[0][0]
            current_price = float(df['close'].iloc[-1])
            change_pct = (predicted_price - current_price) / current_price * 100
            if change_pct > 1.5:
                signal, confidence = 'BUY', min(abs(change_pct) * 10, 95)
            elif change_pct < -1.5:
                signal, confidence = 'SELL', min(abs(change_pct) * 10, 95)
            else:
                signal, confidence = 'HOLD', 50
            return {'signal': signal, 'confidence': confidence,
                    'predicted_price': predicted_price, 'current_price': current_price,
                    'price_change_pct': change_pct}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(df)

    def _fallback_prediction(self, df):
        try:
            close = df['close'].values
            short_ma = np.mean(close[-10:])
            long_ma = np.mean(close[-30:])
            current_price = float(close[-1])
            if short_ma > long_ma * 1.01:
                signal, confidence = 'BUY', 60
            elif short_ma < long_ma * 0.99:
                signal, confidence = 'SELL', 60
            else:
                signal, confidence = 'HOLD', 40
            return {'signal': signal, 'confidence': confidence,
                    'predicted_price': short_ma, 'current_price': current_price,
                    'price_change_pct': (short_ma - current_price) / current_price * 100}
        except Exception as e:
            logger.error(f"Fallback prediction error: {e}")
            return {'signal': 'HOLD', 'confidence': 0, 'predicted_price': 0,
                    'current_price': 0, 'price_change_pct': 0}

    def retrain(self, df):
        logger.info("Retraining model with updated data...")
        self.is_trained = False
        return self.train(df)
