# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
import joblib
from config import *
import lightgbm as lgb
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from preprocessing import (
    load_feature_cols,
    fit_preprocessor,
    transform_with_preprocessor,
)

# ============================================================================
# PYTORCH DATASET
# ============================================================================

class BitcoinDataset(Dataset):
    def __init__(self, df, lookback=LOOKBACK):
        self.lookback = lookback
        self.df = df.reset_index(drop=True)
        feature_cols = [c for c in df.columns if c != 'target']
        self.X_data = df[feature_cols].values
        self.y_data = df['target'].values
        self.valid_indices = list(range(lookback, len(df)))
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        start_idx = real_idx - self.lookback
        end_idx = real_idx
        x = self.X_data[start_idx:end_idx]
        y = self.y_data[real_idx]
        return torch.FloatTensor(x), torch.tensor(y, dtype=torch.long)


# ============================================================================
# GRU MODEL (не используется сейчас, но оставлен для совместимости)
# ============================================================================

class GRUPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        if torch.isnan(out).any():
            out = torch.nan_to_num(out, nan=0.0)
        return self.fc(out)


def train_model(model, train_loader, val_loader, device):
    """Обучение GRU модели (если используется)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(EPOCHS_PER_TRAIN):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if torch.isnan(x).any():
                continue
            optimizer.zero_grad()
            out = model(x)
            if torch.isnan(out).any():
                continue
            loss = criterion(out, y)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        preds, trues = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                if torch.isnan(x).any():
                    continue
                out = model(x)
                if torch.isnan(out).any():
                    continue
                loss = criterion(out, y)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                    trues.extend(y.cpu().numpy())
        
        if len(preds) == 0:
            continue
        
        val_loss /= len(val_loader)
        acc = accuracy_score(trues, preds)
        scheduler.step(val_loss)

        print(f"Эпоха {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {acc:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping")
                break


# ============================================================================
# LIGHTGBM (основная модель)
# ============================================================================

def train_lgbm(X_train, y_train, X_val, y_val, params=None, df_for_cols=None):
    """
    Обучение LightGBM с единым препроцессингом
    df_for_cols нужен только при первом запуске, чтобы инферить feature_cols.
    """
    print("\nОбучаем LightGBM...")

    # ---- 1. Определяем список фичей ----
    # если X_train/X_val - это уже numpy, то предполагаем,
    # что feature_cols заранее сохранены (через df_for_cols или до этого этапа)
    if isinstance(X_train, pd.DataFrame):
        feature_cols = load_feature_cols(df_for_cols if df_for_cols is not None else X_train)
        X_train = X_train[feature_cols].copy()
        X_val = X_val[feature_cols].copy()
    else:
        # numpy -> просто грузим feature_cols, структуры должны совпасть
        feature_cols = joblib.load(SELECTED_FEATURE_COLS_PATH) if os.path.exists(
            SELECTED_FEATURE_COLS_PATH
        ) else joblib.load(FEATURE_COLS_PATH)

    # ---- 2. Обучаем / подгружаем препроцессор ----
    X_train_scaled, imputer, scaler = fit_preprocessor(X_train)
    X_val_scaled = transform_with_preprocessor(X_val, imputer, scaler)

    # ---- 3. Обучение LGBM ----
    lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
    lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)

    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'device': 'cpu'
        }

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_pred_proba = model.predict(X_val_scaled)
    val_pred = (val_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    print(f"LightGBM Val Acc: {acc:.3f}\n")

    joblib.dump(model, LGBM_MODEL_PATH)
    return model, feature_cols, imputer, scaler


# ============================================================================
# КАЛИБРОВКА ВЕРОЯТНОСТЕЙ (ИСПРАВЛЕНО!)
# ============================================================================

def calibrate_lgbm_probs(lgbm_model, X_train, y_train, X_val):
    """
    Калибровка вероятностей LightGBM через Isotonic Regression
    ИСПРАВЛЕНА: теперь работает с Booster объектом!
    """
    print("\nКалибруем вероятности LightGBM...")
    
    try:
        # 1. Получаем train predictions для обучения калибратора
        train_probs = lgbm_model.predict(X_train)
        
        # 2. Обучаем Isotonic Regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(train_probs, y_train)
        
        # 3. Калибруем validation predictions
        val_probs = lgbm_model.predict(X_val)
        calibrated_probs = iso_reg.transform(val_probs)
        
        # 4. Сохраняем калибратор
        joblib.dump(iso_reg, 'data/isotonic_calibrator.pkl')
        
        print("✅ Калибровка успешна!")
        return calibrated_probs
        
    except Exception as e:
        print(f"⚠️ Калибровка не удалась: {e}")
        print("   Используем оригинальные вероятности")
        val_probs = lgbm_model.predict(X_val)
        return val_probs


# ============================================================================
# STACKING ENSEMBLE
# ============================================================================

def train_stacking(tft_probs_train, lgbm_probs_train, y_train, 
                   tft_probs_val, lgbm_probs_val, y_val):
    """
    Мета-модель для комбинирования TFT и LightGBM
    """
    print("\nОбучаем Stacking...")

    # Выравниваем длину массивов
    min_len_train = min(len(tft_probs_train), len(lgbm_probs_train), len(y_train))
    min_len_val = min(len(tft_probs_val), len(lgbm_probs_val), len(y_val))

    tft_probs_train = tft_probs_train[-min_len_train:]
    lgbm_probs_train = lgbm_probs_train[-min_len_train:]
    y_train = y_train[-min_len_train:]

    tft_probs_val = tft_probs_val[-min_len_val:]
    lgbm_probs_val = lgbm_probs_val[-min_len_val:]
    y_val = y_val[-min_len_val:]

    # Создаём мета-фичи
    X_train_meta = np.column_stack([tft_probs_train, lgbm_probs_train])
    X_val_meta = np.column_stack([tft_probs_val, lgbm_probs_val])

    # Обучаем логистическую регрессию
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_meta, y_train)
    
    # Оценка
    pred = model.predict(X_val_meta)
    acc = accuracy_score(y_val, pred)
    print(f"Stacking Val Acc: {acc:.3f}\n")

    os.makedirs('data', exist_ok=True)
    joblib.dump(model, STACKING_MODEL_PATH)
    return model


# ============================================================================
# РЕГРЕССИЯ ДЛЯ % ИЗМЕНЕНИЯ
# ============================================================================

def train_regression(X_train, y_train, X_val, y_val, feature_cols):
    """
    Ridge регрессия для предсказания % изменения цены
    + SHAP анализ для feature importance
    """
    print("\nОбучаем регрессию...")
    
    # Убираем NaN
    mask_train = ~np.isnan(y_train)
    mask_val = ~np.isnan(y_val)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    X_val, y_val = X_val[mask_val], y_val[mask_val]
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("⚠️ Недостаточно данных для регрессии")
        return None
    
    # Обучение
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # Оценка
    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    print(f"Regression MAE: {mae:.3f}%\n")
    
    # Сохранение
    joblib.dump(model, REGRESSION_MODEL_PATH)
    
    # === SHAP АНАЛИЗ ===
    try:
        print("Вычисляем SHAP values...")
        # Для скорости используем подвыборку
        sample_size = min(1000, len(X_val))
        sample_indices = np.random.choice(len(X_val), sample_size, replace=False)
        X_val_sample = X_val[sample_indices]
        
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_val_sample)
        
        # Сохранение SHAP values для feature selection
        joblib.dump(shap_values, SHAP_VALUES_PATH)
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val_sample, feature_names=feature_cols, show=False)
        plt.savefig('data/shap_summary.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("✅ SHAP summary сохранён в data/shap_summary.png")
        
    except Exception as e:
        print(f"⚠️ SHAP ошибка: {e}")
    
    return model