# main.py
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib
import warnings
import optuna
import lightgbm as lgb

# === Унифицированный импорт Lightning ===
try:
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer as _Trainer
    print("Используется lightning.pytorch")
except Exception:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer as _Trainer
    print("Используется pytorch_lightning")

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

from data_loader import load_and_update_data
from features import (
    add_technical_indicators, filter_anomalies, add_onchain_features,
    add_macro_features, create_dual_target, create_regression_target,
    add_multiscale_features, add_fear_greed_index, add_btc_dominance,
    add_google_trends, add_additional_macro, add_correlations_and_external,
    add_temporal_features, add_fed_rate  # НОВОЕ!
)
from model import train_lgbm, train_stacking, train_regression, calibrate_lgbm_probs
from predict import predict_ensemble
from config import *

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pl.seed_everything(42)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_and_normalize(train_df, val_df, feature_cols):
    """Очистка, импутация и нормализация данных"""
    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()

    # Убираем дубликаты колонок
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_val = X_val.loc[:, ~X_val.columns.duplicated()]
    feature_cols = X_train.columns.tolist()

    # Замена inf на NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Импутация
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    # Финальная очистка NaN
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)

    return X_train_scaled, X_val_scaled, scaler, imputer, feature_cols


def tune_tft(training, train_dataloader, val_dataloader):
    """Hyperparameter tuning для TFT через Optuna"""
    def objective(trial):
        config = {
            "learning_rate": trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
            "hidden_size": trial.suggest_int('hidden_size', 16, 128),
            "attention_head_size": trial.suggest_int('attention_head_size', 1, 4),
            "dropout": trial.suggest_float('dropout', 0.1, 0.3),
            "hidden_continuous_size": trial.suggest_int('hidden_continuous_size', 8, 64),
        }
        tft = TemporalFusionTransformer.from_dataset(
            training, **config, output_size=7, loss=QuantileLoss()
        )
        trainer = _Trainer(
            max_epochs=3,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )
        trainer.fit(tft, train_dataloader, val_dataloader)
        val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5)
    return study.best_params


def tune_lgbm(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning для LightGBM через Optuna"""
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'verbose': -1,
            'seed': 42,
            'device': 'cpu'
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(
            params, lgb_train, num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(50)]
        )
        pred = (model.predict(X_val) > 0.5).astype(int)
        return 1 - accuracy_score(y_val, pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    return study.best_params


def backtest_model(df, model, feature_cols, lookback=168, future=6):
    """Простой бэктест модели"""
    preds = []
    for i in range(lookback, len(df) - future, 24):
        window = df.iloc[i-lookback:i][feature_cols]
        pred = model.predict(window.mean().values.reshape(1, -1))[0] > 0.5
        actual = df['close'].iloc[i + future] > df['close'].iloc[i]
        preds.append((pred, actual))
    
    if len(preds) == 0:
        return 0.5
    
    acc = np.mean([p == a for p, a in preds])
    print(f"Backtest accuracy: {acc:.3f}")
    return acc


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("="*80)
    print("🚀 BTC Анализатор v2.3 - УЛУЧШЕННАЯ ВЕРСИЯ")
    print("   ✅ Исправлены баги (duplicate labels, calibration)")
    print("   ✅ Добавлены продвинутые временные фичи (лаги, rolling, momentum)")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}\n")

    # ========================================================================
    # 1. ЗАГРУЗКА И ОБОГАЩЕНИЕ ДАННЫХ
    # ========================================================================
    print("=" * 60)
    print("ЭТАП 1: Загрузка данных")
    print("=" * 60)
    
    df = load_and_update_data()
    print(f"Загружено: {len(df)} часовых свечей\n")
    
    # Базовые индикаторы
    print("=" * 60)
    print("ЭТАП 2: Базовые технические индикаторы")
    print("=" * 60)
    df = add_technical_indicators(df)
    
    # Мультимасштабные фичи (4h, 1d)
    print("=" * 60)
    print("ЭТАП 3: Мультимасштабные фичи")
    print("=" * 60)
    df = add_multiscale_features(df)
    
    # === НОВОЕ: ПРОДВИНУТЫЕ ВРЕМЕННЫЕ ФИЧИ ===
    print("=" * 60)
    print("ЭТАП 4: 🌟 ПРОДВИНУТЫЕ ВРЕМЕННЫЕ ФИЧИ (НОВОЕ!)")
    print("=" * 60)
    df = add_temporal_features(df)
    
    # Фильтрация аномалий
    print("=" * 60)
    print("ЭТАП 5: Фильтрация аномалий")
    print("=" * 60)
    df = filter_anomalies(df)
    
    # On-chain и макро
    print("=" * 60)
    print("ЭТАП 6: On-chain и макро-данные")
    print("=" * 60)
    df = add_onchain_features(df)
    df = add_macro_features(df)
    df = add_fear_greed_index(df)
    df = add_btc_dominance(df)
    df = add_google_trends(df)
    df = add_fed_rate(df)
    df = add_additional_macro(df)
    df = add_correlations_and_external(df)

    # ========================================================================
    # 2. СОЗДАНИЕ ТАРГЕТОВ
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 7: Создание таргетов (6h short, 24h long)")
    print("=" * 60)
    
    df = create_dual_target(df, short=FUTURE_TARGET_SHORT, long=FUTURE_TARGET_LONG)
    df = df[(df['target_short'] != -1) & (df['target_long'] != -1)].copy()
    print(f"Данных после удаления неполных таргетов: {len(df)}\n")

    # ========================================================================
    # 3. TRAIN/VAL SPLIT
    # ========================================================================
    print("=" * 60)
    print("ЭТАП 8: Train/Val split (80/20)")
    print("=" * 60)
    
    train_end_idx = int(0.8 * len(df))
    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:].copy()
    print(f"Train: {len(train_df)} | Val: {len(val_df)}\n")

    # ========================================================================
    # 4. FEATURE SELECTION
    # ========================================================================
    print("=" * 60)
    print("ЭТАП 9: Подготовка фичей")
    print("=" * 60)
    
    feature_cols = [c for c in df.columns if c not in ['target_short', 'target_long', 'pct_change']]
    print(f"Всего фичей: {len(feature_cols)}")
    
    # Нормализация
    X_train_scaled, X_val_scaled, scaler, imputer, feature_cols = clean_and_normalize(
        train_df, val_df, feature_cols
    )

    # Сохранение
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)

    # Таргеты
    y_train_short = train_df['target_short'].values
    y_val_short = val_df['target_short'].values

    # ========================================================================
    # 5. РЕГРЕССИЯ ДЛЯ SHAP (feature importance)
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 10: Регрессия для SHAP анализа")
    print("=" * 60)
    
    df_reg = create_regression_target(df.copy(), future=FUTURE_TARGET_SHORT)
    df_reg = df_reg[df_reg['pct_change'].notna()].copy()

    if len(df_reg) > 100:
        X_reg_train = X_train_scaled[:len(df_reg) - len(val_df)]
        y_reg_train = df_reg['pct_change'].iloc[:len(X_reg_train)].values
        X_reg_val = X_val_scaled[:len(df_reg) - len(X_reg_train)]
        y_reg_val = df_reg['pct_change'].iloc[len(X_reg_train):len(X_reg_train) + len(X_reg_val)].values

        if len(y_reg_train) > 0 and len(y_reg_val) > 0:
            train_regression(X_reg_train, y_reg_train, X_reg_val, y_reg_val, feature_cols)

    # Feature selection через SHAP
    selected_features = feature_cols
    if os.path.exists(SHAP_VALUES_PATH):
        shap_values = joblib.load(SHAP_VALUES_PATH)
        shap_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': shap_importance})
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Убираем bottom 20%
        threshold = importance_df['importance'].quantile(0.2)
        selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
        print(f"\n✅ Feature selection: отобрано {len(selected_features)} из {len(feature_cols)} фичей")
        
        joblib.dump(selected_features, SELECTED_FEATURE_COLS_PATH)

        # Re-normalize на selected features
        train_df_selected = train_df[selected_features + ['target_short']]
        val_df_selected = val_df[selected_features + ['target_short']]

        X_train_scaled, X_val_scaled, scaler, imputer, selected_features = clean_and_normalize(
            train_df_selected, val_df_selected, selected_features
        )

        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(imputer, IMPUTER_PATH)

        y_train_short = train_df_selected['target_short'].values
        y_val_short = val_df_selected['target_short'].values

    # ========================================================================
    # 6. HYPERPARAMETER TUNING
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 11: Hyperparameter Tuning (Optuna)")
    print("=" * 60)
    
    # TFT Tuning
    print("\nTuning TFT...")
    train_raw = train_df[selected_features].copy() if 'selected_features' in locals() else train_df[feature_cols].copy()
    val_raw = val_df[selected_features].copy() if 'selected_features' in locals() else val_df[feature_cols].copy()

    # === КРИТИЧНО: Очистка inf/nan для TFT ===
    print("  Очистка данных от inf/nan...")
    train_raw = train_raw.replace([np.inf, -np.inf], np.nan)
    val_raw = val_raw.replace([np.inf, -np.inf], np.nan)
    
    # Заполняем NaN медианой
    for col in train_raw.columns:
        if train_raw[col].isna().sum() > 0:
            median_val = train_raw[col].median()
            train_raw[col] = train_raw[col].fillna(median_val)
            val_raw[col] = val_raw[col].fillna(median_val)
    
    print(f"  ✅ Данные очищены: train={len(train_raw)}, val={len(val_raw)}")

    train_raw['target'] = y_train_short.astype(float)
    val_raw['target'] = y_val_short.astype(float)
    train_raw['time_idx'] = np.arange(len(train_raw))
    val_raw['time_idx'] = np.arange(len(train_raw), len(train_raw) + len(val_raw))
    train_raw['group'] = 0
    val_raw['group'] = 0

    full_df_tft = pd.concat([train_raw, val_raw], ignore_index=True)

    training = TimeSeriesDataSet(
        full_df_tft[lambda x: x.time_idx < len(train_raw)],
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        min_encoder_length=LOOKBACK//2,
        max_encoder_length=LOOKBACK,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=selected_features if 'selected_features' in locals() else feature_cols,
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, full_df_tft, predict=False, stop_randomization=True)

    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    best_tft_params = tune_tft(training, train_dataloader, val_dataloader)
    print(f"✅ Best TFT params: {best_tft_params}")

    # LGBM Tuning
    print("\nTuning LGBM...")
    best_lgbm_params = tune_lgbm(X_train_scaled, y_train_short, X_val_scaled, y_val_short)
    print(f"✅ Best LGBM params: {best_lgbm_params}")

    # ========================================================================
    # 7. ОБУЧЕНИЕ МОДЕЛЕЙ
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 12: Обучение моделей с оптимальными параметрами")
    print("=" * 60)
    
    # TFT
    print("\nОбучение TFT...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        **best_tft_params,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=3,
    )

    trainer = _Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(TFT_CHECKPOINT_PATH)
    joblib.dump(training, TFT_TRAINING_PATH)
    print("✅ TFT модель обучена и сохранена!")

    # TFT Predictions
    print("\nГенерация TFT прогнозов...")
    tft.eval()
    with torch.no_grad():
        preds = tft.predict(val_dataloader, mode="quantiles")
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        tft_probs_val = preds[:, 0, 3]
        tft_probs_val = 1 / (1 + np.exp(-tft_probs_val))
        tft_probs_val = np.clip(tft_probs_val, 0.01, 0.99)

    # LGBM
    print("\nОбучение LightGBM...")
    best_lgbm_params['objective'] = 'binary'
    best_lgbm_params['metric'] = 'binary_logloss'
    best_lgbm_params['verbose'] = -1
    best_lgbm_params['seed'] = 42
    best_lgbm_params['device'] = 'cpu'
    
    lgbm_model = train_lgbm(X_train_scaled, y_train_short, X_val_scaled, y_val_short, params=best_lgbm_params)
    lgbm_probs_val = calibrate_lgbm_probs(lgbm_model, X_train_scaled, y_train_short, X_val_scaled)

    # Stacking
    print("\nОбучение Stacking...")
    train_stacking(
        tft_probs_val[:len(y_train_short)],
        lgbm_model.predict(X_train_scaled),
        y_train_short,
        tft_probs_val,
        lgbm_probs_val,
        y_val_short
    )

    # Regression (refit)
    print("\nRefit регрессии на selected features...")
    df_reg = create_regression_target(df.copy(), future=FUTURE_TARGET_SHORT)
    df_reg = df_reg[df_reg['pct_change'].notna()].copy()

    if len(df_reg) > 100:
        X_reg_train = X_train_scaled[:len(df_reg) - len(val_df)]
        y_reg_train = df_reg['pct_change'].iloc[:len(X_reg_train)].values
        X_reg_val = X_val_scaled[:len(df_reg) - len(X_reg_train)]
        y_reg_val = df_reg['pct_change'].iloc[len(X_reg_train):len(X_reg_train) + len(X_reg_val)].values

        if len(y_reg_train) > 0 and len(y_reg_val) > 0:
            train_regression(X_reg_train, y_reg_train, X_reg_val, y_reg_val, 
                           selected_features if 'selected_features' in locals() else feature_cols)

    # ========================================================================
    # 8. BACKTESTING
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 13: Backtesting")
    print("=" * 60)
    
    backtest_acc = backtest_model(
        df, lgbm_model,
        selected_features if 'selected_features' in locals() else feature_cols
    )

    # ========================================================================
    # 9. ФИНАЛЬНЫЙ ПРОГНОЗ
    # ========================================================================
    print("\n" + "=" * 60)
    print("ЭТАП 14: Финальный прогноз")
    print("=" * 60)
    
    df_pred = df.drop(columns=['target_short', 'target_long', 'pct_change'], errors='ignore')
    direction, confidence, pct_change, strength, prob_long = predict_ensemble(df_pred, device)

    # Вывод результатов
    print("\n" + "="*80)
    print("🎯 ПРОГНОЗ НА СЛЕДУЮЩИЕ 6 ЧАСОВ:")
    print("="*80)
    print(f"   Направление: {direction}")
    print(f"   Изменение: {pct_change:+.2f}%")
    print(f"   Сила движения: {strength}")
    print(f"   Уверенность: {confidence:.1f}%")
    print(f"\n📊 24H ТРЕНД: {'ВВЕРХ ⬆' if prob_long > 50 else 'ВНИЗ ⬇'} ({prob_long:.1f}%)")
    print("="*80 + "\n")

    # ========================================================================
    # 10. СОХРАНЕНИЕ ИСТОРИИ
    # ========================================================================
    history_path = 'data/forecast_history.csv'
    entry = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        'direction': direction,
        'pct_change': f"{pct_change:+.2f}%",
        'strength': strength,
        'confidence': f"{confidence:.1f}%",
        'backtest_acc': f"{backtest_acc:.3f}"
    }
    hist = pd.DataFrame([entry])
    if os.path.exists(history_path):
        hist = pd.concat([pd.read_csv(history_path), hist], ignore_index=True)
    hist.tail(50).to_csv(history_path, index=False)
    print(f"✅ Прогноз сохранён в {history_path}\n")

    print("="*80)
    print("🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
    print("="*80)


if __name__ == "__main__":
    main()