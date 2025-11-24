# analyze_predictions.py - АНАЛИЗ ПРЕДСКАЗАНИЙ МОДЕЛЕЙ
import torch
import joblib
import numpy as np
import pandas as pd
from data_loader import load_and_update_data
from features import (
    add_technical_indicators, filter_anomalies, add_onchain_features,
    add_macro_features, add_multiscale_features, add_fear_greed_index, 
    add_btc_dominance, add_google_trends, add_additional_macro, 
    add_correlations_and_external, add_temporal_features, add_fed_rate,
    create_dual_target
)
from config import *

print("="*80)
print("🔍 АНАЛИЗ ПРЕДСКАЗАНИЙ МОДЕЛЕЙ")
print("="*80 + "\n")

# Загрузка данных
df = load_and_update_data()
df = add_technical_indicators(df)
df = add_multiscale_features(df)
df = add_temporal_features(df)
df = filter_anomalies(df)
df = add_onchain_features(df)
df = add_macro_features(df)
df = add_fear_greed_index(df)
df = add_btc_dominance(df)
df = add_google_trends(df)
df = add_fed_rate(df)
df = add_additional_macro(df)
df = add_correlations_and_external(df)

# Таргеты
df = create_dual_target(df, short=FUTURE_TARGET_SHORT, long=FUTURE_TARGET_LONG)
df = df[(df['target_short'] != -1) & (df['target_long'] != -1)].copy()

# Split
train_end_idx = int(0.8 * len(df))
val_df = df.iloc[train_end_idx:].copy()

print(f"Validation данных: {len(val_df)} строк")

# Загрузка моделей
try:
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load('data/imputer.pkl')
    feature_cols = joblib.load(SELECTED_FEATURE_COLS_PATH)
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    
    print(f"Модели загружены. Фичей: {len(feature_cols)}\n")
    
    # Подготовка validation данных
    X_val = val_df[feature_cols].copy()
    y_val = val_df['target_short'].values
    
    # Очистка
    for col in feature_cols:
        X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)
    
    X_val_imp = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imp)
    X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
    
    # LGBM прогнозы
    lgbm_probs = lgbm_model.predict(X_val_scaled)
    lgbm_preds = (lgbm_probs > 0.5).astype(int)
    
    # Калибровка
    if os.path.exists('data/isotonic_calibrator.pkl'):
        calibrator = joblib.load('data/isotonic_calibrator.pkl')
        lgbm_probs_calibrated = calibrator.transform(lgbm_probs)
    else:
        lgbm_probs_calibrated = lgbm_probs
    
    # Анализ
    print("="*80)
    print("СТАТИСТИКА LGBM ПРОГНОЗОВ НА VALIDATION:")
    print("="*80)
    print(f"\nБез калибровки:")
    print(f"  Min prob: {lgbm_probs.min():.4f}")
    print(f"  Max prob: {lgbm_probs.max():.4f}")
    print(f"  Mean prob: {lgbm_probs.mean():.4f}")
    print(f"  Median prob: {np.median(lgbm_probs):.4f}")
    print(f"  Std prob: {lgbm_probs.std():.4f}")
    
    print(f"\nС калибровкой:")
    print(f"  Min prob: {lgbm_probs_calibrated.min():.4f}")
    print(f"  Max prob: {lgbm_probs_calibrated.max():.4f}")
    print(f"  Mean prob: {lgbm_probs_calibrated.mean():.4f}")
    print(f"  Median prob: {np.median(lgbm_probs_calibrated):.4f}")
    print(f"  Std prob: {lgbm_probs_calibrated.std():.4f}")
    
    # Accuracy
    from sklearn.metrics import accuracy_score
    acc_uncalib = accuracy_score(y_val, lgbm_preds)
    calibrated_preds = (lgbm_probs_calibrated > 0.5).astype(int)
    acc_calib = accuracy_score(y_val, calibrated_preds)
    
    print(f"\nAccuracy:")
    print(f"  Без калибровки: {acc_uncalib:.3f}")
    print(f"  С калибровкой: {acc_calib:.3f}")
    
    # Распределение
    print(f"\nРаспределение прогнозов (без калибровки):")
    print(f"  ЛОНГ (prob > 0.5): {(lgbm_probs > 0.5).sum()} ({(lgbm_probs > 0.5).sum()/len(lgbm_probs)*100:.1f}%)")
    print(f"  ШОРТ (prob < 0.5): {(lgbm_probs < 0.5).sum()} ({(lgbm_probs < 0.5).sum()/len(lgbm_probs)*100:.1f}%)")
    
    print(f"\nРаспределение прогнозов (с калибровкой):")
    print(f"  ЛОНГ (prob > 0.5): {(lgbm_probs_calibrated > 0.5).sum()} ({(lgbm_probs_calibrated > 0.5).sum()/len(lgbm_probs_calibrated)*100:.1f}%)")
    print(f"  ШОРТ (prob < 0.5): {(lgbm_probs_calibrated < 0.5).sum()} ({(lgbm_probs_calibrated < 0.5).sum()/len(lgbm_probs_calibrated)*100:.1f}%)")
    
    # Последний прогноз (текущий)
    print("\n" + "="*80)
    print("ТЕКУЩИЙ ПРОГНОЗ (последняя строка val):")
    print("="*80)
    print(f"LGBM prob (без калибровки): {lgbm_probs[-1]:.4f}")
    print(f"LGBM prob (с калибровкой): {lgbm_probs_calibrated[-1]:.4f}")
    print(f"Прогноз: {'ЛОНГ' if lgbm_probs_calibrated[-1] > 0.5 else 'ШОРТ'}")
    print(f"Actual target (если известен): {y_val[-1]}")
    
    print("\n" + "="*80)
    print("ВЫВОД:")
    print("="*80)
    if lgbm_probs_calibrated.mean() < 0.3:
        print("⚠️ ПРОБЛЕМА: Калибровка слишком агрессивна!")
        print("   Средняя вероятность < 0.3 означает сильный bias к ШОРТУ")
        print("   Рекомендация: Отключить калибровку или переобучить на balanced данных")
    elif lgbm_probs_calibrated.mean() > 0.7:
        print("⚠️ ПРОБЛЕМА: Калибровка слишком агрессивна в сторону ЛОНГА!")
    else:
        print("✅ Калибровка выглядит нормально (mean ~0.5)")
        
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()

print("="*80)