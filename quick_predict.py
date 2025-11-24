# quick_predict.py - БЫСТРЫЙ ПРОГНОЗ С ОТСЛЕЖИВАНИЕМ
import torch
from data_loader import load_and_update_data
from features import (
    add_technical_indicators, filter_anomalies, add_onchain_features,
    add_macro_features, add_multiscale_features, add_fear_greed_index, 
    add_btc_dominance, add_google_trends, add_additional_macro, 
    add_correlations_and_external, add_temporal_features, add_fed_rate
)
from predict import predict_ensemble
from prediction_tracker import PredictionTracker
import pandas as pd

print("="*80)
print("🔮 БЫСТРЫЙ ПРОГНОЗ BTC с отслеживанием результатов")
print("="*80 + "\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Устройство: {device}\n")

# Инициализация tracker
tracker = PredictionTracker()

# === ШАГ 1: Проверка старых прогнозов ===
print("="*80)
print("ШАГ 1: Проверка предыдущих прогнозов")
print("="*80)

# Загружаем текущую цену для проверки
df_check = load_and_update_data()
current_price_for_check = df_check['close'].iloc[-1]

tracker.check_predictions(current_price_for_check)

# === ШАГ 2: Генерация нового прогноза ===
print("\n" + "="*80)
print("ШАГ 2: Генерация нового прогноза")
print("="*80 + "\n")

print("Загрузка данных...")
df = load_and_update_data()

print("Добавление фичей...")
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

# Прогноз с получением детальной информации
print("\nГенерация прогноза...\n")

# Сохраняем компоненты прогноза для tracker
import joblib
import numpy as np
import os
from config import *

# Получаем детальные прогнозы каждой модели
try:
    scaler = joblib.load(SCALER_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    
    if os.path.exists(SELECTED_FEATURE_COLS_PATH):
        feature_cols = joblib.load(SELECTED_FEATURE_COLS_PATH)
    else:
        feature_cols = joblib.load(FEATURE_COLS_PATH)
    
    # Подготовка данных
    window = df.iloc[-LOOKBACK:].copy()
    latest = df.iloc[-1:].copy()
    
    # Очистка inf/nan
    for col in feature_cols:
        if col in window.columns:
            window[col] = window[col].replace([np.inf, -np.inf], np.nan)
        if col in latest.columns:
            latest[col] = latest[col].replace([np.inf, -np.inf], np.nan)
    
    window_imp = imputer.transform(window[feature_cols])
    latest_imp = imputer.transform(latest[feature_cols])
    window_scaled = scaler.transform(window_imp)
    latest_scaled = scaler.transform(latest_imp)
    window_scaled = np.nan_to_num(window_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    latest_scaled = np.nan_to_num(latest_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # TFT
    tft_prob = 0.5
    try:
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        if os.path.exists(TFT_CHECKPOINT_PATH) and os.path.exists(TFT_TRAINING_PATH):
            training = joblib.load(TFT_TRAINING_PATH)
            tft = TemporalFusionTransformer.load_from_checkpoint(TFT_CHECKPOINT_PATH, map_location=device)
            tft.eval()
            tft.to(device)
            
            encoder_data = pd.DataFrame(window_scaled, columns=feature_cols)
            encoder_data["target"] = 0.0
            encoder_data["time_idx"] = np.arange(LOOKBACK)
            encoder_data["group"] = 0
            decoder_data = encoder_data.iloc[[-1]].copy()
            decoder_data["time_idx"] = LOOKBACK
            full_pred_df = pd.concat([encoder_data, decoder_data], ignore_index=True)
            
            pred_dataset = TimeSeriesDataSet.from_dataset(training, full_pred_df, predict=True, stop_randomization=True)
            pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)
            
            with torch.no_grad():
                raw_preds = tft.predict(pred_loader, mode="quantiles")
                if isinstance(raw_preds, torch.Tensor):
                    raw_preds = raw_preds.cpu().numpy()
                median_pred = raw_preds[0, 0, 3] if raw_preds.ndim == 3 else raw_preds[0]
                tft_prob = 1 / (1 + np.exp(-median_pred))
                tft_prob = np.clip(tft_prob, 0.01, 0.99)
    except:
        pass
    
    # LGBM
    lgbm_prob = 0.5
    try:
        lgbm = joblib.load(LGBM_MODEL_PATH)
        prob = lgbm.predict(latest_scaled)
        lgbm_prob = prob[0] if prob.ndim == 1 else prob[0, 1]
        lgbm_prob = np.clip(lgbm_prob, 0.01, 0.99)
        
        if os.path.exists('data/isotonic_calibrator.pkl'):
            calibrator = joblib.load('data/isotonic_calibrator.pkl')
            lgbm_prob = calibrator.transform([lgbm_prob])[0]
    except:
        pass
    
    # Regression
    regression_pct = 0.0
    try:
        reg = joblib.load(REGRESSION_MODEL_PATH)
        regression_pct = reg.predict(latest_scaled)[0]
    except:
        regression_pct = (lgbm_prob - 0.5) * 8
    
    # Применяем адаптивные веса из tracker
    weights = tracker.get_current_weights()
    final_prob = tft_prob * weights['tft_weight'] + lgbm_prob * weights['lgbm_weight']
    
    # Stacking (если есть)
    try:
        stack = joblib.load(STACKING_MODEL_PATH)
        final_prob = stack.predict_proba([[tft_prob, lgbm_prob]])[0, 1]
    except:
        pass
    
    # Финальное направление
    direction = "ЛОНГ ⬆" if regression_pct > 0 else "ШОРТ ⬇"
    confidence = final_prob * 100 if regression_pct > 0 else (1 - final_prob) * 100
    confidence = np.clip(confidence, 0, 100)
    
    # Текущая цена
    current_price = df['close'].iloc[-1]
    target_price = current_price * (1 + regression_pct / 100)
    
    # === СОХРАНЕНИЕ ПРОГНОЗА В TRACKER ===
    prediction_id = tracker.save_prediction(
        current_price=current_price,
        tft_prob=tft_prob,
        lgbm_prob=lgbm_prob,
        regression_pct=regression_pct,
        final_direction=direction,
        final_confidence=confidence,
        final_pct=regression_pct
    )
    
    # Вывод
    print("\n" + "="*80)
    print("🎯 ПРОГНОЗ НА СЛЕДУЮЩИЕ 6 ЧАСОВ:")
    print("="*80)
    print(f"\n📊 Прогнозы моделей:")
    print(f"   TFT prob: {tft_prob:.4f} ({'ЛОНГ' if tft_prob > 0.5 else 'ШОРТ'})")
    print(f"   LGBM prob: {lgbm_prob:.4f} ({'ЛОНГ' if lgbm_prob > 0.5 else 'ШОРТ'})")
    print(f"   Regression: {regression_pct:+.2f}%")
    
    print(f"\n🎲 Адаптивные веса:")
    print(f"   TFT: {weights['tft_weight']:.3f}")
    print(f"   LGBM: {weights['lgbm_weight']:.3f}")
    
    print(f"\n🎯 ФИНАЛЬНЫЙ ПРОГНОЗ:")
    print(f"   Направление: {direction}")
    print(f"   Изменение: {regression_pct:+.2f}%")
    print(f"   Уверенность: {confidence:.1f}%")
    
    print(f"\n💰 Цены:")
    print(f"   Текущая: ${current_price:,.2f}")
    print(f"   Целевая (6h): ${target_price:,.2f}")
    
    print("\n" + "="*80)
    print(f"💾 Прогноз сохранён с ID: {prediction_id}")
    print("   Запусти этот скрипт через 6 часов для автопроверки!")
    print("="*80)
    
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()