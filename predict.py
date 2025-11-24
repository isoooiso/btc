import os
import torch
import joblib
import numpy as np
import pandas as pd
from config import *

def predict_ensemble(df, device):
    """
    Ансамблевый прогноз направления и силы движения BTC
    TFT + LGBM + Stacking + Regression
    """
    try:
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        
        # Загрузка feature columns (prioritize selected)
        if os.path.exists(SELECTED_FEATURE_COLS_PATH):
            feature_cols = joblib.load(SELECTED_FEATURE_COLS_PATH)
            print(f"Используем {len(feature_cols)} отобранных фичей")
        else:
            feature_cols = joblib.load(FEATURE_COLS_PATH)
            print(f"Используем все {len(feature_cols)} фичей")
            
    except Exception as e:
        print(f"Ошибка загрузки scaler/imputer/feature_cols: {e}")
        return "МОДЕЛИ НЕ НАЙДЕНЫ", 50.0, 0.0, "слабый", 50.0

    df = df.copy()
    
    # Проверка минимальной длины
    if len(df) < LOOKBACK:
        print(f"⚠️ Недостаточно данных: {len(df)} < {LOOKBACK}")
        return "НЕДОСТАТОЧНО ДАННЫХ", 50.0, 0.0, "слабый", 50.0

    # Берём последние LOOKBACK строк для encoder + 1 для decoder
    window = df.iloc[-LOOKBACK:].copy()
    latest = df.iloc[-1:].copy()

    # Проверка наличия всех фичей
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        print(f"⚠️ Отсутствующие фичи: {missing_features}")
        # Добавляем отсутствующие фичи с нулями
        for feat in missing_features:
            window[feat] = 0.0
            latest[feat] = 0.0

    # === КРИТИЧНО: АГРЕССИВНАЯ очистка inf/nan! ===
    print("  Очистка inf/nan в window и latest...")
    
    # Для window
    for col in feature_cols:
        if col in window.columns:
            window[col] = window[col].replace([np.inf, -np.inf], np.nan)
    
    # Для latest
    for col in feature_cols:
        if col in latest.columns:
            latest[col] = latest[col].replace([np.inf, -np.inf], np.nan)
    
    print(f"  После замены inf: window NaN={window[feature_cols].isna().sum().sum()}, latest NaN={latest[feature_cols].isna().sum().sum()}")

    # Импутация и масштабирование
    try:
        window_imp = imputer.transform(window[feature_cols])
        latest_imp = imputer.transform(latest[feature_cols])
        
        # Проверка после импутации
        if np.isinf(window_imp).any() or np.isinf(latest_imp).any():
            print("  ⚠️ Обнаружены inf после импутации, применяем nan_to_num...")
            window_imp = np.nan_to_num(window_imp, nan=0.0, posinf=0.0, neginf=0.0)
            latest_imp = np.nan_to_num(latest_imp, nan=0.0, posinf=0.0, neginf=0.0)
        
        window_scaled = scaler.transform(window_imp)
        latest_scaled = scaler.transform(latest_imp)
        
        # Финальная проверка
        window_scaled = np.nan_to_num(window_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        latest_scaled = np.nan_to_num(latest_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"  ✅ Предобработка завершена: window_scaled shape={window_scaled.shape}, latest_scaled shape={latest_scaled.shape}")
        
    except Exception as e:
        print(f"❌ Ошибка предобработки: {e}")
        import traceback
        traceback.print_exc()
        return "ОШИБКА ПРЕДОБРАБОТКИ", 50.0, 0.0, "слабый", 50.0

    # ========================================================================
    # 1. TEMPORAL FUSION TRANSFORMER
    # ========================================================================
    tft_prob = 0.5
    try:
        tft_ckpt = TFT_CHECKPOINT_PATH
        training_path = TFT_TRAINING_PATH

        if os.path.exists(tft_ckpt) and os.path.exists(training_path):
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

            training = joblib.load(training_path)
            tft = TemporalFusionTransformer.load_from_checkpoint(tft_ckpt, map_location=device)
            tft.eval()
            tft.to(device)

            # Подготовка данных для TFT
            encoder_data = pd.DataFrame(window_scaled, columns=feature_cols)
            encoder_data["target"] = 0.0  # placeholder
            encoder_data["time_idx"] = np.arange(LOOKBACK)
            encoder_data["group"] = 0

            # Decoder (1 шаг вперёд)
            decoder_data = encoder_data.iloc[[-1]].copy()
            decoder_data["time_idx"] = LOOKBACK

            # Объединяем
            full_pred_df = pd.concat([encoder_data, decoder_data], ignore_index=True)

            # Создаём датасет для inference
            pred_dataset = TimeSeriesDataSet.from_dataset(
                training,
                full_pred_df,
                predict=True,
                stop_randomization=True
            )

            pred_loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            # Прогноз
            with torch.no_grad():
                raw_preds = tft.predict(pred_loader, mode="quantiles")
                
                # Обработка формата выхода
                if isinstance(raw_preds, torch.Tensor):
                    raw_preds = raw_preds.cpu().numpy()
                
                # Берём медианный квантиль (index 3 из 7)
                if raw_preds.ndim == 3:
                    median_pred = raw_preds[0, 0, 3]
                else:
                    median_pred = raw_preds[0]
                
                # Преобразуем в вероятность через sigmoid
                tft_prob = 1 / (1 + np.exp(-median_pred))
                tft_prob = np.clip(tft_prob, 0.01, 0.99)  # Clipping для стабильности
                
            print(f"TFT OK → raw={median_pred:.4f}, prob={tft_prob:.4f}")
        else:
            print("TFT файлы не найдены, используем fallback")
            
    except Exception as e:
        print(f"TFT ошибка: {e}")
        import traceback
        traceback.print_exc()
        tft_prob = 0.5

    # ========================================================================
    # 2. LIGHTGBM
    # ========================================================================
    lgbm_prob = 0.5
    try:
        lgbm = joblib.load(LGBM_MODEL_PATH)
        prob = lgbm.predict(latest_scaled)
        
        # Обработка формата выхода
        if isinstance(prob, np.ndarray):
            lgbm_prob = prob[0] if prob.ndim == 1 else prob[0, 1]
        else:
            lgbm_prob = float(prob)
        
        lgbm_prob = np.clip(lgbm_prob, 0.01, 0.99)
        
        # Применяем калибровку если доступна
        if os.path.exists('data/isotonic_calibrator.pkl'):
            calibrator = joblib.load('data/isotonic_calibrator.pkl')
            lgbm_prob = calibrator.transform([lgbm_prob])[0]
            print(f"LGBM OK (calibrated) → prob={lgbm_prob:.4f}")
        else:
            print(f"LGBM OK → prob={lgbm_prob:.4f}")
            
    except Exception as e:
        print(f"LGBM ошибка: {e}")
        lgbm_prob = 0.5

    # ========================================================================
    # 3. STACKING META-MODEL
    # ========================================================================
    final_prob = tft_prob * 0.5 + lgbm_prob * 0.5  # дефолтное взвешивание
    
    try:
        if os.path.exists(STACKING_MODEL_PATH):
            stack = joblib.load(STACKING_MODEL_PATH)
            final_prob = stack.predict_proba([[tft_prob, lgbm_prob]])[0, 1]
            print(f"Stacking OK → final_prob={final_prob:.4f}")
        else:
            print("Stacking модель не найдена, используем среднее")
    except Exception as e:
        print(f"Stacking ошибка: {e}")

    # ========================================================================
    # 4. РЕГРЕССИЯ (% изменения)
    # ========================================================================
    pct = 0.0
    try:
        if os.path.exists(REGRESSION_MODEL_PATH):
            reg = joblib.load(REGRESSION_MODEL_PATH)
            pct = reg.predict(latest_scaled)[0]
            print(f"Regression OK → pct_change={pct:.2f}%")
        else:
            # Fallback: оцениваем через вероятность
            pct = (final_prob - 0.5) * 8  # -4% до +4%
            print(f"Regression не найдена, используем fallback → {pct:.2f}%")
    except Exception as e:
        print(f"Regression ошибка: {e}")
        pct = (final_prob - 0.5) * 8

    # ========================================================================
    # 5. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
    # ========================================================================
    
    # Направление
    direction = "ЛОНГ ⬆" if pct > 0 else "ШОРТ ⬇"
    
    # Сила движения
    abs_pct = abs(pct)
    if abs_pct > 5:
        strength = "СИЛЬНЫЙ 🔥"
    elif abs_pct > 2:
        strength = "средний 📊"
    else:
        strength = "слабый 📉"
    
    # Уверенность (confidence)
    confidence = final_prob * 100 if pct > 0 else (1 - final_prob) * 100
    confidence = np.clip(confidence, 0, 100)
    
    # Общая оценка (для 24h тренда)
    trend_24h_prob = final_prob * 100

    return direction, confidence, pct, strength, trend_24h_prob