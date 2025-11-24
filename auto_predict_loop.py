# auto_predict_loop.py - ПОЛНОСТЬЮ АВТОМАТИЗИРОВАННЫЙ ЦИКЛ
"""
Этот скрипт можно запускать по расписанию (например, каждые 6 часов)
и он автоматически:
1. Проверяет старые прогнозы
2. Дообучается если накопилось достаточно данных
3. Делает новый прогноз
4. Сохраняет всё в базу
"""

import sys
from datetime import datetime
import os

import numpy as np
import joblib
import torch
import pandas as pd

from online_learning import OnlineLearner
from prediction_tracker import PredictionTracker
from data_loader import load_and_update_data
from features import build_feature_pipeline, add_realized_vol_if_missing
from config import (
    LOOKBACK,
    SCALER_PATH,
    IMPUTER_PATH,
    LGBM_MODEL_PATH,
    REGRESSION_MODEL_PATH,
    TFT_CHECKPOINT_PATH,
    TFT_TRAINING_PATH,
    STACKING_MODEL_PATH,
)


def classify_signal_strength(effective_pct: float, confidence: float, direction: str) -> str:
    """
    Грубая классификация силы сигнала для человека:
    - NEUTRAL: если direction содержит 'НЕЙТРАЛ'
    - STRONG / MEDIUM / WEAK / VERY_WEAK по сочетанию амплитуды и уверенности
    """
    if "НЕЙТРАЛ" in direction.upper():
        return "NEUTRAL"

    amp = abs(effective_pct)

    if amp >= 1.5 and confidence >= 80:
        return "STRONG"
    if amp >= 1.0 and confidence >= 70:
        return "MEDIUM"
    if amp >= 0.5 and confidence >= 60:
        return "WEAK"
    return "VERY_WEAK"


def main() -> bool:
    print("=" * 80)
    print("🤖 АВТОМАТИЧЕСКИЙ ЦИКЛ ПРОГНОЗИРОВАНИЯ BTC")
    print(f"   Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === ЭТАП 1: ПРОВЕРКА СТАРЫХ ПРОГНОЗОВ ===
    print("\n📊 ЭТАП 1: Проверка предыдущих прогнозов")
    print("-" * 80)

    tracker = PredictionTracker()

    # Загружаем данные для проверки
    df_check = load_and_update_data()
    current_price_check = df_check["close"].iloc[-1]

    tracker.check_predictions(current_price_check)

    # === ЭТАП 2: ДООБУЧЕНИЕ (если накопилось достаточно данных) ===
    print("\n🎓 ЭТАП 2: Проверка необходимости дообучения")
    print("-" * 80)

    learner = OnlineLearner(min_samples_for_retrain=50)
    retrained = learner.run()

    if retrained:
        print("\n⚠️ Модель обновлена! Используем новую версию для прогноза.")

    # === ЭТАП 3: ГЕНЕРАЦИЯ НОВОГО ПРОГНОЗА ===
    print("\n🔮 ЭТАП 3: Генерация нового прогноза")
    print("-" * 80)

    print("\nЗагрузка и обогащение данных...")
    df = load_and_update_data()

    # Единый пайплайн фичей для LIVE
    df = build_feature_pipeline(
        df,
        mode="live",
        use_onchain=True,
        use_macro=True,
        use_trends=True,
        use_derivatives=True,
        use_orderbook_live=True,
    )

    # гарантируем наличие realized_vol_24h
    df = add_realized_vol_if_missing(df)

    # текущий режим волатильности по метке (low_vol / normal_vol / high_vol)
    if "vol_regime_label" in df.columns:
        current_vol_regime = df["vol_regime_label"].iloc[-1]
    else:
        current_vol_regime = "unknown"

    print("Генерация прогноза...")

    try:
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)

        # фичи: если есть отбор, берём его, иначе базовый список
        if os.path.exists("data/selected_feature_cols.pkl"):
            feature_cols = joblib.load("data/selected_feature_cols.pkl")
        else:
            feature_cols = joblib.load("data/feature_cols.pkl")

        # Подготовка данных
        window = df.iloc[-LOOKBACK:].copy()
        latest = df.iloc[-1:].copy()

        # Очистка inf -> NaN
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

        # ===== TFT =====
        tft_prob = 0.5
        try:
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

            if os.path.exists(TFT_CHECKPOINT_PATH) and os.path.exists(TFT_TRAINING_PATH):
                training = joblib.load(TFT_TRAINING_PATH)
                tft = TemporalFusionTransformer.load_from_checkpoint(
                    TFT_CHECKPOINT_PATH, map_location=device
                )
                tft.eval()
                tft.to(device)

                encoder_data = pd.DataFrame(window_scaled, columns=feature_cols)
                encoder_data["target"] = 0.0
                encoder_data["time_idx"] = np.arange(LOOKBACK)
                encoder_data["group"] = 0
                decoder_data = encoder_data.iloc[[-1]].copy()
                decoder_data["time_idx"] = LOOKBACK
                full_pred_df = pd.concat([encoder_data, decoder_data], ignore_index=True)

                pred_dataset = TimeSeriesDataSet.from_dataset(
                    training, full_pred_df, predict=True, stop_randomization=True
                )
                pred_loader = pred_dataset.to_dataloader(
                    train=False, batch_size=1, num_workers=0
                )

                with torch.no_grad():
                    raw_preds = tft.predict(pred_loader, mode="quantiles")
                    if isinstance(raw_preds, torch.Tensor):
                        raw_preds = raw_preds.cpu().numpy()
                    median_pred = (
                        raw_preds[0, 0, 3] if raw_preds.ndim == 3 else raw_preds[0]
                    )
                    tft_prob = 1 / (1 + np.exp(-median_pred))
                    tft_prob = np.clip(tft_prob, 0.01, 0.99)
        except Exception as e:
            print(f"  TFT warning: {e}")

        # ===== LGBM =====
        lgbm_prob = 0.5
        try:
            lgbm = joblib.load(LGBM_MODEL_PATH)
            prob = lgbm.predict(latest_scaled)
            lgbm_prob = prob[0] if prob.ndim == 1 else prob[0, 1]
            lgbm_prob = np.clip(lgbm_prob, 0.01, 0.99)

            if os.path.exists("data/isotonic_calibrator.pkl"):
                calibrator = joblib.load("data/isotonic_calibrator.pkl")
                lgbm_prob = calibrator.transform([lgbm_prob])[0]
        except Exception as e:
            print(f"  LGBM warning: {e}")

        # ===== Regression: vol-нормированный -> реальные % =====
        regression_pct = 0.0
        try:
            reg = joblib.load(REGRESSION_MODEL_PATH)
            regression_pct_norm = reg.predict(latest_scaled)[0]
            rv_last = float(df["realized_vol_24h"].iloc[-1])
            regression_pct = float(regression_pct_norm * rv_last)
        except Exception:
            # fallback: грубая эвристика от LGBM
            regression_pct = float((lgbm_prob - 0.5) * 8.0)

        # ===== Адаптивные веса ансамбля =====
        weights = tracker.get_current_weights()
        tft_w = weights.get("tft_weight", 0.5)
        lgbm_w = weights.get("lgbm_weight", 0.5)

        # Линейный бленд как базовый final_prob
        final_prob = tft_prob * tft_w + lgbm_prob * lgbm_w

        # Stacking (если обучен)
        stacking_used = False
        try:
            stack = joblib.load(STACKING_MODEL_PATH)
            final_prob_stack = stack.predict_proba([[tft_prob, lgbm_prob]])[0, 1]
            final_prob = final_prob_stack
            stacking_used = True
        except Exception:
            pass

        # ===== RAW направление по регрессии =====
        raw_direction = "ЛОНГ ⬆" if regression_pct > 0 else "ШОРТ ⬇"
        raw_confidence = (
            final_prob * 100.0 if regression_pct > 0 else (1.0 - final_prob) * 100.0
        )
        raw_confidence = float(np.clip(raw_confidence, 0.0, 100.0))

        # ===== RISK-AWARE слой =====
        # масштабируем по режиму волатильности
        vol_scale_pct = 1.0
        vol_scale_conf = 1.0

        if current_vol_regime == "low_vol":
            vol_scale_pct = 1.0
            vol_scale_conf = 1.05
        elif current_vol_regime == "normal_vol":
            vol_scale_pct = 1.0
            vol_scale_conf = 1.0
        elif current_vol_regime == "high_vol":
            vol_scale_pct = 0.7
            vol_scale_conf = 0.7
        else:  # unknown
            vol_scale_pct = 1.0
            vol_scale_conf = 1.0

        effective_pct = regression_pct * vol_scale_pct

        # базовое направление / уверенность
        direction = raw_direction
        confidence = float(np.clip(raw_confidence * vol_scale_conf, 0.0, 100.0))

        # логика NEUTRAL при сильном конфликте или слабом сигнале
        neutral = False
        # слабый по амплитуде
        if abs(effective_pct) < 0.3:
            neutral = True
        # сильный конфликт: модель говорит "рост", а регрессия сильно вниз, или наоборот
        if final_prob > 0.8 and regression_pct < 0:
            neutral = True
        if final_prob < 0.2 and regression_pct > 0:
            neutral = True

        if neutral:
            direction = "НЕЙТРАЛЬНО ⚪"
            # уверенность интерпретируем как "уверены, что однозначного сигнала нет"
            confidence = float(
                np.clip(max(final_prob, 1.0 - final_prob) * 100.0, 0.0, 100.0)
            )
            effective_pct = 0.0

        # ===== Итоговая цена =====
        current_price = df["close"].iloc[-1]
        target_price = current_price * (1.0 + effective_pct / 100.0)

        # ===== Классификация силы сигнала =====
        signal_strength = classify_signal_strength(
            effective_pct=effective_pct,
            confidence=confidence,
            direction=direction,
        )

        # 👀 Диагностический вывод ансамбля + риск слоя

        print("\n---------------- АНАЛИТИКА АНСАМБЛЯ ----------------")
        print(f"TFT prob (up):          {tft_prob:.3f}")
        print(f"LGBM prob (up):         {lgbm_prob:.3f}")
        print(f"RAW reg. pct_change:    {regression_pct:+.3f}%")
        print(f"Vol regime:             {current_vol_regime}")
        print(f"Vol scales:             pct={vol_scale_pct:.2f}, conf={vol_scale_conf:.2f}")
        print(f"RAW weights:            TFT={tft_w:.3f}, LGBM={lgbm_w:.3f}")
        print(f"Stacking used:          {stacking_used}")
        print(f"Ensemble final_prob:    {final_prob:.3f} (prob роста)")
        print(f"RAW direction/conf:     {raw_direction}, {raw_confidence:.1f}%")
        print(f"RISK-AWARE pct_change:  {effective_pct:+.3f}%")
        print(
            f"RISK-AWARE direction:   {direction}, confidence: {confidence:.1f}%"
        )
        print(f"Signal strength:        {signal_strength}")
        print("----------------------------------------------------")

        # Сохранение прогноза (final_pct — уже риск-осознанный)
        prediction_id = tracker.save_prediction(
            current_price=current_price,
            tft_prob=tft_prob,
            lgbm_prob=lgbm_prob,
            regression_pct=regression_pct,  # raw
            final_direction=direction,
            final_confidence=confidence,
            final_pct=effective_pct,
        )

        # === ВЫВОД РЕЗУЛЬТАТОВ ДЛЯ ЧЕЛОВЕКА ===
        print("\n" * 1 + "=" * 80)
        print("🎯 НОВЫЙ ПРОГНОЗ СОХРАНЁН")
        print("=" * 80)
        print(f"\n💰 Текущая цена: ${current_price:,.2f}")
        print(f"🎯 Целевая цена (6h): ${target_price:,.2f}")
        print(f"\n📊 Прогноз: {direction}")
        print(f"   Изменение (risk-aware): {effective_pct:+.2f}%")
        print(f"   Изменение (raw):        {regression_pct:+.2f}%")
        print(f"   Уверенность:            {confidence:.1f}%")
        print(f"   Сила сигнала:           {signal_strength}")
        print(f"\n💾 ID прогноза: {prediction_id}")
        print("⏰ Проверка: через 6 часов")

        # Показать статистику
        tracker.show_statistics()

        print("\n" + "=" * 80)
        print("✅ ЦИКЛ ЗАВЕРШЁН УСПЕШНО")
        print("   Следующий запуск: через 6 часов")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
