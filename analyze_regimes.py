# analyze_regimes.py
import pandas as pd
import numpy as np

from config import FUTURE_TARGET_SHORT
from data_loader import load_and_update_data
from features import (
    add_technical_indicators,
    add_multiscale_features,
    add_temporal_features,
    filter_anomalies,
    add_onchain_features,
    add_macro_features,
    add_fear_greed_index,
    add_btc_dominance,
    add_google_trends,
    add_fed_rate,
    add_additional_macro,
    add_correlations_and_external,
    add_derivatives_features,
    create_dual_target,
    create_regression_target,
)
from sklearn.metrics import accuracy_score, mean_absolute_error


def build_full_dataset():
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
    df = add_derivatives_features(df)

    df = create_dual_target(df, short=FUTURE_TARGET_SHORT)
    df = create_regression_target(df, future=FUTURE_TARGET_SHORT)

    df = df[(df['target_short'] != -1) & df['pct_change'].notna()].copy()
    return df


def classify_vol_regime(vol_regime):
    if vol_regime < 0.8:
        return "low_vol"
    elif vol_regime > 1.5:
        return "high_vol"
    else:
        return "normal_vol"


def classify_trend(trend_strength):
    # простая грубая классификация по 72h-тренду
    if trend_strength > 0:
        return "up_trend"
    elif trend_strength < 0:
        return "down_trend"
    else:
        return "flat"


def main():
    print("Загружаем фичи...")
    df = build_full_dataset()
    df = df.copy()
    df.index.name = 'timestamp'
    df.reset_index(inplace=True)

    # Загружаем результаты backtest'а
    print("Загружаем результаты walk-forward backtest...")
    res = pd.read_csv('data/backtest_walkforward_results.csv', parse_dates=['timestamp'])

    # Мержим по времени
    merged = pd.merge(
        res,
        df,
        on='timestamp',
        how='inner',
        suffixes=('_res', '_feat')
    )

    print(f"Всего смержено точек: {len(merged)}")

    # Режимы волатильности
    if 'vol_regime' not in merged.columns:
        print("vol_regime не найден в данных. Убедись, что add_temporal_features добавляет этот столбец.")
        return

    merged['vol_bucket'] = merged['vol_regime'].apply(classify_vol_regime)

    # Режимы тренда (используем trend_strength_72h)
    if 'trend_strength_72h' not in merged.columns:
        print("trend_strength_72h не найден в данных. Убедись, что add_temporal_features добавляет этот столбец.")
        return

    merged['trend_bucket'] = merged['trend_strength_72h'].apply(classify_trend)

    # Для удобства: по направлению регрессии
    merged['y_pred_dir_reg_bool'] = (merged['y_pred_pct'] > 0).astype(int)

    # --- Сводка по волатильности ---
    print("\n=== РЕЗУЛЬТАТЫ ПО РЕЖИМАМ ВОЛАТИЛЬНОСТИ ===")
    for bucket in merged['vol_bucket'].unique():
        sub = merged[merged['vol_bucket'] == bucket]
        if len(sub) < 100:
            continue

        acc_lgbm = accuracy_score(sub['y_true_dir'], sub['y_pred_dir_lgbm'])
        acc_reg = accuracy_score(sub['y_true_dir'], sub['y_pred_dir_reg_bool'])
        mae = mean_absolute_error(sub['y_true_pct'], sub['y_pred_pct'])

        print(f"\nVol regime: {bucket} ({len(sub)} точек)")
        print(f"  LGBM dir acc:      {acc_lgbm:.3f}")
        print(f"  Regression dir acc:{acc_reg:.3f}")
        print(f"  MAE pct_change:    {mae:.3f}%")

    # --- Сводка по тренду ---
    print("\n=== РЕЗУЛЬТАТЫ ПО РЕЖИМАМ ТРЕНДА (72h) ===")
    for bucket in merged['trend_bucket'].unique():
        sub = merged[merged['trend_bucket'] == bucket]
        if len(sub) < 100:
            continue

        acc_lgbm = accuracy_score(sub['y_true_dir'], sub['y_pred_dir_lgbm'])
        acc_reg = accuracy_score(sub['y_true_dir'], sub['y_pred_dir_reg_bool'])
        mae = mean_absolute_error(sub['y_true_pct'], sub['y_pred_pct'])

        print(f"\nTrend regime: {bucket} ({len(sub)} точек)")
        print(f"  LGBM dir acc:      {acc_lgbm:.3f}")
        print(f"  Regression dir acc:{acc_reg:.3f}")
        print(f"  MAE pct_change:    {mae:.3f}%")


if __name__ == "__main__":
    main()
