import os
import numpy as np
import pandas as pd

# будем использовать тот же загрузчик данных, что и в боевом коде
from data_loader import load_and_update_data


def load_predictions_db():
    """
    Пытаемся взять путь из config.PREDICTIONS_DB_PATH, если он есть.
    Если нет – используем data/predictions_db.csv.
    """
    db_path = None
    try:
        from config import PREDICTIONS_DB_PATH  # type: ignore
        db_path = PREDICTIONS_DB_PATH
    except Exception:
        db_path = os.path.join("data", "predictions_db.csv")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Файл с прогнозами не найден: {db_path}")

    df = pd.read_csv(db_path)

    expected_cols = [
        "timestamp",
        "check_time",
        "current_price",
        "tft_prob",
        "lgbm_prob",
        "regression_pct",
        "final_direction",
        "final_confidence",
        "final_pct",
        "actual_price",
        "actual_pct",
        "tft_correct",
        "lgbm_correct",
        "regression_correct",
        "final_correct",
        "checked",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"В predictions_db.csv нет колонок: {missing}")

    return df


def compute_signal_strength_row(row):
    """
    Восстанавливаем силу сигнала по final_direction, final_pct и final_confidence.
    Логика согласована с auto_predict_loop.py.
    """
    direction = str(row["final_direction"]) if pd.notna(row["final_direction"]) else ""
    final_pct = float(row["final_pct"])
    conf = float(row["final_confidence"])

    # Нейтральный сигнал
    if "НЕЙТРАЛ" in direction.upper():
        return "NEUTRAL"

    mag = abs(final_pct)

    if conf >= 90 and mag >= 1.5:
        return "STRONG"
    elif conf >= 75 and mag >= 1.0:
        return "MEDIUM"
    elif conf >= 60 and mag >= 0.5:
        return "WEAK"
    else:
        return "VERY_WEAK"


def build_market_regimes():
    """
    Загружаем историю цен через load_and_update_data()
    и строим на ней:
      - 24h realized volatility (rv_24h)
      - 72h trend (trend_72h)
      - режимы волатильности: low_vol / normal_vol / high_vol
      - режим тренда: down_trend / sideways / up_trend

    Возвращаем DataFrame с колонками:
      ['mkt_time', 'rv_24h', 'vol_regime', 'trend_72h', 'trend_regime']
    """
    print("\n[REGIME] Загружаем рыночные данные для анализа режимов...")
    df = load_and_update_data().copy()

    # предполагаем, что index – это datetime, если нет – пробуем привести
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
        else:
            df.index = pd.to_datetime(df.index)

    # базовые вещи
    df = df.sort_index()
    df["return"] = df["close"].pct_change()

    # 24h realized vol % (аналогично тому, как ты делал в таргете)
    df["rv_24h"] = df["return"].rolling(24).std() * 100.0
    # подчищаем
    med_rv = df["rv_24h"].median()
    df["rv_24h"] = df["rv_24h"].fillna(med_rv)

    # 72h тренд в %
    df["trend_72h"] = df["close"].pct_change(72) * 100.0
    df["trend_72h"] = df["trend_72h"].fillna(0.0)

    # --- режимы волатильности по квантилям ---
    q1, q2 = df["rv_24h"].quantile([0.33, 0.66])

    def _vol_regime(rv):
        if rv <= q1:
            return "low_vol"
        elif rv <= q2:
            return "normal_vol"
        else:
            return "high_vol"

    df["vol_regime"] = df["rv_24h"].apply(_vol_regime)

    # --- режим тренда ---
    # небольшой порог, чтобы отсечь шум
    trend_thr = 0.5  # 0.5% за 72 часа

    def _trend_regime(tr):
        if tr >= trend_thr:
            return "up_trend"
        elif tr <= -trend_thr:
            return "down_trend"
        else:
            return "sideways"

    df["trend_regime"] = df["trend_72h"].apply(_trend_regime)

    # для merge_asof сбрасываем index в колонку времени
    df_regimes = df[["rv_24h", "vol_regime", "trend_72h", "trend_regime"]].copy()
    df_regimes = df_regimes.reset_index().rename(columns={df_regimes.index.name or "index": "mkt_time"})

    # если после reset_index имя получилось странное — ещё раз подправим
    if "mkt_time" not in df_regimes.columns:
        # предположим, что первая колонка — это время
        first_col = df_regimes.columns[0]
        df_regimes = df_regimes.rename(columns={first_col: "mkt_time"})

    return df_regimes


def attach_regimes_to_predictions(df_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого прогноза подбираем ближайшую (по времени <= timestamp) свечу
    и подтягиваем к нему:
      - rv_24h
      - vol_regime
      - trend_72h
      - trend_regime
    """
    df_regimes = build_market_regimes()

    # timestamp прогноза
    df_pred = df_pred.copy()
    df_pred["timestamp_dt"] = pd.to_datetime(df_pred["timestamp"])

    # сортируем для merge_asof
    df_pred_sorted = df_pred.sort_values("timestamp_dt")
    df_regimes_sorted = df_regimes.sort_values("mkt_time")

    # asof-merge: для каждого прогноза берём последнюю доступную свечу
    merged = pd.merge_asof(
        df_pred_sorted,
        df_regimes_sorted,
        left_on="timestamp_dt",
        right_on="mkt_time",
        direction="backward",
    )

    return merged


def main():
    df = load_predictions_db()

    # Берём только проверенные прогнозы
    df = df[df["checked"] == True].copy()

    if df.empty:
        print("Нет проверенных прогнозов (checked=True).")
        return

    # сначала прикрутим режимы рынка
    df = attach_regimes_to_predictions(df)

    n_total = len(df)
    print(f"\nВсего проверенных прогнозов: {n_total}\n")

    # Приводим типы
    for col in ["tft_correct", "lgbm_correct", "regression_correct", "final_correct"]:
        df[col] = df[col].astype(bool)

    # === ОБЩАЯ ТОЧНОСТЬ ===
    print("=== ОБЩАЯ ТОЧНОСТЬ МОДЕЛЕЙ И АНСАМБЛЯ ===")
    acc_tft = df["tft_correct"].mean()
    acc_lgbm = df["lgbm_correct"].mean()
    acc_reg = df["regression_correct"].mean()
    acc_final = df["final_correct"].mean()

    print(f"TFT                 : acc = {acc_tft:.3f} (n={n_total})")
    print(f"LGBM                : acc = {acc_lgbm:.3f} (n={n_total})")
    print(f"Regression (sign)   : acc = {acc_reg:.3f} (n={n_total})")
    print(f"Final ensemble      : acc = {acc_final:.3f} (n={n_total})")
    print()

    # === РЕГРЕССИЯ ===
    print("=== РЕГРЕССИЯ (прогноз % изменения) ===")
    reg_pct = df["regression_pct"].astype(float)
    act_pct = df["actual_pct"].astype(float)

    mae_reg = np.mean(np.abs(reg_pct - act_pct))
    dir_acc_reg = np.mean((reg_pct > 0) == (act_pct > 0))

    print(f"MAE по pct_change (regression vs actual): {mae_reg:.3f}%")
    print(f"Direction accuracy по знаку regression_pct: {dir_acc_reg:.3f}")
    print()

    # === FINAL PCT (risk-aware) ===
    print("=== RISK-AWARE % (final_pct) ===")
    final_pct = df["final_pct"].astype(float)
    mae_final = np.mean(np.abs(final_pct - act_pct))
    dir_acc_final_pct = np.mean((final_pct > 0) == (act_pct > 0))

    print(f"MAE по pct_change (final_pct vs actual): {mae_final:.3f}%")
    print(f"Direction accuracy по знаку final_pct:   {dir_acc_final_pct:.3f}")
    print()

    # === ТОЧНОСТЬ ПО НАПРАВЛЕНИЮ ===
    print("=== ТОЧНОСТЬ ПО НАПРАВЛЕНИЮ (final_direction) ===")
    df_long = df[df["final_direction"].str.contains("ЛОНГ", na=False)]
    df_short = df[df["final_direction"].str.contains("ШОРТ", na=False)]
    df_neutral = df[df["final_direction"].str.contains("НЕЙТРАЛ", na=False)]

    def _print_dir_stats(name, sub_df):
        n = len(sub_df)
        if n == 0:
            print(f"{name:<10}: n=  {n:3d}")
            return
        acc = sub_df["final_correct"].mean()
        mean_act = sub_df["actual_pct"].mean()
        mean_abs = sub_df["actual_pct"].abs().mean()
        print(
            f"{name:<10}: n={n:4d}, acc={acc:.3f}, "
            f"mean actual={mean_act:+.3f}%, mean |actual|={mean_abs:.3f}%"
        )

    _print_dir_stats("ЛОНГ ⬆", df_long)
    _print_dir_stats("ШОРТ ⬇", df_short)
    _print_dir_stats("НЕЙТРАЛ", df_neutral)
    print()

    # === БИНЫ УВЕРЕННОСТИ ===
    print("=== ТОЧНОСТЬ ПО БИНАМ УВЕРЕННОСТИ (final_conf_pct) ===")
    df["final_conf_pct"] = df["final_confidence"].astype(float)

    bins = [0, 40, 70, 90, 100.1]
    labels = ["0-40", "40-70", "70-90", "90-100"]
    df["conf_bin"] = pd.cut(df["final_conf_pct"], bins=bins, labels=labels, right=False)

    for label in labels:
        sub = df[df["conf_bin"] == label]
        n = len(sub)
        if n == 0:
            continue
        acc = sub["final_correct"].mean()
        avg_conf = sub["final_conf_pct"].mean()
        avg_abs = sub["actual_pct"].abs().mean()
        print(
            f"conf={label:<5}: n={n:4d}, acc={acc:.3f}, "
            f"avg_conf={avg_conf:.1f}%, avg |actual_pct|={avg_abs:.3f}%"
        )
    print()

    # === СИЛА СИГНАЛА (recomputed) ===
    print("=== ТОЧНОСТЬ ПО СИЛЕ СИГНАЛА (recomputed) ===")
    df["signal_strength"] = df.apply(compute_signal_strength_row, axis=1)

    strength_order = ["STRONG", "MEDIUM", "WEAK", "VERY_WEAK", "NEUTRAL"]

    for strength in strength_order:
        sub = df[df["signal_strength"] == strength]
        n = len(sub)
        if n == 0:
            continue
        acc = sub["final_correct"].mean()
        mean_conf = sub["final_conf_pct"].mean()
        mean_mag = sub["final_pct"].abs().mean()
        mean_abs_act = sub["actual_pct"].abs().mean()
        print(
            f"{strength:<9}: n={n:4d}, acc={acc:.3f}, "
            f"avg_conf={mean_conf:.1f}%, avg |pred_pct|={mean_mag:.3f}%, "
            f"avg |actual_pct|={mean_abs_act:.3f}%"
        )
    print()

    # === РЕЖИМЫ РЫНКА: ВОЛАТИЛЬНОСТЬ ===
    print("=== РЕЗУЛЬТАТЫ ПО РЕЖИМАМ ВОЛАТИЛЬНОСТИ (rv_24h) ===")
    for regime in ["low_vol", "normal_vol", "high_vol"]:
        sub = df[df["vol_regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        acc = sub["final_correct"].mean()
        mae_regime = np.mean(np.abs(sub["final_pct"].astype(float) - sub["actual_pct"].astype(float)))
        avg_rv = sub["rv_24h"].mean()
        print(
            f"{regime:<10}: n={n:4d}, acc={acc:.3f}, "
            f"MAE={mae_regime:.3f}%, avg rv_24h={avg_rv:.2f}%"
        )
    print()

    # === РЕЖИМЫ РЫНКА: ТРЕНД ===
    print("=== РЕЗУЛЬТАТЫ ПО РЕЖИМАМ ТРЕНДА (72h) ===")
    for regime in ["down_trend", "sideways", "up_trend"]:
        sub = df[df["trend_regime"] == regime]
        n = len(sub)
        if n == 0:
            continue
        acc = sub["final_correct"].mean()
        mae_regime = np.mean(np.abs(sub["final_pct"].astype(float) - sub["actual_pct"].astype(float)))
        avg_trend = sub["trend_72h"].mean()
        print(
            f"{regime:<10}: n={n:4d}, acc={acc:.3f}, "
            f"MAE={mae_regime:.3f}%, avg trend_72h={avg_trend:+.2f}%"
        )

    print("\n=== АНАЛИЗ ЗАВЕРШЁН ===")


if __name__ == "__main__":
    main()
