import pandas as pd
import numpy as np
from pathlib import Path


PREDICTIONS_DB_PATH = Path("data/predictions_db.csv")


def load_predictions():
    if not PREDICTIONS_DB_PATH.exists():
        print(f"Файл {PREDICTIONS_DB_PATH} не найден.")
        return None

    df = pd.read_csv(PREDICTIONS_DB_PATH)

    # приведение типов / названий (если нужно подогнать под твой формат — подправим)
    # ожидаем колонки:
    # timestamp, current_price, regression_pct, final_direction, final_confidence, final_pct,
    # actual_price, actual_pct, final_correct, checked, ...
    if "checked" in df.columns:
        df = df[df["checked"] == True].copy()

    # нормализуем направление по final_pct и actual_pct
    df["pred_dir"] = np.sign(df["final_pct"].fillna(0.0))
    df["real_dir"] = np.sign(df["actual_pct"].fillna(0.0))

    # корректность по направлению
    df["dir_correct"] = (df["pred_dir"] == df["real_dir"]).astype(int)

    # сила сигнала: если уже есть колонка — используем, если нет — простая классификация
    if "signal_strength" not in df.columns:
        def classify_strength(row):
            direction = str(row.get("final_direction", "")).upper()
            pct = float(row.get("final_pct", 0.0))
            conf = float(row.get("final_confidence", 0.0))

            if "НЕЙТРАЛ" in direction:
                return "NEUTRAL"

            amp = abs(pct)
            if amp >= 1.5 and conf >= 80:
                return "STRONG"
            if amp >= 1.0 and conf >= 70:
                return "MEDIUM"
            if amp >= 0.5 and conf >= 60:
                return "WEAK"
            return "VERY_WEAK"

        df["signal_strength"] = df.apply(classify_strength, axis=1)

    # режим волатильности, если сохраняется (иначе считаем unknown)
    if "vol_regime" not in df.columns:
        df["vol_regime"] = "unknown"

    return df


def evaluate_rule(df, name, mask):
    sub = df[mask].copy()
    n = len(sub)
    if n == 0:
        print(f"{name}: n=0, сигналы отсутствуют по этому правилу")
        return

    acc = sub["dir_correct"].mean()
    mae = sub["actual_pct"].sub(sub["final_pct"]).abs().mean()
    avg_move = sub["actual_pct"].mean()
    avg_abs_move = sub["actual_pct"].abs().mean()
    avg_conf = sub.get("final_confidence", pd.Series([np.nan] * n)).mean()

    print(f"\n=== ПРАВИЛО: {name} ===")
    print(f"Кол-во сигналов: {n}")
    print(f"Accuracy по направлению: {acc:.3f}")
    print(f"MAE по pct_change (final_pct vs actual): {mae:.3f}%")
    print(f"Среднее фактическое изменение: {avg_move:+.3f}%")
    print(f"Средний |факт. %|: {avg_abs_move:.3f}%")
    print(f"Средняя уверенность: {avg_conf:.1f}%")

    # Разбивка по LONG/SHORT
    for label, dsub in sub.groupby("final_direction"):
        acc_dir = dsub["dir_correct"].mean()
        n_dir = len(dsub)
        avg_move_dir = dsub["actual_pct"].mean()
        print(f"  - {label}: n={n_dir:3d}, acc={acc_dir:.3f}, mean actual={avg_move_dir:+.3f}%")


def main():
    df = load_predictions()
    if df is None or len(df) == 0:
        return

    print(f"Всего проверенных прогнозов в базе: {len(df)}")

    # Базовая метрика без фильтров
    evaluate_rule(df, "ALL_SIGNALS", mask=(df["final_direction"].notna()))

    # 1) Только STRONG + MEDIUM
    mask_strong_med = df["signal_strength"].isin(["STRONG", "MEDIUM"])
    evaluate_rule(df, "STRONG_OR_MEDIUM", mask_strong_med)

    # 2) STRONG/MEDIUM + confidence >= 80
    mask_strong_med_80 = mask_strong_med & (df["final_confidence"] >= 80)
    evaluate_rule(df, "STRONG_OR_MEDIUM + conf>=80", mask_strong_med_80)

    # 3) Исключаем high_vol режим (если он у тебя есть в колонке vol_regime)
    mask_no_high_vol = df["vol_regime"] != "high_vol"
    evaluate_rule(df, "ALL_EXCEPT_HIGH_VOL", mask_no_high_vol)

    # 4) STRONG/MEDIUM + НЕ high_vol
    mask_strong_med_no_high = mask_strong_med & mask_no_high_vol
    evaluate_rule(df, "STRONG_OR_MEDIUM & !high_vol", mask_strong_med_no_high)

    # 5) Только НЕЙТРАЛЬНЫЕ сигналы (для интереса)
    mask_neutral = df["signal_strength"] == "NEUTRAL"
    evaluate_rule(df, "NEUTRAL_ONLY", mask_neutral)

    print("\n=== АНАЛИЗ ПРАВИЛ ЗАВЕРШЁН ===")


if __name__ == "__main__":
    main()
