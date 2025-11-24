# backtest.py
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb

from config import FUTURE_TARGET_SHORT
from data_loader import load_and_update_data
from features import (
    build_feature_pipeline,
    create_dual_target,
    create_regression_target,
)
from preprocessing import load_feature_cols
from model import train_regression


# ------------------------
# 1. Подготовка датасета
# ------------------------


def build_full_dataset() -> pd.DataFrame:
    print("Загружаем и обогащаем данные для backtest...")
    df = load_and_update_data()

    # Единый пайплайн фичей (backtest режим)
    df = build_feature_pipeline(
        df,
        mode="backtest",
        use_onchain=True,
        use_macro=True,
        use_trends=True,
        use_derivatives=True,
        use_orderbook_live=False,  # backtest без live-ордербука
    )

    df = create_dual_target(df, short=FUTURE_TARGET_SHORT)
    df = create_regression_target(df, future=FUTURE_TARGET_SHORT)

    df = df[(df["target_short"] != -1) & df["pct_change"].notna()].copy()

    print(f"Всего наблюдений после подготовки: {len(df)}")
    return df


# --------------------------------
# 2. Локальный препроцессинг
# --------------------------------


def clean_inf_nan(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def fit_local_preprocessor(X_train: pd.DataFrame):
    """
    Локальный imputer + scaler для backtest.
    Ничего не сохраняем на диск, чтобы не трогать продовую модель.
    """
    X_train = clean_inf_nan(X_train)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, imputer, scaler


def transform_local(
    X: pd.DataFrame, imputer: SimpleImputer, scaler: StandardScaler
) -> np.ndarray:
    X = clean_inf_nan(X)
    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    return X_scaled


# --------------------------------
# 3. Локальное обучение LGBM
# --------------------------------


def train_lgbm_local(X_train, y_train, X_val, y_val, params=None):
    """
    Упрощённая версия обучения LGBM для backtest (без сохранения на диск).
    """
    if params is None:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
            "device": "cpu",
        }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    print("  Обучаем LGBM (локальный backtest)...")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50)],
    )

    val_pred = (model.predict(X_val) > 0.5).astype(int)
    acc = accuracy_score(y_val, val_pred)
    print(f"  LGBM Val Acc (локально): {acc:.3f}")

    return model


# --------------------------------
# 4. Walk-forward backtest
# --------------------------------


def walk_forward_backtest(
    df: pd.DataFrame,
    train_start_ratio: float = 0.5,
    retrain_every: int = 24,  # переобучаем модель раз в 24 шага (примерно раз в сутки)
    step: int = 1,  # шаг по времени (1 час)
):
    """
    Честный walk-forward:
    - сначала берём train_start_ratio данных для первого обучения;
    - потом двигаемся вперёд по времени:
      - каждые `retrain_every` шагов переобучаем модель на всех данных до текущего момента;
      - на каждом шаге делаем прогноз для следующей точки.
    """

    # используем тот же список фичей, что и в проде
    feature_cols = load_feature_cols(df)

    n = len(df)
    start_idx = int(n * train_start_ratio)

    print(
        f"Запуск walk-forward backtest: "
        f"train_start_ratio={train_start_ratio}, retrain_every={retrain_every}, step={step}"
    )
    print(f"Начальный индекс начала теста: {start_idx}, всего точек: {n}")

    # Аккумуляторы результатов
    timestamps = []
    y_true_dir = []
    y_pred_dir_lgbm = []
    y_pred_dir_reg = []

    y_true_pct = []
    y_pred_pct = []

    # Текущие модель и препроцессор
    lgbm_model = None
    reg_model = None
    imputer = None
    scaler = None

    # Итерируем по индексам тестовых точек
    # На шаге i мы прогнозируем для точки idx, тренируясь на [0:idx)
    for i, idx in enumerate(range(start_idx, n, step)):
        if idx >= n:
            break

        # каждые retrain_every шагов переобучаемся
        if (lgbm_model is None) or (i % retrain_every == 0):
            train_df = df.iloc[:idx].copy()

            X_train = train_df[feature_cols]
            y_train_cls = train_df["target_short"].values
            y_train_reg = train_df["pct_change"].values

            # делим train_df на внутренние train/val для контроля LGBM
            split_inner = int(len(train_df) * 0.8)
            X_tr_inner = X_train.iloc[:split_inner].copy()
            y_tr_inner_cls = y_train_cls[:split_inner]

            X_val_inner = X_train.iloc[split_inner:].copy()
            y_val_inner_cls = y_train_cls[split_inner:]

            # препроцессинг
            X_tr_scaled, imputer, scaler = fit_local_preprocessor(X_tr_inner)
            X_val_scaled = transform_local(X_val_inner, imputer, scaler)

            # обучение LGBM
            lgbm_model = train_lgbm_local(
                X_tr_scaled,
                y_tr_inner_cls,
                X_val_scaled,
                y_val_inner_cls,
            )

            # обучение регрессии (используем ту же нормализацию)
            X_train_reg_scaled = transform_local(X_train, imputer, scaler)
            reg_model = train_regression(
                X_train_reg_scaled,
                y_train_reg,
                X_train_reg_scaled,  # для backtest тут не критично,
                y_train_reg,  # главное — получить модель
                feature_cols=feature_cols,
            )

            print(f"\n=== Переобучение на шаге i={i}, idx={idx} ===\n")

        # --- тестовая точка ---
        test_row = df.iloc[idx]
        X_test = pd.DataFrame([test_row[feature_cols]], columns=feature_cols)
        timestamps.append(test_row.name)

        X_test_scaled = transform_local(X_test, imputer, scaler)

        # LGBM: вероятность роста
        proba_up = lgbm_model.predict(X_test_scaled)[0]
        pred_cls = int(proba_up > 0.5)

        # Regression: предсказанный % изменения
        if reg_model is not None:
            pct_pred = float(reg_model.predict(X_test_scaled)[0])
        else:
            pct_pred = 0.0

        # Истинные значения
        true_cls = int(test_row["target_short"])
        true_pct = float(test_row["pct_change"])

        # Записываем
        y_true_dir.append(true_cls)
        y_pred_dir_lgbm.append(pred_cls)
        y_pred_dir_reg.append(int(pct_pred > 0))

        y_true_pct.append(true_pct)
        y_pred_pct.append(pct_pred)

    # ---- Финальная агрегация ----
    y_true_dir = np.array(y_true_dir)
    y_pred_dir_lgbm = np.array(y_pred_dir_lgbm)
    y_pred_dir_reg = np.array(y_pred_dir_reg)

    y_true_pct = np.array(y_true_pct)
    y_pred_pct = np.array(y_pred_pct)

    print("\n=== ИТОГИ WALK-FORWARD BACKTEST ===")
    print(f"Количество тестовых точек: {len(y_true_dir)}")

    acc_lgbm = accuracy_score(y_true_dir, y_pred_dir_lgbm)
    acc_reg_dir = accuracy_score(y_true_dir, y_pred_dir_reg)
    mae_pct = mean_absolute_error(y_true_pct, y_pred_pct)

    print(f"Direction accuracy LGBM: {acc_lgbm:.3f}")
    print(f"Direction accuracy Regression (по знаку pct_change): {acc_reg_dir:.3f}")
    print(f"MAE по pct_change: {mae_pct:.3f}%")

    # Можно сохранить результаты для дальнейшего анализа
    results_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true_dir": y_true_dir,
            "y_pred_dir_lgbm": y_pred_dir_lgbm,
            "y_pred_dir_reg": y_pred_dir_reg,
            "y_true_pct": y_true_pct,
            "y_pred_pct": y_pred_pct,
        }
    )
    results_df.to_csv("data/backtest_walkforward_results.csv", index=False)
    print(
        "Результаты walk-forward сохранены в data/backtest_walkforward_results.csv"
    )


# --------------------------------
# 5. Entry point
# --------------------------------


def main():
    df = build_full_dataset()
    walk_forward_backtest(
        df,
        train_start_ratio=0.5,  # половина истории на первичное обучение
        retrain_every=24,  # переобучаемся раз в сутки
        step=1,  # прогноз на каждый час
    )


if __name__ == "__main__":
    main()
