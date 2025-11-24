# train_core_models.py
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error

from config import (
    FUTURE_TARGET_SHORT,
    LGBM_MODEL_PATH,
    REGRESSION_MODEL_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
    FEATURE_COLS_PATH,
    SELECTED_FEATURE_COLS_PATH,
)
from data_loader import load_and_update_data
from features import (
    build_feature_pipeline,
    create_dual_target,
    create_regression_target,
)
from preprocessing import load_feature_cols, transform_with_preprocessor
from model import train_lgbm, train_regression


def build_full_dataset() -> pd.DataFrame:
    """
    Собираем полный датасет для обучения:
    - загружаем свечи и обогащаем фичами через единый build_feature_pipeline
    - создаём таргеты
    - фильтруем NaN/invalid таргеты
    """
    print("Загружаем и обогащаем данные для TRAIN...")
    df = load_and_update_data()

    # Единый пайплайн фичей (train режим)
    df = build_feature_pipeline(
        df,
        mode="train",
        use_onchain=True,
        use_macro=True,
        use_trends=True,
        use_derivatives=True,
        use_orderbook_live=False,  # в TRAIN live-ордербук не нужен
    )

    # Таргеты
    df = create_dual_target(df, short=FUTURE_TARGET_SHORT)
    df = create_regression_target(df, future=FUTURE_TARGET_SHORT)

    # Фильтрация валидных наблюдений
    df = df[(df["target_short"] != -1) & df["pct_change"].notna()].copy()
    print(f"Всего наблюдений после подготовки (train): {len(df)}")
    return df


def main():
    # 1. Собираем датасет
    df = build_full_dataset()

    # 2. Грузим/определяем список фичей
    feature_cols = load_feature_cols(df)
    print(f"Количество фичей для модели: {len(feature_cols)}")

    X = df[feature_cols].copy()
    y_cls = df["target_short"].values
    y_reg = df["pct_change"].values

    # 3. Train/val split по времени
    split_idx = int(0.8 * len(df))
    X_train = X.iloc[:split_idx].copy()
    X_val = X.iloc[split_idx:].copy()

    y_train_cls = y_cls[:split_idx]
    y_val_cls = y_cls[split_idx:]

    y_train_reg = y_reg[:split_idx]
    y_val_reg = y_reg[split_idx:]

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    # 4. Обучаем LGBM + общий препроцессор
    lgbm_model, feature_cols_used, imputer, scaler = train_lgbm(
        X_train,
        y_train_cls,
        X_val,
        y_val_cls,
        params=None,
        df_for_cols=df,
    )

    # 4.1 Явно сохраняем препроцессор и список фичей
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols_used, FEATURE_COLS_PATH)
    print(f"\n✅ Импьютер и скейлер сохранены в:\n  {IMPUTER_PATH}\n  {SCALER_PATH}")
    print(f"✅ Список фичей сохранён в: {FEATURE_COLS_PATH}")

    # 5. Готовим данные для регрессии с тем же препроцессором
    X_train_scaled = transform_with_preprocessor(X_train, imputer, scaler)
    X_val_scaled = transform_with_preprocessor(X_val, imputer, scaler)

    # 6. Обучаем регрессионную модель
    reg_model = train_regression(
        X_train_scaled,
        y_train_reg,
        X_val_scaled,
        y_val_reg,
        feature_cols_used,
    )

    if reg_model is not None:
        joblib.dump(reg_model, REGRESSION_MODEL_PATH)
        print(f"✅ Регрессионная модель сохранена в: {REGRESSION_MODEL_PATH}")

        # Небольшая проверка качества
        y_val_pred = reg_model.predict(X_val_scaled)
        dir_acc = np.mean((y_val_pred > 0) == (y_val_reg > 0))
        mae = mean_absolute_error(y_val_reg, y_val_pred)
        print("\nПроверка на val:")
        print(f"  Direction accuracy (reg): {dir_acc:.3f}")
        print(f"  MAE pct_change (reg):     {mae:.3f}%")


if __name__ == "__main__":
    main()
