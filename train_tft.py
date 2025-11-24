# train_tft.py
import os
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import torch

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from config import (
    FUTURE_TARGET_SHORT,
    LOOKBACK,
    TFT_CHECKPOINT_PATH,
    TFT_TRAINING_PATH,
    IMPUTER_PATH,
    SCALER_PATH,
)
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
from preprocessing import load_feature_cols, transform_with_preprocessor


# -----------------------------------------------------------
# 1. Подготовка датасета (та же логика, что в train_core_models)
# -----------------------------------------------------------

def build_full_dataset_for_tft() -> pd.DataFrame:
    """
    Строим полный датасет с теми же фичами и таргетами,
    что используем для LGBM/регрессии.
    """
    print("Загружаем и обогащаем данные для TRAIN TFT...")
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

    # таргеты — как в train_core_models / backtest
    df = create_dual_target(df, short=FUTURE_TARGET_SHORT)
    df = create_regression_target(df, future=FUTURE_TARGET_SHORT)

    # убираем хвост/начало без таргетов
    df = df[(df["target_short"] != -1) & df["pct_change"].notna()].copy()
    print(f"Всего наблюдений для TFT: {len(df)}")
    return df


# -----------------------------------------------------------
# 2. Подготовка данных для TimeSeriesDataSet
# -----------------------------------------------------------

def prepare_tft_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Грузим список фичей (тот же, что для LGBM/регрессии).
    2) Применяем ТЕ ЖЕ imputer + scaler, что были обучены в train_core_models.
    3) Собираем df_tft с колонками:
       - feature_cols (scaled)
       - 'target'  (pct_change)
       - 'time_idx'
       - 'group'
    """
    print("\nПодготавливаем данные для TFT...")

    # --- 2.1. грузим список фичей ---
    feature_cols = load_feature_cols(df)
    print(f"Фичей для TFT: {len(feature_cols)}")

    # --- 2.2. грузим имьютер и скейлер (обучены в train_core_models.py) ---
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # --- 2.3. берём только нужные фичи ---
    X = df[feature_cols].copy()
    X_scaled = transform_with_preprocessor(X, imputer, scaler)

    # собираем финальный df для TFT
    df_tft = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

    # таргет TFT — регрессионный pct_change
    df_tft["target"] = df["pct_change"].values

    # временной индекс (0..N-1)
    df_tft["time_idx"] = np.arange(len(df_tft))

    # одна группа (один тайм-сериал)
    df_tft["group"] = 0

    # убираем NaN по таргету
    df_tft = df_tft.dropna(subset=["target"]).copy()

    # 🔧 КРИТИЧЕСКИЙ ФИКС: уникальный индекс для TimeSeriesDataSet
    df_tft = df_tft.reset_index(drop=True)

    print(f"Финальный размер df_tft для TFT: {len(df_tft)}")
    return df_tft, feature_cols



# -----------------------------------------------------------
# 3. Обучение TFT
# -----------------------------------------------------------

def train_tft():
    seed_everything(42, workers=True)

    # 3.1. строим датасет
    df = build_full_dataset_for_tft()
    df_tft, feature_cols = prepare_tft_dataframe(df)

    # 3.2. train/val split по времени
    split_idx = int(0.8 * len(df_tft))
    train_df = df_tft.iloc[:split_idx].copy()
    val_df = df_tft.iloc[split_idx:].copy()

    print(f"\nTrain size (TFT): {len(train_df)}, Val size (TFT): {len(val_df)}")

    # 3.3. создаём TimeSeriesDataSet
    max_encoder_length = LOOKBACK
    max_prediction_length = 1  # предсказываем 1 шаг вперёд (6h pct_change)

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=feature_cols + ["target"],
        # можно добавить известные вперёд фичи, если появятся
        time_varying_known_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_df,
        predict=False,
        stop_randomization=True,
    )

    train_loader = training.to_dataloader(
        train=True,
        batch_size=64,
        num_workers=0,
    )
    val_loader = validation.to_dataloader(
        train=False,
        batch_size=64,
        num_workers=0,
    )

    # 3.4. создаём TFT модель
    print("\nИнициализируем TemporalFusionTransformer...")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        loss=QuantileLoss(),  # регрессия pct_change по квантилям
        log_interval=50,
        log_val_interval=1,
    )

    # 3.5. коллбеки
    ckpt_dir = os.path.dirname(TFT_CHECKPOINT_PATH)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=os.path.basename(TFT_CHECKPOINT_PATH).replace(".ckpt", ""),
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        mode="min",
    )

    # 3.6. Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        accelerator=accelerator,
        devices="auto",
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=0.1,
    )

    print("\nНачинаем обучение TFT...")
    trainer.fit(
        tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    best_ckpt = checkpoint_callback.best_model_path
    if not best_ckpt:
        print("⚠️ Не удалось найти лучший чекпоинт, сохраняем последний вес.")
        trainer.save_checkpoint(TFT_CHECKPOINT_PATH)
    else:
        # копируем/переименовываем лучший чекпоинт в TFT_CHECKPOINT_PATH
        if best_ckpt != TFT_CHECKPOINT_PATH:
            import shutil

            shutil.copy2(best_ckpt, TFT_CHECKPOINT_PATH)
        print(f"\n✅ Лучший чекпоинт TFT сохранён в: {TFT_CHECKPOINT_PATH}")

    # 3.7. сохраняем объект training TimeSeriesDataSet
    joblib.dump(training, TFT_TRAINING_PATH)
    print(f"✅ TimeSeriesDataSet (training) сохранён в: {TFT_TRAINING_PATH}")

    # 3.8. быстрая проверка качества на валидации
    tft = TemporalFusionTransformer.load_from_checkpoint(TFT_CHECKPOINT_PATH)
    tft.eval()

    with torch.no_grad():
        preds = tft.predict(val_loader, mode="prediction")
    preds = preds.cpu().numpy().reshape(-1)
    true = val_df["target"].values[: len(preds)]

    mae = np.mean(np.abs(true - preds))
    print(f"\n🔍 Быстрая оценка TFT на val: MAE pct_change ≈ {mae:.3f}%")

    print("\n🎉 Обучение TFT завершено!")


if __name__ == "__main__":
    train_tft()
