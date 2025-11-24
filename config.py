#config.py
import os

# === БАЗОВЫЕ ПАРАМЕТРЫ ===
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LOOKBACK = 168          # 1 неделя (7 дней * 24 часа)
FUTURE_TARGET_SHORT = 6 # 6h вперёд (интрадей)
FUTURE_TARGET_LONG = 24 # 24h вперёд (дневной тренд)

# === ПАРАМЕТРЫ ОБУЧЕНИЯ ===
BATCH_SIZE = 32
EPOCHS_PER_TRAIN = 30
PATIENCE = 7

# === ПУТИ К ФАЙЛАМ ===
DATA_PATH = 'data/btc_data.csv'
MODEL_PATH = 'data/btc_gru_model.pth'
SCALER_PATH = 'data/scaler.pkl'
IMPUTER_PATH = 'data/imputer.pkl'
FEATURE_COLS_PATH = 'data/feature_cols.pkl'
SELECTED_FEATURE_COLS_PATH = 'data/selected_feature_cols.pkl'

# === ПУТИ К МОДЕЛЯМ ===
LGBM_MODEL_PATH = 'data/lgbm_model.pkl'
STACKING_MODEL_PATH = 'data/stacking_model.pkl'
REGRESSION_MODEL_PATH = 'data/regression_model.pkl'
SHAP_VALUES_PATH = 'data/shap_values.pkl'
TFT_CHECKPOINT_PATH = 'models/tft.ckpt'
TFT_TRAINING_PATH = 'models/tft_training.pkl'

# === API КЛЮЧИ ===
COINMETRICS_API_KEY = "YOUR_COINMETRICS_KEY"
ALPHA_VANTAGE_KEY = "MOFCYQS4JX9W57UT"
FRED_API_KEY = "c4d279c17a9dbd1db38cad4bca7d60f5"

# === ПАРАМЕТРЫ ВРЕМЕННЫХ ФИЧЕЙ (НОВОЕ!) ===
LAG_PERIODS = [1, 2, 3, 6, 12, 24, 48, 72, 168]  # Лаги от 1h до 1 недели
ROLLING_WINDOWS = [6, 12, 24, 72, 168]           # Rolling окна

# === ПАРАМЕТРЫ WINSORIZATION ===
WINSOR_LIMITS = [0.01, 0.01]  # 1% с каждой стороны

# Создание директорий
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)