# preprocessing.py
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (
    FEATURE_COLS_PATH,
    SELECTED_FEATURE_COLS_PATH,
    SCALER_PATH,
    IMPUTER_PATH,
)

EXCLUDE_COLS = [
    'target',
    'target_short',
    'target_long',
    'pct_change',
    'future_close',
]


def infer_feature_cols(df: pd.DataFrame) -> list:
    """
    Инференс списка фичей из df:
    исключаем таргеты и служебные столбцы.
    """
    exclude = {
        'target',
        'target_short',
        'target_long',
        'pct_change',
        'pct_change_vol_norm',
        'future_close',
        'realized_vol_24h',
    }
    return [c for c in df.columns if c not in exclude]


def load_feature_cols(df: pd.DataFrame = None):
    """
    Загружаем список фичей:
    1) если есть SELECTED_FEATURE_COLS_PATH -> используем его (после feature selection),
    2) иначе если есть FEATURE_COLS_PATH -> используем его (первичный список),
    3) иначе инферим из df и сохраняем в FEATURE_COLS_PATH.
    """
    if os.path.exists(SELECTED_FEATURE_COLS_PATH):
        return joblib.load(SELECTED_FEATURE_COLS_PATH)

    if os.path.exists(FEATURE_COLS_PATH):
        return joblib.load(FEATURE_COLS_PATH)

    if df is None:
        raise ValueError(
            "Невозможно инферить feature_cols: нет ни файлов, ни df. "
            "Передай df в load_feature_cols(df)."
        )

    feature_cols = infer_feature_cols(df)
    os.makedirs(os.path.dirname(FEATURE_COLS_PATH), exist_ok=True)
    joblib.dump(feature_cols, FEATURE_COLS_PATH)
    return feature_cols


def clean_inf_nan(X: pd.DataFrame):
    """Заменяем inf на NaN, чтобы дальше им не ломать импретер."""
    X = X.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def fit_preprocessor(X_train: pd.DataFrame):
    """
    Обучаем imputer + scaler и возвращаем преобразованный X_train.
    Одно место, где они должны фититься.
    """
    X_train = clean_inf_nan(X_train)

    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, imputer, scaler


def load_preprocessor():
    """
    Загружаем imputer + scaler для инференса или retrain.
    """
    if not os.path.exists(IMPUTER_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Imputer / scaler не найдены. "
            "Сначала обучи их через fit_preprocessor при первичном training."
        )

    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return imputer, scaler


def transform_with_preprocessor(X: pd.DataFrame, imputer=None, scaler=None):
    """
    Применяем уже обученный imputer + scaler к X.
    """
    X = clean_inf_nan(X)

    if imputer is None or scaler is None:
        imputer, scaler = load_preprocessor()

    X_imp = imputer.transform(X)
    X_scaled = scaler.transform(X_imp)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    return X_scaled
