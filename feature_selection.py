# feature_selection.py
import os
import joblib
import numpy as np

from config import (
    SHAP_VALUES_PATH,
    FEATURE_COLS_PATH,
    SELECTED_FEATURE_COLS_PATH,
)


def select_top_features_by_shap(
    top_k: int = 80,
    min_importance_ratio: float = 0.01,
):
    """
    Выбор топ-N фич по SHAP важности.

    top_k — максимум фич, которые сохраняем.
    min_importance_ratio — отсечка по относительной важности
        (например, 0.01 = отбрасываем признаки, у которых важность < 1% от максимальной).
    """
    if not os.path.exists(SHAP_VALUES_PATH):
        raise FileNotFoundError(
            f"SHAP_VALUES_PATH не найден: {SHAP_VALUES_PATH}. "
            f"Сначала запусти backtest/train_regression, чтобы SHAP values были сохранены."
        )

    if not os.path.exists(FEATURE_COLS_PATH):
        raise FileNotFoundError(
            f"FEATURE_COLS_PATH не найден: {FEATURE_COLS_PATH}. "
            f"Убедись, что первичное обучение уже сохраняло список фич."
        )

    print(f"Загружаем SHAP values из: {SHAP_VALUES_PATH}")
    shap_values = joblib.load(SHAP_VALUES_PATH)  # shape: (n_samples, n_features)

    print(f"Загружаем список фич из: {FEATURE_COLS_PATH}")
    feature_cols = joblib.load(FEATURE_COLS_PATH)

    shap_values = np.array(shap_values)
    if shap_values.ndim == 3:
        # На всякий случай (мультиаутпут) — берём сумму по выходам
        shap_values = shap_values.sum(axis=-1)

    # Средняя абсолютная важность по каждому признаку
    mean_abs_importance = np.mean(np.abs(shap_values), axis=0)

    # Нормируем относительно максимума, чтобы можно было отсечь "хвост"
    max_imp = mean_abs_importance.max()
    rel_importance = mean_abs_importance / (max_imp + 1e-12)

    # Сортируем признаки по важности
    sorted_idx = np.argsort(-mean_abs_importance)  # по убыванию
    top_idx = sorted_idx[:top_k]

    # Фильтруем по порогу относительной важности
    selected_idx = [i for i in top_idx if rel_importance[i] >= min_importance_ratio]

    selected_features = [feature_cols[i] for i in selected_idx]

    print(f"\nВсего фич: {len(feature_cols)}")
    print(f"Топ по SHAP (до порога): {len(top_idx)}")
    print(f"Отобрано после порога {min_importance_ratio:.3f}: {len(selected_features)}")

    # Покажем топ-20 для понимания
    print("\nТоп-20 фич по важности:")
    for rank, i in enumerate(sorted_idx[:20], start=1):
        print(
            f"{rank:2d}. {feature_cols[i]:40s} | "
            f"mean|SHAP|={mean_abs_importance[i]:.6f} "
            f"(rel={rel_importance[i]*100:.2f}%)"
        )

    # Сохраняем
    os.makedirs(os.path.dirname(SELECTED_FEATURE_COLS_PATH), exist_ok=True)
    joblib.dump(selected_features, SELECTED_FEATURE_COLS_PATH)

    print(f"\n✅ Отобранные фичи сохранены в: {SELECTED_FEATURE_COLS_PATH}")
    return selected_features


if __name__ == "__main__":
    select_top_features_by_shap(top_k=80, min_importance_ratio=0.01)
