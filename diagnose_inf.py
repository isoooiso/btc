# diagnose_inf.py - ДИАГНОСТИКА INF В ДАННЫХ
import pandas as pd
import numpy as np
from data_loader import load_and_update_data
from features import (
    add_technical_indicators, filter_anomalies, add_onchain_features,
    add_macro_features, add_multiscale_features, add_fear_greed_index, 
    add_btc_dominance, add_google_trends, add_additional_macro, 
    add_correlations_and_external, add_temporal_features, add_fed_rate
)

print("="*80)
print("🔍 ДИАГНОСТИКА INF В ДАННЫХ")
print("="*80 + "\n")

# Загрузка данных
df = load_and_update_data()
print(f"Исходные данные: {len(df)} строк\n")

# Пошаговое добавление фичей с проверкой
def check_inf(df, stage_name):
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"❌ {stage_name}: найдено {inf_count} inf значений")
        inf_cols = df.select_dtypes(include=[np.number]).columns[np.isinf(df.select_dtypes(include=[np.number])).any()].tolist()
        print(f"   Колонки с inf: {inf_cols[:10]}...")  # Показываем первые 10
        return True
    else:
        print(f"✅ {stage_name}: inf не найдено")
        return False

print("Проверка после каждого этапа:\n")

df = add_technical_indicators(df)
check_inf(df, "Технические индикаторы")

df = add_multiscale_features(df)
check_inf(df, "Мультимасштабные фичи")

df = add_temporal_features(df)
if check_inf(df, "Временные фичи"):
    print("\n🔍 Подробная диагностика временных фичей:")
    for col in df.columns:
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            print(f"   - {col}: {inf_count} inf значений")
            # Показываем пример значения
            inf_sample = df[col][np.isinf(df[col])].head(3).values
            print(f"     Примеры: {inf_sample}")

df = filter_anomalies(df)
check_inf(df, "После фильтрации аномалий")

df = add_onchain_features(df)
check_inf(df, "On-chain")

df = add_macro_features(df)
check_inf(df, "Макро")

df = add_fear_greed_index(df)
check_inf(df, "Fear & Greed")

df = add_btc_dominance(df)
check_inf(df, "BTC Dominance")

df = add_google_trends(df)
check_inf(df, "Google Trends")

df = add_fed_rate(df)
check_inf(df, "Fed Rate")

df = add_additional_macro(df)
check_inf(df, "Additional Macro")

df = add_correlations_and_external(df)
check_inf(df, "Корреляции и внешние")

print("\n" + "="*80)
print("ИТОГОВАЯ СТАТИСТИКА:")
print("="*80)

numeric_cols = df.select_dtypes(include=[np.number]).columns
total_inf = np.isinf(df[numeric_cols]).sum().sum()
total_nan = df[numeric_cols].isna().sum().sum()
total_cells = len(df) * len(numeric_cols)

print(f"Всего ячеек: {total_cells:,}")
print(f"Inf значений: {total_inf:,} ({total_inf/total_cells*100:.2f}%)")
print(f"NaN значений: {total_nan:,} ({total_nan/total_cells*100:.2f}%)")

if total_inf > 0:
    print("\n❌ ПРОБЛЕМА: Обнаружены inf значения!")
    print("   Нужно добавить более агрессивную очистку в features.py")
else:
    print("\n✅ Отлично! Inf значений нет")

print("="*80)