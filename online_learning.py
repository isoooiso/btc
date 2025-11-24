# online_learning.py - СИСТЕМА ОНЛАЙН-ОБУЧЕНИЯ
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from config import *
from features import *

class OnlineLearner:
    """
    Система непрерывного обучения на новых данных
    
    Стратегия:
    1. Собираем новые проверенные прогнозы
    2. Когда накопилось достаточно (например, 50+) → дообучаем модели
    3. Используем incremental learning для быстрого обновления
    4. Сохраняем историю версий моделей
    """
    
    def __init__(self, min_samples_for_retrain=50):
        self.min_samples = min_samples_for_retrain
        self.predictions_db = 'data/predictions_db.csv'
        self.retrain_log = 'data/retrain_log.csv'
    
    def should_retrain(self):
        """Проверка, нужно ли дообучаться"""
        if not os.path.exists(self.predictions_db):
            return False, 0
        
        df = pd.read_csv(self.predictions_db)
        checked = df[df['checked'] == True]
        
        # Проверяем, сколько новых данных после последнего обучения
        if os.path.exists(self.retrain_log):
            log = pd.read_csv(self.retrain_log)
            last_retrain = pd.to_datetime(log.iloc[-1]['timestamp'])
            new_data = checked[pd.to_datetime(checked['timestamp']) > last_retrain]
        else:
            new_data = checked
        
        return len(new_data) >= self.min_samples, len(new_data)
    
    def retrain_lgbm(self, new_data_df):
        """
        Дообучение LightGBM на новых данных
        
        Стратегия: добавляем новые данные к валидационному сету
        и переобучаем модель с новыми гиперпараметрами
        """
        print("\n🔄 Дообучение LightGBM...")
        
        try:
            # Загружаем текущую модель и данные
            lgbm_model = joblib.load(LGBM_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            imputer = joblib.load(IMPUTER_PATH)
            feature_cols = joblib.load(SELECTED_FEATURE_COLS_PATH if os.path.exists(SELECTED_FEATURE_COLS_PATH) else FEATURE_COLS_PATH)
            
            # Загружаем полные исторические данные
            from data_loader import load_and_update_data

            
            print("  Загрузка данных...")
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
            df = create_dual_target(df, short=FUTURE_TARGET_SHORT, long=FUTURE_TARGET_LONG)
            df = df[(df['target_short'] != -1) & (df['target_long'] != -1)].copy()
            
            # Новый split: используем последние 90% для train, 10% для val
            # Это позволяет модели учиться на более свежих данных
            split_idx = int(0.9 * len(df))
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
            
            print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
            
            from preprocessing import transform_with_preprocessor

            X_train = train_df[feature_cols].copy()
            X_val = val_df[feature_cols].copy()

            # Используем уже обученный imputer + scaler
            X_train_scaled = transform_with_preprocessor(X_train, imputer, scaler)
            X_val_scaled = transform_with_preprocessor(X_val, imputer, scaler)

            
            y_train = train_df['target_short'].values
            y_val = val_df['target_short'].values
            
            # Обучение с оптимальными параметрами
            lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
            lgb_val = lgb.Dataset(X_val_scaled, label=y_val, reference=lgb_train)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
                'device': 'cpu'
            }
            
            print("  Обучение...")
            model = lgb.train(
                params,
                lgb_train,
                num_boost_round=1000,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(50)]
            )
            
            # Оценка
            val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
            acc = accuracy_score(y_val, val_pred)
            
            print(f"  ✅ Новая Val Accuracy: {acc:.3f}")
            
            # Сохранение новой версии
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f'data/lgbm_model_backup_{timestamp}.pkl'
            
            # Бэкап старой модели
            if os.path.exists(LGBM_MODEL_PATH):
                joblib.dump(lgbm_model, backup_path)
                print(f"  Старая модель сохранена: {backup_path}")
            
            # Сохранение новой модели
            joblib.dump(model, LGBM_MODEL_PATH)
            
            print("  ✅ Новая модель сохранена!")
            
            return acc
            
        except Exception as e:
            print(f"  ❌ Ошибка дообучения: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def log_retrain(self, accuracy, samples_used):
        """Логирование дообучения"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'samples_used': samples_used,
            'model_version': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        if os.path.exists(self.retrain_log):
            log_df = pd.read_csv(self.retrain_log)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])
        
        log_df.to_csv(self.retrain_log, index=False)
        print(f"\n✅ Дообучение залогировано!")
    
    def run(self):
        """Главный метод: проверка и дообучение если нужно"""
        print("="*80)
        print("🤖 ПРОВЕРКА НЕОБХОДИМОСТИ ДООБУЧЕНИЯ")
        print("="*80)
        
        should_train, n_new = self.should_retrain()
        
        if not should_train:
            print(f"\n✅ Дообучение не требуется")
            print(f"   Новых проверенных прогнозов: {n_new}/{self.min_samples}")
            return False
        
        print(f"\n🔄 Накоплено {n_new} новых прогнозов → начинаем дообучение!")
        
        # Загружаем новые данные
        df = pd.read_csv(self.predictions_db)
        checked = df[df['checked'] == True]
        
        if os.path.exists(self.retrain_log):
            log = pd.read_csv(self.retrain_log)
            last_retrain = pd.to_datetime(log.iloc[-1]['timestamp'])
            new_data = checked[pd.to_datetime(checked['timestamp']) > last_retrain]
        else:
            new_data = checked
        
        # Дообучение LGBM
        acc = self.retrain_lgbm(new_data)
        
        if acc is not None:
            self.log_retrain(acc, len(new_data))
            print("\n" + "="*80)
            print("✅ ДООБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print("❌ ДООБУЧЕНИЕ НЕ УДАЛОСЬ")
            print("="*80)
            return False


if __name__ == "__main__":
    learner = OnlineLearner(min_samples_for_retrain=50)
    learner.run()