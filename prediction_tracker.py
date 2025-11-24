# prediction_tracker.py - СИСТЕМА ОТСЛЕЖИВАНИЯ ПРОГНОЗОВ
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

class PredictionTracker:
    """
    Система отслеживания и валидации прогнозов
    
    Функционал:
    - Сохранение прогнозов с временными метками
    - Автоматическая проверка через 6 часов
    - Подсчёт метрик точности для каждой модели
    - Adaptive weighting на основе performance
    """
    
    def __init__(self, db_path='data/predictions_db.csv', config_path='data/tracker_config.json'):
        self.db_path = db_path
        self.config_path = config_path
        self.load_config()
        
    def load_config(self):
        """Загрузка конфигурации (веса моделей)"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Начальные веса
            self.config = {
                'tft_weight': 0.4,
                'lgbm_weight': 0.6,
                'regression_weight': 1.0,
                'last_updated': datetime.now().isoformat()
            }
            self.save_config()
    
    def save_config(self):
        """Сохранение конфигурации"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def save_prediction(self, current_price, tft_prob, lgbm_prob, regression_pct, 
                       final_direction, final_confidence, final_pct):
        """
        Сохранение нового прогноза
        
        Args:
            current_price: текущая цена BTC
            tft_prob: вероятность от TFT (0-1)
            lgbm_prob: вероятность от LGBM (0-1)
            regression_pct: прогноз изменения в % от regression
            final_direction: финальное направление ("ЛОНГ ⬆" или "ШОРТ ⬇")
            final_confidence: финальная уверенность (0-100%)
            final_pct: финальное изменение в %
        """
        timestamp = datetime.now()
        check_time = timestamp + timedelta(hours=6)
        
        prediction = {
            'timestamp': timestamp.isoformat(),
            'check_time': check_time.isoformat(),
            'current_price': current_price,
            'tft_prob': tft_prob,
            'lgbm_prob': lgbm_prob,
            'regression_pct': regression_pct,
            'final_direction': final_direction,
            'final_confidence': final_confidence,
            'final_pct': final_pct,
            'actual_price': None,
            'actual_pct': None,
            'tft_correct': None,
            'lgbm_correct': None,
            'regression_correct': None,
            'final_correct': None,
            'checked': False
        }
        
        # Сохранение в DataFrame
        if os.path.exists(self.db_path):
            df = pd.read_csv(self.db_path)
            df = pd.concat([df, pd.DataFrame([prediction])], ignore_index=True)
        else:
            df = pd.DataFrame([prediction])
        
        df.to_csv(self.db_path, index=False)
        
        print(f"\n✅ Прогноз сохранён! ID: {len(df)}")
        print(f"   Проверка через 6 часов: {check_time.strftime('%Y-%m-%d %H:%M')}")
        
        return len(df) - 1  # Возвращаем ID прогноза
    
    def check_predictions(self, current_price):
        """
        Проверка прогнозов, которые должны были сбыться
        
        Args:
            current_price: текущая цена BTC
        """
        if not os.path.exists(self.db_path):
            print("База прогнозов пуста")
            return
        
        df = pd.read_csv(self.db_path)
        df['check_time'] = pd.to_datetime(df['check_time'], format='ISO8601')
        
        now = datetime.now()
        
        # Находим прогнозы, которые нужно проверить
        to_check = df[(df['check_time'] <= now) & (df['checked'] == False)]
        
        if len(to_check) == 0:
            print("Нет прогнозов для проверки")
            return
        
        print(f"\n🔍 Проверяем {len(to_check)} прогнозов...")
        
        for idx in to_check.index:
            original_price = df.loc[idx, 'current_price']
            actual_pct = (current_price - original_price) / original_price * 100
            
            # Сохраняем фактические данные
            df.loc[idx, 'actual_price'] = current_price
            df.loc[idx, 'actual_pct'] = actual_pct
            
            # Проверяем каждую модель
            df.loc[idx, 'tft_correct'] = self._check_direction(
                df.loc[idx, 'tft_prob'], actual_pct
            )
            df.loc[idx, 'lgbm_correct'] = self._check_direction(
                df.loc[idx, 'lgbm_prob'], actual_pct
            )
            df.loc[idx, 'regression_correct'] = self._check_direction_regression(
                df.loc[idx, 'regression_pct'], actual_pct
            )
            df.loc[idx, 'final_correct'] = self._check_direction_regression(
                df.loc[idx, 'final_pct'], actual_pct
            )
            
            df.loc[idx, 'checked'] = True
            
            # Выводим результат
            pred_time = pd.to_datetime(df.loc[idx, 'timestamp'])
            print(f"\n📊 Прогноз от {pred_time.strftime('%Y-%m-%d %H:%M')}:")
            print(f"   Цена: ${original_price:,.2f} → ${current_price:,.2f}")
            print(f"   Изменение: {actual_pct:+.2f}%")
            print(f"   TFT: {'✅' if df.loc[idx, 'tft_correct'] else '❌'}")
            print(f"   LGBM: {'✅' if df.loc[idx, 'lgbm_correct'] else '❌'}")
            print(f"   Regression: {'✅' if df.loc[idx, 'regression_correct'] else '❌'}")
            print(f"   Final: {'✅' if df.loc[idx, 'final_correct'] else '❌'}")
        
        # Сохраняем обновлённый DataFrame
        df.to_csv(self.db_path, index=False)
        
        # Обновляем веса на основе performance
        self.update_weights(df)
        
        # Показываем общую статистику
        self.show_statistics()
    
    def _check_direction(self, prob, actual_pct):
        """Проверка направления для классификатора (prob)"""
        predicted_up = prob > 0.5
        actual_up = actual_pct > 0
        return predicted_up == actual_up
    
    def _check_direction_regression(self, predicted_pct, actual_pct):
        """Проверка направления для регрессии"""
        predicted_up = predicted_pct > 0
        actual_up = actual_pct > 0
        return predicted_up == actual_up
    
    def update_weights(self, df):
        """
        Обновление весов моделей на основе recent performance
        Используем последние 20 проверенных прогнозов
        """
        checked = df[df['checked'] == True].tail(20)
        
        if len(checked) < 5:
            print("\nНедостаточно данных для обновления весов (минимум 5)")
            return
        
        # Считаем accuracy для каждой модели
        tft_acc = checked['tft_correct'].mean()
        lgbm_acc = checked['lgbm_correct'].mean()
        regression_acc = checked['regression_correct'].mean()
        
        print(f"\n📈 Performance за последние {len(checked)} прогнозов:")
        print(f"   TFT: {tft_acc:.1%}")
        print(f"   LGBM: {lgbm_acc:.1%}")
        print(f"   Regression: {regression_acc:.1%}")
        
        # Обновляем веса (softmax для нормализации)
        # Более высокая accuracy → больший вес
        weights = np.array([tft_acc, lgbm_acc])
        weights = np.exp(weights * 5)  # Температура 5 для усиления разницы
        weights = weights / weights.sum()
        
        old_tft = self.config['tft_weight']
        old_lgbm = self.config['lgbm_weight']
        
        # Плавное обновление (learning rate = 0.3)
        self.config['tft_weight'] = 0.7 * old_tft + 0.3 * weights[0]
        self.config['lgbm_weight'] = 0.7 * old_lgbm + 0.3 * weights[1]
        self.config['regression_weight'] = regression_acc
        self.config['last_updated'] = datetime.now().isoformat()
        
        print(f"\n🔄 Обновление весов:")
        print(f"   TFT: {old_tft:.3f} → {self.config['tft_weight']:.3f}")
        print(f"   LGBM: {old_lgbm:.3f} → {self.config['lgbm_weight']:.3f}")
        
        self.save_config()
    
    def show_statistics(self):
        """Показать общую статистику прогнозов"""
        if not os.path.exists(self.db_path):
            return
        
        df = pd.read_csv(self.db_path)
        checked = df[df['checked'] == True]
        
        if len(checked) == 0:
            print("\nНет проверенных прогнозов для статистики")
            return
        
        print("\n" + "="*80)
        print("📊 ОБЩАЯ СТАТИСТИКА ПРОГНОЗОВ")
        print("="*80)
        
        total = len(checked)
        
        print(f"\nВсего проверенных прогнозов: {total}")
        
        # Accuracy по моделям
        tft_acc = checked['tft_correct'].mean() * 100
        lgbm_acc = checked['lgbm_correct'].mean() * 100
        reg_acc = checked['regression_correct'].mean() * 100
        final_acc = checked['final_correct'].mean() * 100
        
        print(f"\nТочность моделей:")
        print(f"  TFT:        {tft_acc:.1f}% ({'✅' if tft_acc > 55 else '⚠️' if tft_acc > 50 else '❌'})")
        print(f"  LGBM:       {lgbm_acc:.1f}% ({'✅' if lgbm_acc > 55 else '⚠️' if lgbm_acc > 50 else '❌'})")
        print(f"  Regression: {reg_acc:.1f}% ({'✅' if reg_acc > 55 else '⚠️' if reg_acc > 50 else '❌'})")
        print(f"  Final:      {final_acc:.1f}% ({'✅' if final_acc > 55 else '⚠️' if final_acc > 50 else '❌'})")
        
        # Средняя ошибка
        mae = checked['actual_pct'].sub(checked['final_pct']).abs().mean()
        print(f"\nСредняя ошибка (MAE): {mae:.2f}%")
        
        # Лучшие/худшие прогнозы
        best = checked.iloc[checked['actual_pct'].sub(checked['final_pct']).abs().argmin()]
        worst = checked.iloc[checked['actual_pct'].sub(checked['final_pct']).abs().argmax()]
        
        print(f"\nЛучший прогноз:")
        print(f"  Время: {pd.to_datetime(best['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        print(f"  Прогноз: {best['final_pct']:+.2f}% | Факт: {best['actual_pct']:+.2f}%")
        
        print(f"\nХудший прогноз:")
        print(f"  Время: {pd.to_datetime(worst['timestamp']).strftime('%Y-%m-%d %H:%M')}")
        print(f"  Прогноз: {worst['final_pct']:+.2f}% | Факт: {worst['actual_pct']:+.2f}%")
        
        print("="*80)
    
    def get_current_weights(self):
        """Получить текущие веса для ансамбля"""
        return {
            'tft_weight': self.config['tft_weight'],
            'lgbm_weight': self.config['lgbm_weight'],
            'regression_weight': self.config['regression_weight']
        }


# Пример использования
if __name__ == "__main__":
    tracker = PredictionTracker()
    
    # Пример: сохранение прогноза
    # tracker.save_prediction(
    #     current_price=104508.60,
    #     tft_prob=0.5051,
    #     lgbm_prob=0.1307,
    #     regression_pct=1.43,
    #     final_direction="ЛОНГ ⬆",
    #     final_confidence=0.5,
    #     final_pct=1.43
    # )
    
    # Пример: проверка прогнозов
    # tracker.check_predictions(current_price=105000.00)
    
    # Показать статистику
    tracker.show_statistics()