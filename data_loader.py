import ccxt
import pandas as pd
import os
from config import DATA_PATH, SYMBOL, TIMEFRAME
import time

def fetch_binance_data(since=None, years=None, hours=None, limit_per_request=1000):
    """
    Загружает данные:
    - since: custom ms timestamp
    - years: целые годы (3)
    - hours: для обновления (2.4)
    """
    exchange = ccxt.binance()
    all_ohlcv = []

    if since is None:
        if years is not None:
            since = exchange.parse8601(f'{(pd.Timestamp.now() - pd.DateOffset(years=years)).strftime("%Y-%m-%dT00:00:00Z")}')
            total_needed = years * 365 * 24
        elif hours is not None:
            since = int((pd.Timestamp.now() - pd.Timedelta(hours=hours)).timestamp() * 1000)
            total_needed = hours
        else:
            raise ValueError("Укажи since, years или hours")

    max_per_request = limit_per_request
    fetched = 0

    print(f"Загружаем свечи 1h с Binance from {pd.to_datetime(since, unit='ms')}...")

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since, limit=max_per_request)
            if not ohlcv:
                print("   Данные закончились.")
                break
            all_ohlcv.extend(ohlcv)
            fetched += len(ohlcv)
            since = ohlcv[-1][0] + 1  # next after last
            print(f"   Загружено: {fetched} | Последняя: {pd.to_datetime(ohlcv[-1][0], unit='ms')}")

            if len(ohlcv) < max_per_request:
                break
        except Exception as e:
            print(f"   Ошибка: {e}. Повтор через 5 сек...")
            time.sleep(5)

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    print(f"Итого: {len(df)} строк | {df.index[0]} → {df.index[-1]}")
    return df

def load_and_update_data():
    """Загрузка и обновление кэша данных (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
    if os.path.exists(DATA_PATH):
        print("Обновляем кэш...")
        cached = pd.read_csv(DATA_PATH, index_col='timestamp', parse_dates=True)
        last_ts = int(cached.index[-1].timestamp() * 1000)
        
        # Обновляем последние данные (с запасом 1 час назад)
        new_data = fetch_binance_data(since=last_ts + 1)
        
        # === ИСПРАВЛЕНИЕ: Проверка на пустой DataFrame ===
        if new_data.empty:
            print("Нет новых данных для обновления")
            return cached
        
        # Фильтруем только новые данные
        new_data = new_data[new_data.index > cached.index[-1]]
        
        if new_data.empty:
            print("Нет новых данных после фильтрации")
            return cached
            
    else:
        print("Кэш не найден → загружаем 3 года...")
        new_data = fetch_binance_data(years=3)
        cached = pd.DataFrame()

    # Объединяем
    combined = pd.concat([cached, new_data])
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    # Gap detection
    expected = pd.date_range(start=combined.index.min(), end=combined.index.max(), freq='1h')
    missing = expected.difference(combined.index)
    
    if len(missing) > 0:
        print(f"Обнаружено {len(missing)} пропусков. Дозагружаем...")
        for miss in sorted(missing[:10]):  # Ограничиваем 10 пропусками для скорости
            since_miss = int((miss - pd.Timedelta(hours=1)).timestamp() * 1000)
            gap_data = fetch_binance_data(since=since_miss)
            if not gap_data.empty:
                combined = pd.concat([combined, gap_data]).sort_index().drop_duplicates(keep='last')
        
        # Повторная проверка пропусков
        if len(missing) > 10:
            print(f"Осталось {len(missing) - 10} пропусков (пропущено для скорости)")

    # Сохранение
    combined.to_csv(DATA_PATH)
    print(f"Кэш обновлён: {len(combined)} строк")
    return combined