import pandas as pd
import requests
from datetime import datetime, timedelta
import ta
from config import *
from pytrends.request import TrendReq
import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
import joblib
import os
import yfinance as yf
from scipy.stats.mstats import winsorize
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# ШАГ 1: ИСПРАВЛЕНИЯ КРИТИЧЕСКИХ БАГОВ
# ============================================================================

# === 1. ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ (без изменений, работает) ===
def add_technical_indicators(df):
    """Базовые технические индикаторы"""
    df = df.copy()
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Returns and Volatility
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(20).std()
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    
    # Cleanup
    if 'volume_sma20' in df.columns:
        df = df.drop(columns=['volume_sma20'])
    
    return df


# === 2. ФИЛЬТРАЦИЯ АНОМАЛИЙ (улучшенная) ===
def filter_anomalies(df, winsor_limits=WINSOR_LIMITS):
    """Улучшенная фильтрация с winsorization"""
    if len(df) == 0:
        return df
    
    print(f"До обработки аномалий: {len(df)} строк")
    df = df.copy()
    
    # Замена inf на NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Заполнение NaN медианой
    df_num = df.select_dtypes(include=[np.number])
    for col in df_num.columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Winsorization ключевых колонок
    key_cols = ['return', 'volume_ratio', 'rsi', 'macd']
    key_cols = [col for col in key_cols if col in df.columns]
    
    for col in key_cols:
        if df[col].notna().sum() > 10:  # Минимум 10 значений
            df[col] = winsorize(df[col].dropna(), limits=winsor_limits)
    
    print(f"После обработки: {len(df)} строк (outliers clipped)")
    return df


# === 3. ON-CHAIN (без изменений, работает) ===
def add_onchain_features(df):
    """On-chain метрики"""
    print("Добавляем on-chain...")
    index = df.index
    sopr = pd.Series(1.0, index=index, name='sopr')
    mvrv = pd.Series(1.0, index=index, name='mvrv')

    # CoinMetrics SOPR
    if COINMETRICS_API_KEY and COINMETRICS_API_KEY != "YOUR_COINMETRICS_KEY":
        try:
            start = (index[0] - timedelta(days=1)).strftime('%Y-%m-%d')
            end = index[-1].strftime('%Y-%m-%d')
            url = "https://api.coinmetrics.io/v4/timeseries/asset-metrics"
            params = {
                'api_key': COINMETRICS_API_KEY,
                'assets': 'btc',
                'metrics': 'sopr',
                'frequency': '1h',
                'start_time': start,
                'end_time': end
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200 and 'data' in resp.json():
                data = resp.json()['data']
                temp_df = pd.DataFrame(data)
                temp_df['time'] = pd.to_datetime(temp_df['time'])
                temp_df.set_index('time', inplace=True)
                sopr_series = temp_df['sopr'].astype(float).resample('1h').ffill()
                sopr = sopr_series.reindex(index, method='nearest').fillna(1.0)
        except Exception as e:
            print(f"SOPR ошибка: {e}")

    # MVRV через CoinGecko
    try:
        start_ts = int((index[0] - timedelta(days=1)).timestamp())
        end_ts = int(index[-1].timestamp())
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {'vs_currency': 'usd', 'from': start_ts, 'to': end_ts}
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200 and 'market_caps' in resp.json():
            caps = resp.json()['market_caps']
            temp_df = pd.DataFrame(caps, columns=['ts', 'cap'])
            temp_df['ts'] = pd.to_datetime(temp_df['ts'], unit='ms')
            temp_df.set_index('ts', inplace=True)
            mvrv_series = temp_df['cap'].resample('1h').ffill()
            mvrv = mvrv_series.reindex(index, method='nearest')
            mvrv = (mvrv / mvrv.mean()).fillna(1.0)
    except Exception as e:
        print(f"MVRV ошибка: {e}")

    df = df.copy()
    df['sopr'] = sopr
    df['mvrv'] = mvrv
    return df


# === 4. МАКРО (ИСПРАВЛЕНО: S&P500 через yfinance) ===
def add_macro_features(df):
    """Макроэкономические показатели (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
    if df.empty or len(df) < 2:
        return df
    
    print("Добавляем макро...")
    index = df.index
    start = index[0].strftime('%Y-%m-%d')
    end = (index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Инициализация дефолтными значениями
    df['dxy'] = 100.0
    df['vix'] = 20.0
    df['sp500'] = 0.0
    
    try:
        # === S&P500 через yfinance (ИСПРАВЛЕНО!) ===
        print("  Загружаем S&P500...")
        sp500_data = yf.download('^GSPC', start=start, end=end, progress=False)
        if not sp500_data.empty and 'Close' in sp500_data.columns:
            sp500_hourly = sp500_data['Close'].resample('1h').ffill()
            sp500_reindexed = sp500_hourly.reindex(index, method='ffill')
            df['sp500'] = sp500_reindexed.fillna(method='ffill').fillna(4000)
            print(f"  S&P500 OK: {len(sp500_reindexed.dropna())} точек")
        else:
            print("  S&P500: нет данных, используем fallback")
    except Exception as e:
        print(f"  S&P500 ошибка: {e}")
    
    try:
        # === VIX через yfinance ===
        print("  Загружаем VIX...")
        vix_data = yf.download('^VIX', start=start, end=end, progress=False)
        if not vix_data.empty and 'Close' in vix_data.columns:
            vix_hourly = vix_data['Close'].resample('1h').ffill()
            vix_reindexed = vix_hourly.reindex(index, method='ffill')
            df['vix'] = vix_reindexed.fillna(method='ffill').fillna(20)
            print(f"  VIX OK: {len(vix_reindexed.dropna())} точек")
    except Exception as e:
        print(f"  VIX ошибка: {e}")
    
    try:
        # === DXY через yfinance ===
        print("  Загружаем DXY...")
        dxy_data = yf.download('DX-Y.NYB', start=start, end=end, progress=False)
        if not dxy_data.empty and 'Close' in dxy_data.columns:
            dxy_hourly = dxy_data['Close'].resample('1h').ffill()
            dxy_reindexed = dxy_hourly.reindex(index, method='ffill')
            df['dxy'] = dxy_reindexed.fillna(method='ffill').fillna(100)
            print(f"  DXY OK: {len(dxy_reindexed.dropna())} точек")
    except Exception as e:
        print(f"  DXY ошибка: {e}")
    
    return df


# === 5. FEAR & GREED (без изменений) ===
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_fear_greed_index(df):
    """Fear & Greed Index"""
    print("Добавляем Fear & Greed Index...")
    index = df.index
    fg_series = pd.Series(50.0, index=index, name='fear_greed')
    cache_path = 'data/fg_cache.pkl'

    try:
        if os.path.exists(cache_path):
            cached = joblib.load(cache_path)
            if cached.index[-1] >= index[-1]:
                fg_series = cached.reindex(index, method='nearest').fillna(50)
                print("  Использован cache для F&G")
                df['fear_greed'] = fg_series
                return df

        url = "https://api.alternative.me/fng/?limit=0"
        resp = requests.get(url, timeout=10).json()
        fg_data = resp['data']
        fg_df = pd.DataFrame(fg_data)
        fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s')
        fg_df.set_index('timestamp', inplace=True)
        fg_df['value'] = fg_df['value'].astype(float)
        fg_series = fg_df['value'].reindex(index, method='nearest').fillna(method='ffill').fillna(50)
        joblib.dump(fg_series, cache_path)
        print("  F&G загружен успешно")
    except Exception as e:
        print(f"  F&G ошибка: {e}. Fallback на 50")

    df['fear_greed'] = fg_series
    return df


# === 6. BTC DOMINANCE (ИСПРАВЛЕНО: убраны дубликаты) ===
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_btc_dominance(df):
    """BTC Dominance (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
    print("Добавляем BTC Dominance...")
    index = df.index
    dom_series = pd.Series(50.0, index=index, name='btc_dominance')
    cache_path = 'data/dom_cache.pkl'

    try:
        if os.path.exists(cache_path):
            cached = joblib.load(cache_path)
            # Убираем дубликаты из кэша перед использованием
            cached = cached[~cached.index.duplicated(keep='last')]
            dom_series = cached.reindex(index, method='nearest').fillna(50)
            print("  Использован cache для Dominance")
        else:
            start_ts = int(index[0].timestamp())
            end_ts = int(index[-1].timestamp())
            url_btc = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range?vs_currency=usd&from={start_ts}&to={end_ts}"
            
            resp_btc = requests.get(url_btc, timeout=10).json()
            if 'market_caps' in resp_btc:
                btc_cap = pd.DataFrame(resp_btc['market_caps'], columns=['ts', 'cap'])
                btc_cap['ts'] = pd.to_datetime(btc_cap['ts'], unit='ms')
                
                # === АГРЕССИВНАЯ очистка дубликатов ===
                btc_cap = btc_cap.drop_duplicates(subset='ts', keep='last')
                btc_cap = btc_cap.set_index('ts')
                
                # Убираем возможные дубликаты индекса после set_index
                btc_cap = btc_cap[~btc_cap.index.duplicated(keep='last')]
                
                btc_cap_series = btc_cap['cap'].resample('1h').ffill()
                
                # Финальная проверка на дубликаты
                btc_cap_series = btc_cap_series[~btc_cap_series.index.duplicated(keep='last')]
                
                dom_series = btc_cap_series.reindex(index, method='ffill').fillna(50.0)

                joblib.dump(dom_series, cache_path)
                print("  Dominance загружен успешно")
    except Exception as e:
        print(f"  Dominance ошибка: {e}. Fallback на 50")

    df['btc_dominance'] = dom_series
    return df


# === 7. GOOGLE TRENDS (без изменений) ===
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_google_trends(df):
    """Google Trends"""
    print("Добавляем Google Trends...")
    index = df.index
    trends_series = pd.Series(50.0, index=index, name='google_trends')
    cache_path = 'data/trends_cache.pkl'

    try:
        if os.path.exists(cache_path):
            cached = joblib.load(cache_path)
            if cached.index[-1] >= index[-1]:
                trends_series = cached.reindex(index, method='nearest').fillna(50)
                print("  Использован cache для Trends")
                df['google_trends'] = trends_series
                return df

        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ["bitcoin"]
        timeframe = f"{(index[0] - pd.Timedelta(days=7)).strftime('%Y-%m-%d')} {index[-1].strftime('%Y-%m-%d')}"
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        if 'bitcoin' in trends_df.columns:
            trends_df = trends_df['bitcoin'].resample('1h').ffill()
            trends_series = trends_df.reindex(index, method='nearest').fillna(50)
            joblib.dump(trends_series, cache_path)
            print("  Trends загружен успешно")
    except Exception as e:
        print(f"  Trends ошибка: {e}. Fallback на 50")
    
    df['google_trends'] = trends_series
    return df


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_fed_rate(df):
    """Fed Funds Rate (ИСПРАВЛЕНО: убраны дубликаты индекса)"""
    print("Добавляем Fed Funds Rate...")
    index = df.index
    rate_series = pd.Series(5.0, index=index, name='fed_rate')
    cache_path = 'data/fed_cache.pkl'

    try:
        if os.path.exists(cache_path):
            cached = joblib.load(cache_path)
            # Убираем дубликаты в кэше
            cached = cached[~cached.index.duplicated(keep='last')]
            rate_series = cached.reindex(index, method='ffill').fillna(5.0)
            print("  Использован cache для Fed Rate")
        else:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'FEDFUNDS',
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'observation_start': index[0].strftime('%Y-%m-%d'),
                'observation_end': index[-1].strftime('%Y-%m-%d')
            }
            resp = requests.get(url, params=params, timeout=10).json()
            if 'observations' in resp:
                data = [
                    {'date': obs['date'], 'value': float(obs['value'])}
                    for obs in resp['observations'] if obs['value'] != '.'
                ]
                fed_df = pd.DataFrame(data)
                fed_df['date'] = pd.to_datetime(fed_df['date'])

                # Агрессивная очистка дублей
                fed_df = fed_df.drop_duplicates(subset='date', keep='last')
                fed_df = fed_df.set_index('date')
                fed_df = fed_df[~fed_df.index.duplicated(keep='last')]

                hourly = fed_df['value'].resample('1h').ffill()
                hourly = hourly[~hourly.index.duplicated(keep='last')]

                rate_series = hourly.reindex(index, method='ffill').fillna(5.0)
                joblib.dump(rate_series, cache_path)
                print("  Fed Rate загружен успешно")
    except Exception as e:
        print(f"  Fed Rate ошибка: {e}. Fallback на 5.0")

    df['fed_rate'] = rate_series
    return df



# === 9. ДОПОЛНИТЕЛЬНЫЕ МАКРО (ИСПРАВЛЕНО) ===
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_additional_macro(df):
    """Unemployment и CPI (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
    print("Добавляем дополнительные макро (unemployment, inflation)...")
    index = df.index
    unrate_series = pd.Series(4.0, index=index, name='unemployment_rate')
    cpi_series = pd.Series(3.0, index=index, name='inflation_cpi')
    cache_path_un = 'data/unrate_cache.pkl'
    cache_path_cpi = 'data/cpi_cache.pkl'

    try:
        # Unemployment
        if os.path.exists(cache_path_un):
            cached = joblib.load(cache_path_un)
            # Убираем дубликаты из кэша
            cached = cached[~cached.index.duplicated(keep='last')]
            unrate_series = cached.reindex(index, method='ffill').fillna(4.0)
            print("  Использован cache для Unemployment")
        else:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'UNRATE',
                'api_key': FRED_API_KEY,
                'file_type': 'json',
                'observation_start': index[0].strftime('%Y-%m-%d'),
                'observation_end': index[-1].strftime('%Y-%m-%d')
            }
            resp = requests.get(url, params=params, timeout=10).json()
            if 'observations' in resp:
                data = [{'date': obs['date'], 'value': float(obs['value'])} 
                        for obs in resp['observations'] if obs['value'] != '.']
                df_un = pd.DataFrame(data)
                df_un['date'] = pd.to_datetime(df_un['date'])
                
                # === АГРЕССИВНАЯ очистка дубликатов ===
                df_un = df_un.drop_duplicates(subset='date', keep='last')
                df_un = df_un.set_index('date')
                df_un = df_un[~df_un.index.duplicated(keep='last')]
                
                unrate_series = df_un['value'].resample('1h').ffill()
                unrate_series = unrate_series[~unrate_series.index.duplicated(keep='last')]
                unrate_series = unrate_series.reindex(index, method='ffill').fillna(4.0)
                
                joblib.dump(unrate_series, cache_path_un)
                print("  Unemployment загружен успешно")

        # CPI
        if os.path.exists(cache_path_cpi):
            cached = joblib.load(cache_path_cpi)
            # Убираем дубликаты из кэша
            cached = cached[~cached.index.duplicated(keep='last')]
            cpi_series = cached.reindex(index, method='ffill').fillna(3.0)
            print("  Использован cache для CPI")
        else:
            params['series_id'] = 'CPIAUCSL'
            resp = requests.get(url, params=params, timeout=10).json()
            if 'observations' in resp:
                data = [{'date': obs['date'], 'value': float(obs['value'])} 
                        for obs in resp['observations'] if obs['value'] != '.']
                df_cpi = pd.DataFrame(data)
                df_cpi['date'] = pd.to_datetime(df_cpi['date'])
                
                # === АГРЕССИВНАЯ очистка дубликатов ===
                df_cpi = df_cpi.drop_duplicates(subset='date', keep='last')
                df_cpi = df_cpi.set_index('date')
                df_cpi = df_cpi[~df_cpi.index.duplicated(keep='last')]
                
                cpi_series = df_cpi['value'].pct_change(12)
                cpi_series = cpi_series.resample('1h').ffill()
                cpi_series = cpi_series[~cpi_series.index.duplicated(keep='last')]
                cpi_series = cpi_series.reindex(index, method='ffill').fillna(3.0) * 100
                
                joblib.dump(cpi_series, cache_path_cpi)
                print("  CPI загружен успешно")
    except Exception as e:
        print(f"  Доп. макро ошибка: {e}. Fallback")

    df['unemployment_rate'] = unrate_series
    df['inflation_cpi'] = cpi_series
    return df


# === 10. КОРРЕЛЯЦИИ И ВНЕШНИЕ АКТИВЫ ===
def add_correlations_and_external(df):
    """Корреляции с другими активами"""
    print("Добавляем корреляции и внешние цены (Nasdaq, Gold, ETH/BTC)...")
    index = df.index
    start = index[0].strftime('%Y-%m-%d')
    end = (index[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Инициализация
    df['nasdaq_close'] = 0.0
    df['gold_close'] = 0.0
    df['eth_btc_ratio'] = 0.05
    df['btc_sp500_corr'] = 0.0
    df['btc_nasdaq_corr'] = 0.0

    try:
        # Nasdaq
        nasdaq = yf.download('^IXIC', start=start, end=end, progress=False)
        if not nasdaq.empty and 'Close' in nasdaq.columns:
            nasdaq_hourly = nasdaq['Close'].resample('1h').ffill()
            df['nasdaq_close'] = nasdaq_hourly.reindex(index, method='ffill').fillna(0)
            print("  Nasdaq загружен")

        # Gold
        gold = yf.download('GC=F', start=start, end=end, progress=False)
        if not gold.empty and 'Close' in gold.columns:
            gold_hourly = gold['Close'].resample('1h').ffill()
            df['gold_close'] = gold_hourly.reindex(index, method='ffill').fillna(0)
            print("  Gold загружен")

        # Rolling correlations
        if 'sp500' in df.columns and df['sp500'].notna().sum() > 20:
            df['btc_sp500_corr'] = df['close'].rolling(20).corr(df['sp500']).fillna(0)
        
        if df['nasdaq_close'].notna().sum() > 20:
            df['btc_nasdaq_corr'] = df['close'].rolling(20).corr(df['nasdaq_close']).fillna(0)

        # ETH/BTC ratio
        start_ts = int(index[0].timestamp())
        end_ts = int(index[-1].timestamp())
        url_eth = f"https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range?vs_currency=btc&from={start_ts}&to={end_ts}"
        resp_eth = requests.get(url_eth, timeout=10).json()
        if 'prices' in resp_eth:
            eth_df = pd.DataFrame(resp_eth['prices'], columns=['ts', 'price'])
            eth_df['ts'] = pd.to_datetime(eth_df['ts'], unit='ms')
            eth_df.set_index('ts', inplace=True)
            df['eth_btc_ratio'] = eth_df['price'].resample('1h').ffill().reindex(index, method='ffill').fillna(0.05)
            print("  ETH/BTC загружен")

    except Exception as e:
        print(f"  Корреляции ошибка: {e}. Fallback")

    return df


# ============================================================================
# ШАГ 2: ПРОДВИНУТЫЕ ВРЕМЕННЫЕ ФИЧИ (НОВОЕ!)
# ============================================================================

def add_temporal_features(df):
    """
    КРИТИЧНЫЕ ВРЕМЕННЫЕ ФИЧИ для повышения точности
    Это самый важный апгрейд!
    """
    print("\n🚀 Добавляем продвинутые временные фичи...")
    df = df.copy()
    
    # === 1. ЛАГИ ЦЕНЫ (самое важное!) ===
    print("  → Создаём лаги цены и returns...")
    for lag in LAG_PERIODS:
        df[f'close_lag_{lag}h'] = df['close'].shift(lag)
        df[f'return_lag_{lag}h'] = df['return'].shift(lag)
        df[f'volume_lag_{lag}h'] = df['volume'].shift(lag)
    
    # === 2. ROLLING STATISTICS ===
    print("  → Вычисляем rolling статистики...")
    for window in ROLLING_WINDOWS:
        # Price rolling
        df[f'close_mean_{window}h'] = df['close'].rolling(window).mean()
        df[f'close_std_{window}h'] = df['close'].rolling(window).std()
        df[f'close_min_{window}h'] = df['close'].rolling(window).min()
        df[f'close_max_{window}h'] = df['close'].rolling(window).max()
        
        # Distance from extremes
        df[f'dist_from_high_{window}h'] = (df['close'] - df[f'close_max_{window}h']) / df['close']
        df[f'dist_from_low_{window}h'] = (df['close'] - df[f'close_min_{window}h']) / df['close']
        
        # Volume rolling
        df[f'volume_mean_{window}h'] = df['volume'].rolling(window).mean()
        df[f'volume_std_{window}h'] = df['volume'].rolling(window).std()
        
        # Volatility rolling
        df[f'volatility_{window}h'] = df['return'].rolling(window).std()
    
    # === 3. MOMENTUM INDICATORS ===
    print("  → Добавляем momentum индикаторы...")
    df['momentum_1h'] = df['close'].pct_change(1)
    df['momentum_3h'] = df['close'].pct_change(3)
    df['momentum_6h'] = df['close'].pct_change(6)
    df['momentum_12h'] = df['close'].pct_change(12)
    df['momentum_24h'] = df['close'].pct_change(24)
    df['momentum_48h'] = df['close'].pct_change(48)
    df['momentum_7d'] = df['close'].pct_change(168)
    
    # === 4. RATE OF CHANGE (ROC) ===
    print("  → Вычисляем rate of change...")
    for period in [3, 6, 12, 24, 72]:
        df[f'roc_{period}h'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
    
    # === 5. ЦИКЛИЧЕСКИЕ ВРЕМЕННЫЕ ФИЧИ (важно для TFT!) ===
    print("  → Создаём циклические временные фичи...")
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['week_of_year'] = df.index.isocalendar().week
    
    # Синусоидальное кодирование (лучше для ML!)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # === 6. ACCELERATION (ускорение изменения цены) ===
    print("  → Вычисляем ускорение...")
    df['acceleration_1h'] = df['momentum_1h'].diff()
    df['acceleration_6h'] = df['momentum_6h'].diff()
    df['acceleration_24h'] = df['momentum_24h'].diff()
    
    # === 7. RELATIVE STRENGTH ===
    print("  → Считаем относительную силу...")
    for window in [6, 12, 24]:
        df[f'relative_strength_{window}h'] = df['close'] / df[f'close_mean_{window}h']
    
    # === 8. VOLATILITY REGIME ===
    print("  → Определяем режим волатильности...")
    vol_median = df['volatility'].rolling(168).median()  # медианная волатильность за неделю
    df['vol_regime'] = (df['volatility'] / vol_median).fillna(1.0)
    df['high_vol_regime'] = (df['vol_regime'] > 1.5).astype(int)
    
    # === 9. TREND STRENGTH ===
    print("  → Вычисляем силу тренда...")
    for window in [24, 72, 168]:
        # Linear regression slope
        rolling_slope = df['close'].rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
            raw=True
        )
        df[f'trend_strength_{window}h'] = rolling_slope
    
    # === 10. CANDLE PATTERNS (базовые паттерны свечей) ===
    print("  → Анализируем паттерны свечей...")
    df['body'] = df['close'] - df['open']
    df['body_pct'] = df['body'] / df['open'] * 100
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_doji'] = (abs(df['body_pct']) < 0.1).astype(int)
    
    # === 11. VOLUME ANALYSIS ===
    print("  → Углубленный анализ объёма...")
    df['volume_price_trend'] = df['volume'] * df['return']
    df['obv'] = (df['volume'] * np.sign(df['return'])).cumsum()  # On-Balance Volume
    df['obv_ema'] = df['obv'].ewm(span=20).mean()
    
    # === 12. SUPPORT/RESISTANCE LEVELS ===
    print("  → Вычисляем уровни поддержки/сопротивления...")
    for window in [24, 168]:
        df[f'support_{window}h'] = df['low'].rolling(window).min()
        df[f'resistance_{window}h'] = df['high'].rolling(window).max()
        df[f'price_position_{window}h'] = (df['close'] - df[f'support_{window}h']) / \
                                           (df[f'resistance_{window}h'] - df[f'support_{window}h'] + 1e-8)
    
    # === ФИНАЛЬНАЯ ОЧИСТКА: убираем inf и заменяем на NaN ===
    print("  → Финальная очистка inf/nan...")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Считаем количество фичей
    temporal_features = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']]
    print(f"✅ Добавлено {len(temporal_features)} временных фичей!\n")
    
    return df

# === 11. ДЕРИВАТИВЫ: FUNDING + OPEN INTEREST ===
def add_derivatives_features(df):
    """Корректный и расширенный блок деривативов (funding + OI)."""
    print("Добавляем деривативы (funding, open interest)...")
    df = df.copy()
    index = df.index

    import requests
    import time

    # ---------- FUNDING RATE ----------
    funding_series = pd.Series(0.0, index=index, name='funding_rate')
    funding_cache_path = 'data/funding_cache.pkl'

    try:
        if os.path.exists(funding_cache_path):
            cached = joblib.load(funding_cache_path)
            cached = cached[~cached.index.duplicated(keep='last')]
            funding_series = cached.reindex(index, method='ffill').fillna(0.0)
            print("  Использован cache для funding")
        else:
            print("  Загружаем funding rate с Binance...")
            base_url = "https://fapi.binance.com"
            endpoint = "/fapi/v1/fundingRate"
            symbol = "BTCUSDT"

            start_ts = int((index[0] - pd.Timedelta(days=10)).timestamp() * 1000)
            end_ts = int((index[-1] + pd.Timedelta(days=1)).timestamp() * 1000)

            all_rows = []
            cur_start = start_ts
            limit = 1000

            while cur_start < end_ts:
                params = {
                    "symbol": symbol,
                    "startTime": cur_start,
                    "limit": limit,
                }
                resp = requests.get(base_url + endpoint, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break

                all_rows.extend(data)
                last_time = int(data[-1]['fundingTime'])
                cur_start = last_time + 1

                time.sleep(0.1)
                if len(data) < limit:
                    break

            if all_rows:
                f_df = pd.DataFrame(all_rows)
                f_df['fundingTime'] = pd.to_datetime(f_df['fundingTime'], unit='ms')
                f_df.set_index('fundingTime', inplace=True)
                f_df['fundingRate'] = f_df['fundingRate'].astype(float)

                f_hourly = f_df['fundingRate'].resample('1h').ffill()
                f_hourly = f_hourly[~f_hourly.index.duplicated(keep='last')]
                funding_series = f_hourly.reindex(index, method='ffill').fillna(0.0)

                joblib.dump(funding_series, funding_cache_path)
                print(f"  Funding загружен: {len(f_hourly.dropna())} точек")

    except Exception as e:
        print(f"  Funding ошибка: {e}. Fallback на 0.0")

    df['funding_rate'] = funding_series

    # Базовые фичи
    df['funding_rate_abs'] = df['funding_rate'].abs()
    df['funding_rate_rolling_24h'] = df['funding_rate'].rolling(24).mean()
    df['funding_rate_change_8h'] = df['funding_rate'].diff(8)

    # Расширенные фичи по funding
    # z-score на окне 7 дней
    roll_window = 24 * 7
    roll_mean = df['funding_rate'].rolling(roll_window).mean()
    roll_std = df['funding_rate'].rolling(roll_window).std()
    df['funding_zscore'] = (df['funding_rate'] - roll_mean) / (roll_std + 1e-8)

    # Режим funding: -1 / 0 / +1
    df['funding_regime'] = 0
    df.loc[df['funding_rate'] > 0.0001, 'funding_regime'] = 1
    df.loc[df['funding_rate'] < -0.0001, 'funding_regime'] = -1

    # Был ли flip за последние 24 часа
    df['funding_flip_24h'] = (
        np.sign(df['funding_rate']).diff(24).fillna(0).ne(0).astype(int)
    )

    # ---------- OPEN INTEREST ----------
    oi_series = pd.Series(np.nan, index=index, name='open_interest')
    oi_cache_path = 'data/oi_cache.pkl'

    try:
        if os.path.exists(oi_cache_path):
            cached = joblib.load(oi_cache_path)
            cached = cached[~cached.index.duplicated(keep='last')]
            oi_series = cached.reindex(index, method='ffill')
            print("  Использован cache для open interest")
        else:
            print("  Загружаем open interest с Binance...")
            base_url = "https://fapi.binance.com"
            endpoint = "/futures/data/openInterestHist"
            symbol = "BTCUSDT"

            start_ts = int((index[0] - pd.Timedelta(days=30)).timestamp() * 1000)
            end_ts = int((index[-1] + pd.Timedelta(days=1)).timestamp() * 1000)

            all_rows = []
            cur_start = start_ts
            limit = 500

            while cur_start < end_ts:
                params = {
                    "symbol": symbol,
                    "period": "1h",
                    "limit": limit,
                    "startTime": cur_start,
                    "endTime": end_ts,
                }
                resp = requests.get(base_url + endpoint, params=params, timeout=10)
                if resp.status_code == 400:
                    # У Binance часто 400, если диапазон слишком большой или не нравится запрос
                    print("  Open Interest 400 ошибка, прекращаем попытки и используем fallback")
                    break

                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break

                all_rows.extend(data)
                last_time = int(data[-1]['timestamp'])
                cur_start = last_time + 1

                time.sleep(0.1)
                if len(data) < limit:
                    break

            if all_rows:
                oi_df = pd.DataFrame(all_rows)
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df.set_index('timestamp', inplace=True)
                oi_df['sumOpenInterest'] = oi_df['sumOpenInterest'].astype(float)

                oi_hourly = oi_df['sumOpenInterest'].resample('1h').ffill()
                oi_hourly = oi_hourly[~oi_hourly.index.duplicated(keep='last')]
                oi_series = oi_hourly.reindex(index, method='ffill')

                joblib.dump(oi_series, oi_cache_path)
                print(f"  Open Interest загружен: {len(oi_hourly.dropna())} точек")
    except Exception as e:
        print(f"  Open Interest ошибка: {e}. Fallback")

    # Даже если OI не удалось получить, фичи должны быть валидными
    df['open_interest'] = (
        oi_series.fillna(method='ffill').fillna(method='bfill').fillna(0.0)
    )

    # Базовые фичи по OI
    df['open_interest_norm'] = df['open_interest'] / (
        df['open_interest'].rolling(24 * 30).mean() + 1e-8
    )
    df['open_interest_change_24h'] = df['open_interest'].diff(24)

    # Расширенные фичи по OI
    oi_roll_mean_7d = df['open_interest'].rolling(24 * 7).mean()
    oi_roll_std_7d = df['open_interest'].rolling(24 * 7).std()
    df['oi_zscore_7d'] = (df['open_interest'] - oi_roll_mean_7d) / (oi_roll_std_7d + 1e-8)

    df['oi_change_7d'] = df['open_interest'].diff(24 * 7)

    # Отношение OI к объёму (перегретость)
    if 'volume' in df.columns:
        vol_roll = df['volume'].rolling(24).sum()
        df['oi_volume_ratio'] = df['open_interest'] / (vol_roll + 1e-8)
    else:
        df['oi_volume_ratio'] = 0.0

    # Crowded long / crowded short
    # Высокий funding + высокий OI => crowded long
    # Низкий funding + высокий OI => crowded short
    df['crowded_long_score'] = (
        df['funding_zscore'].clip(lower=0) * df['oi_zscore_7d'].clip(lower=0)
    )
    df['crowded_short_score'] = (
        (-df['funding_zscore']).clip(lower=0) * df['oi_zscore_7d'].clip(lower=0)
    )

    return df

# === 12. LIVE ORDERBOOK FEATURES (ТОЛЬКО ДЛЯ ONLINE-ПРОГНОЗОВ) ===
@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def add_orderbook_features_live(df, symbol="BTCUSDT", depth_limit=50, use_futures=True):
    """
    Добавление фичей из ТЕКУЩЕГО ордербука Binance.

    ВАЖНО:
    - Работает только в online-режиме (auto_predict_loop), где нам нужен прогноз
      по последней свече.
    - Для исторического backtest честно использовать это нельзя, поэтому
      в backtest мы ЭТО НЕ ВЫЗЫВАЕМ.

    Фичи:
    - ob_bid_volume_sum
    - ob_ask_volume_sum
    - ob_imbalance = (bid - ask) / (bid + ask)
    - ob_top_bid, ob_top_ask, ob_spread
    - ob_weighted_mid_price
    """
    print("Добавляем LIVE orderbook фичи...")
    df = df.copy()
    index = df.index

    # Инициализируем дефолтами
    df['ob_bid_volume_sum'] = 0.0
    df['ob_ask_volume_sum'] = 0.0
    df['ob_imbalance'] = 0.0
    df['ob_top_bid'] = df['close']
    df['ob_top_ask'] = df['close']
    df['ob_spread'] = 0.0
    df['ob_weighted_mid_price'] = df['close']

    if df.empty:
        return df

    # Работаем только с последней точкой (последний timestamp)
    last_ts = index[-1]

    try:
        import requests
        base_url = "https://fapi.binance.com" if use_futures else "https://api.binance.com"
        endpoint = "/fapi/v1/depth" if use_futures else "/api/v3/depth"

        params = {
            "symbol": symbol,
            "limit": depth_limit,
        }

        resp = requests.get(base_url + endpoint, params=params, timeout=10)
        resp.raise_for_status()
        orderbook = resp.json()

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        # bids/asks: [ [price, qty], ... ]
        bid_prices = np.array([float(b[0]) for b in bids], dtype=float)
        bid_qty = np.array([float(b[1]) for b in bids], dtype=float)

        ask_prices = np.array([float(a[0]) for a in asks], dtype=float)
        ask_qty = np.array([float(a[1]) for a in asks], dtype=float)

        bid_vol_sum = bid_qty.sum()
        ask_vol_sum = ask_qty.sum()

        if bid_vol_sum + ask_vol_sum > 0:
            imbalance = (bid_vol_sum - ask_vol_sum) / (bid_vol_sum + ask_vol_sum)
        else:
            imbalance = 0.0

        top_bid = bid_prices[0] if len(bid_prices) > 0 else df.loc[last_ts, 'close']
        top_ask = ask_prices[0] if len(ask_prices) > 0 else df.loc[last_ts, 'close']
        spread = top_ask - top_bid

        # Простой взвешенный mid price
        if bid_vol_sum > 0 and ask_vol_sum > 0:
            weighted_bid = (bid_prices * bid_qty).sum() / bid_vol_sum
            weighted_ask = (ask_prices * ask_qty).sum() / ask_vol_sum
            weighted_mid = 0.5 * (weighted_bid + weighted_ask)
        else:
            weighted_mid = df.loc[last_ts, 'close']

        # Записываем только в последнюю строку
        df.loc[last_ts, 'ob_bid_volume_sum'] = bid_vol_sum
        df.loc[last_ts, 'ob_ask_volume_sum'] = ask_vol_sum
        df.loc[last_ts, 'ob_imbalance'] = imbalance
        df.loc[last_ts, 'ob_top_bid'] = top_bid
        df.loc[last_ts, 'ob_top_ask'] = top_ask
        df.loc[last_ts, 'ob_spread'] = spread
        df.loc[last_ts, 'ob_weighted_mid_price'] = weighted_mid

        print("  LIVE orderbook фичи успешно добавлены")

    except Exception as e:
        print(f"  Orderbook ошибка: {e}. Используем дефолтные значения")

    return df


# === МУЛЬТИМАСШТАБНЫЕ ФИЧИ ===
def add_multiscale_features(df_1h):
    """Добавление фичей с разных таймфреймов (1h, 4h, 12h, 1d, 3d)."""
    print("Добавляем мультимасштабные фичи (4h, 12h, 1d, 3d)...")
    df_1h = df_1h.copy()

    # --- 4h ---
    df_4h = df_1h.resample('4h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    })
    df_4h = add_technical_indicators(df_4h)
    df_4h = df_4h.add_suffix('_4h')
    df_4h = df_4h.reindex(df_1h.index, method='ffill')

    # --- 12h ---
    df_12h = df_1h.resample('12h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    })
    df_12h = add_technical_indicators(df_12h)
    df_12h = df_12h.add_suffix('_12h')
    df_12h = df_12h.reindex(df_1h.index, method='ffill')

    # --- 1d ---
    df_1d = df_1h.resample('1d').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    })
    df_1d = add_technical_indicators(df_1d)
    df_1d = df_1d.add_suffix('_1d')
    df_1d = df_1d.reindex(df_1h.index, method='ffill')

    # --- 3d ---
    df_3d = df_1h.resample('3d').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    })
    df_3d = add_technical_indicators(df_3d)
    df_3d = df_3d.add_suffix('_3d')
    df_3d = df_3d.reindex(df_1h.index, method='ffill')

    # Сшиваем всё
    df = pd.concat([df_1h, df_4h, df_12h, df_1d, df_3d], axis=1)
    print("  Мультимасштабные фичи добавлены\n")
    return df



# === ТАРГЕТЫ ===
def create_dual_target(df, short=FUTURE_TARGET_SHORT, long=FUTURE_TARGET_LONG):
    """Создание двух таргетов: короткий (6h) и длинный (24h)"""
    df = df.copy()
    if len(df) < long + 1:
        df['target_short'] = -1
        df['target_long'] = -1
        return df

    future_short = df['close'].shift(-short)
    future_long = df['close'].shift(-long)

    df['target_short'] = (future_short > df['close']).astype(int)
    df['target_long'] = (future_long > df['close']).astype(int)

    df.loc[df['target_short'].isna(), 'target_short'] = -1
    df.loc[df['target_long'].isna(), 'target_long'] = -1
    
    return df


def create_regression_target(df, future=FUTURE_TARGET_SHORT):
    """Регрессионный таргет: % изменения цены + vol-нормированный таргет."""
    df = df.copy()
    if len(df) < future + 1:
        df['pct_change'] = np.nan
        df['realized_vol_24h'] = np.nan
        df['pct_change_vol_norm'] = np.nan
        return df

    # Базовый таргет: % изменения цены вперёд на future часов
    df['future_close'] = df['close'].shift(-future)
    df['pct_change'] = (df['future_close'] - df['close']) / df['close'] * 100.0
    df.drop('future_close', axis=1, inplace=True, errors='ignore')

    # Реализованная волатильность за 24 часа (std ретёрнов в %, rolling 24 бара)
    if 'return' not in df.columns:
        df['return'] = df['close'].pct_change()

    df['realized_vol_24h'] = df['return'].rolling(24).std() * 100.0
    rv_med = df['realized_vol_24h'].median()
    df['realized_vol_24h'] = df['realized_vol_24h'].fillna(rv_med)

    # Волатильно-нормированный таргет: "сколько волатильностей" прошли
    df['pct_change_vol_norm'] = df['pct_change'] / (df['realized_vol_24h'] + 1e-8)

    return df

def add_realized_vol_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Гарантирует наличие realized_vol_24h (в %),
    используя тот же подход, что и в create_regression_target.
    """
    df = df.copy()
    if 'realized_vol_24h' not in df.columns:
        if 'return' not in df.columns:
            df['return'] = df['close'].pct_change()
        df['realized_vol_24h'] = df['return'].rolling(24).std() * 100.0
        rv_med = df['realized_vol_24h'].median()
        df['realized_vol_24h'] = df['realized_vol_24h'].fillna(rv_med)
    return df


def add_vol_regime_label(
    df: pd.DataFrame,
    vol_col: str = 'realized_vol_24h',
    low_quantile: float = 0.33,
    high_quantile: float = 0.66,
) -> pd.DataFrame:
    """
    Строит категориальный режим волатильности на основе realized_vol_24h.
    Это НЕ ломает фичи модели (мы добавляем новый столбец vol_regime_label).
    """
    df = add_realized_vol_if_missing(df).copy()

    q_low = df[vol_col].quantile(low_quantile)
    q_high = df[vol_col].quantile(high_quantile)

    def _label(v: float) -> str:
        if v <= q_low:
            return 'low_vol'
        elif v >= q_high:
            return 'high_vol'
        else:
            return 'normal_vol'

    df['vol_regime_label'] = df[vol_col].apply(_label)
    return df


def build_feature_pipeline(
    df: pd.DataFrame,
    mode: str = "train",
    use_onchain: bool = True,
    use_macro: bool = True,
    use_trends: bool = True,
    use_derivatives: bool = True,
    use_orderbook_live: bool = False,
) -> pd.DataFrame:
    """
    Единый пайплайн фичей для train/backtest/live.

    mode:
      - "train": всё для обучения (без LIVE orderbook)
      - "backtest": аналог train, но без live-запросов
      - "live": те же фичи + LIVE orderbook (если разрешено)
    """
    df = df.copy()

    # 1. Базовые ценовые фичи
    df = add_technical_indicators(df)
    df = add_multiscale_features(df)
    df = add_temporal_features(df)
    df = filter_anomalies(df)

    # 2. On-chain
    if use_onchain:
        df = add_onchain_features(df)

    # 3. Макро + связанные вещи
    if use_macro:
        df = add_macro_features(df)
        df = add_fear_greed_index(df)
        df = add_btc_dominance(df)

        if use_trends:
            df = add_google_trends(df)

        df = add_fed_rate(df)
        df = add_additional_macro(df)
        df = add_correlations_and_external(df)

    # 4. Деривативы
    if use_derivatives:
        df = add_derivatives_features(df)

    # 5. LIVE orderbook — только для live
    if mode == "live" and use_orderbook_live:
        df = add_orderbook_features_live(df)

    # 6. Волатильность + режим (категория)
    df = add_realized_vol_if_missing(df)
    df = add_vol_regime_label(df)

    return df