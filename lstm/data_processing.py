import pandas as pd
import os
import ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta.momentum
import ta.trend
import ta.volatility
import ta.volume

def normalize_series(series):
    """Normaliza una serie usando min-max scaling, valores en [0,1]."""
    if series.max() == series.min():
        return series - series.min()  # Si es constante
    return (series - series.min()) / (series.max() - series.min())

def calculate_technical_indicators(df):  # Ya NO recibe ohlc_prefix
    """
    Calcula indicadores técnicos.  Usa columnas normalizadas si existen,
    si no, usa las originales.
    """

    
    close_col = 'close'
    high_col = 'high'
    low_col = 'low'
    volume_col = "volume"

    # Verifica columnas necesarias (usa las NO normalizadas para verificar)
    if not all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
        raise ValueError("Columnas OHLC/Volume necesarias no están presentes.")

    # --- Indicadores de Tendencia ---
    # Medias móviles optimizadas para H1
    df['SMA_34'] = ta.trend.sma_indicator(df[close_col], window=34)
    df['EMA_21'] = ta.trend.ema_indicator(df[close_col], window=21)
    df['EMA_8'] = ta.trend.ema_indicator(df[close_col], window=8)
    
    # MACD adaptado para H1
    df['MACD_line'] = ta.trend.macd(df[close_col], window_slow=21, window_fast=8)
    df['MACD_signal'] = ta.trend.macd_signal(df[close_col], window_slow=21, window_fast=8, window_sign=5)
    df['MACD_diff'] = ta.trend.macd_diff(df[close_col], window_slow=21, window_fast=8, window_sign=5)

    # --- Indicadores de Momentum ---
    df['RSI_14'] = ta.momentum.rsi(df[close_col], window=14)
    stoch = ta.momentum.StochasticOscillator(
        df[high_col], 
        df[low_col], 
        df[close_col], 
        window=14, 
        smooth_window=3
    )
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()

    # --- Indicadores de Volatilidad ---
    df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    bollinger = ta.volatility.BollingerBands(df[close_col], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Width'] = bollinger.bollinger_wband()

    # --- Indicadores de Volumen ---
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df[volume_col]).on_balance_volume()
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df[volume_col], window=14).volume_weighted_average_price()
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df[volume_col], window=20).chaikin_money_flow()

    # --- Otros Indicadores ---
    df['ADX_14'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

    return df

def calculate_trend_indicator(df):
    """
    Calcula un indicador de tendencia continuo normalizado entre 0 y 1.
    
    Returns:
        DataFrame con nueva columna 'TrendStrength' donde:
        - Valores cercanos a 1: Fuerte tendencia alcista
        - Valores cercanos a 0: Fuerte tendencia bajista
        - Valores cercanos a 0.5: Mercado estable/lateral
    """
    # Validación de datos
    required_cols = ['RSI_14', 'Stoch_k', 'EMA_8', 'EMA_21', 'SMA_34', 
                    'BB_Low', 'BB_High', 'BB_Width', 'VWAP', 'CMF', 'close',
                    'MACD_diff', 'ADX_14']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Faltan columnas requeridas para el cálculo de tendencia")

    # Componentes direccionales
    trend_direction = (
        # Momentum
        0.3 * (df['RSI_14'] / 100) +  # RSI normalizado
        0.2 * (df['Stoch_k'] / 100) +  # Estocástico normalizado
        0.2 * np.clip(df['MACD_diff'], -1, 1) +  # MACD diferencia normalizado
        0.3 * df['CMF']  # Chaikin Money Flow ya está normalizado
    )

    # Fuerza de la tendencia
    trend_strength = (
        df['ADX_14'] / 100 +  # ADX normalizado
        df['BB_Width'] / df['BB_Width'].rolling(window=20).mean()  # Ancho relativo de Bollinger
    ) / 2

    # Posición del precio
    price_position = (
        # Posición respecto a las medias móviles
        0.4 * np.sign(df['close'] - df['EMA_8']) + 
        0.3 * np.sign(df['close'] - df['EMA_21']) +
        0.3 * np.sign(df['close'] - df['SMA_34'])
    )

    # Combinar componentes
    raw_trend = (
        0.4 * trend_direction +
        0.3 * np.sign(price_position) * trend_strength +
        0.3 * (df['close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])  # Posición en Bollinger
    )

    # Normalización final usando sigmoid suavizada
    df['TrendStrength_norm'] = 1 / (1 + np.exp(-4 * raw_trend))  # Factor 4 para mejor distribución

    # Suavizado para reducir ruido
    window_size = 3
    df['TrendStrength_norm'] = df['TrendStrength_norm'].rolling(window=window_size, center=True).mean()
    df['TrendStrength_norm'] = df['TrendStrength_norm'].fillna(method='ffill').fillna(method='bfill')

    # Validación final
    if not (0 <= df['TrendStrength_norm'].min() <= df['TrendStrength_norm'].max() <= 1):
        print("Advertencia: Valores fuera del rango [0,1] detectados")
        df['TrendStrength_norm'] = np.clip(df['TrendStrength_norm'], 0, 1)

    return df


def calculate_trend_indicator_bak(df):
    """
    Calcula un indicador de tendencia compuesto.
    """
    ema_fast = df['EMA_12']
    ema_medium = df['EMA_26']
    ema_slow = df['SMA_50']
    ema_diff_fast = (ema_fast - ema_medium) / ema_medium
    ema_diff_slow = (ema_medium - ema_slow) / ema_slow
    ema_component = 0.5 * ema_diff_fast + 0.3 * ema_diff_slow

    macd_component = np.tanh(df['MACD_diff'])
    rsi_component = (df['RSI_14'] - 50) / 50
    obv_change = df['OBV'].diff()
    obv_component = np.tanh(obv_change)
    adx_component = df['ADX_14'] / 100.0

    trend_strength = (
        0.3 * ema_component +
        0.2 * macd_component +
        0.2 * rsi_component +
        0.15 * obv_component +
        0.15 * adx_component
    )
    trend_indicator = 1 / (1 + np.exp(-trend_strength))
    df['TrendStrength'] = trend_indicator
    return df

def normalize_indicators(df, exclude_cols=('date', 'TrendStrength_norm')):
    """Normaliza todas las columnas numéricas, excepto las excluidas."""
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64'] and col not in exclude_cols:
            df[f"{col}_norm"] = normalize_series(df[col])
    return df

def preprocess_data(symbol, df, date_col='date', ohlc_cols=('open', 'high', 'low', 'close'), seq_length=60):
    """Función principal de preprocesamiento."""

    # 1. Manejo de Fechas
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        raise ValueError(f"Columna de fecha '{date_col}' no existe.")

    # 2. Normalizar OHLC *antes* de los indicadores
    # for col in ohlc_cols:
    #     if col in df.columns:
    #         df[f"{col}_norm"] = normalize_series(df[col])
    #     else:
    #         raise ValueError(f"Falta columna {col}.")

    # 3. Calcular Indicadores (usa columnas normalizadas si existen)
    df = calculate_technical_indicators(df)  # Sin prefijo
    df = calculate_trend_indicator(df)

    # 4. Normalizar *Todos* los Indicadores
    df = normalize_indicators(df)

    # 5. Eliminar NaN
    df.dropna(inplace=True)

    """Guarda el DataFrame completamente preprocesado en un archivo CSV."""
    data_dir="data"
    output_dir = os.path.join(data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{symbol}_processed.csv")
    df.to_csv(output_file, index=False)  # Guarda sin el índice
    print(f"Datos completamente preprocesados guardados para {symbol} en {output_file}")

    # 6. Crear Secuencias - Train/Test Split *antes*
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]

    feature_cols = [col for col in df.columns if col.endswith('_norm')]
    if "close_norm" not in feature_cols:
        feature_cols.append("close_norm")

    x_train, y_train = create_sequences(train_df[feature_cols].values, train_df['close_norm'].values, seq_length)
    x_test, y_test = create_sequences(test_df[feature_cols].values, test_df['close_norm'].values, seq_length)

    scaler = None  # Escalar en model.py
    return x_train, y_train, x_test, y_test, scaler, train_df, test_df, feature_cols

def create_sequences(data, target, seq_length):
    """Crea secuencias para el LSTM."""
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def load_and_preprocess(symbol, data_dir="data", seq_length=60):
    """Carga y preprocesa datos."""
    input_file = os.path.join(data_dir, "raw", f"{symbol}.csv")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"No se encontró: {input_file}")
    df = pd.read_csv(input_file)
    return preprocess_data(symbol, df, seq_length=seq_length)

def save_train_test_data(train_df, test_df, symbol, data_dir="data"):
    """Guarda train/test CSVs (opcional)."""
    output_dir = os.path.join(data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, f"{symbol}_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{symbol}_test.csv"), index=False)
    print(f"Datos train/test guardados para {symbol} en {output_dir}")

if __name__ == '__main__':
    symbol = "AAPL"
    try:
        x_train, y_train, x_test, y_test, scaler, train_df, test_df, feature_cols = load_and_preprocess(symbol)
        print(f"Datos para {symbol} preprocesados.")
        print("x_train shape:", x_train.shape)
        # save_train_test_data(train_df, test_df, symbol)  # Opcional

    except (FileNotFoundError, ValueError) as e:
        print(e)