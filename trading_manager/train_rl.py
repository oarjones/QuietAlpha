import logging
import pandas as pd
import numpy as np
import os
import shutil
from trading_manager.rl_trading_manager import RLTradingManager
from ibkr_api.interface import IBKRInterface

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Inicializar IBKR
ibkr = IBKRInterface(host='127.0.0.1', port=4002)
ibkr.connect()

# Limpiar modelos anteriores para evitar errores de arquitectura
save_path = 'models/rl/trading_manager_new'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

# Configuración mejorada
rl_config = {
    'rl': {
        'num_episodes': 100,         # Más episodios para mejor aprendizaje
        'max_steps_per_episode': 500,
        'initial_balance': 10000.0,
        'max_position': 200,          # Mayor tamaño de posición
        'transaction_fee': 0.0001,    # Reducir aún más la comisión para facilitar el aprendizaje inicial
        'use_lstm_predictions': True,
        'min_confidence': 0.4,        # Reducir el umbral para permitir más operaciones
        'entropy_coef_initial': 0.2,  # Alta exploración inicial
        'entropy_coef_final': 0.02,   # Mantener algo de exploración
        'actor_lr': 0.0002,           # Tasas de aprendizaje más bajas
        'critic_lr': 0.0005
    }
}

# Crear gestor con configuración mejorada
rl_manager = RLTradingManager(ibkr_interface=ibkr)
rl_manager.rl_config = rl_config

# Fase 1: Entrenamiento inicial en un solo símbolo (mercado alcista)
# Para establecer una base sólida
primary_symbol = 'AAPL'  # Apple tiende a ser menos volátil, buen punto de partida
logging.info(f"Fase 1: Entrenamiento inicial en {primary_symbol}")

data = ibkr.get_historical_data(
    symbol=primary_symbol,
    duration="6 M",
    bar_size="1 hour",
    what_to_show="TRADES"
)

if not data.empty:
    # Preprocesar datos para asegurar que tengan todos los indicadores necesarios
    from data.processing import calculate_technical_indicators, calculate_trend_indicator, normalize_indicators
    data = calculate_technical_indicators(data, include_all=True)
    data = calculate_trend_indicator(data)
    data = normalize_indicators(data)
    
    # Entrenar por más episodios en esta fase inicial
    result = rl_manager.train_model(
        symbol=primary_symbol,
        data=data,
        epochs=150,  # Más episodios para la fase inicial
        save_path=save_path
    )
    
    logging.info(f"Fase 1 completada: {result}")
    
    # Evaluar después del entrenamiento
    eval_result = rl_manager.evaluate_model(
        symbol=primary_symbol,
        data=data.iloc[-200:],  # Últimos ~40 días
        episodes=10
    )
    
    logging.info(f"Evaluación Fase 1: {eval_result}")

# Fase 2: Refinamiento con múltiples símbolos
symbols = ['MSFT', 'GOOGL', 'AMZN', 'TSLA']
logging.info("Fase 2: Refinamiento con múltiples símbolos")

for symbol in symbols:
    logging.info(f"Entrenando en {symbol}...")
    data = ibkr.get_historical_data(
        symbol=symbol,
        duration="3 M",  # Periodo más corto para refinamiento
        bar_size="1 hour",
        what_to_show="TRADES"
    )
    
    if not data.empty:
        # Preprocesar datos
        data = calculate_technical_indicators(data, include_all=True)
        data = calculate_trend_indicator(data)
        data = normalize_indicators(data)
        
        # Entrenar con menos episodios para refinamiento
        result = rl_manager.train_model(
            symbol=symbol,
            data=data,
            epochs=50,  # Menos episodios para refinamiento
            save_path=save_path
        )
        
        logging.info(f"Entrenamiento en {symbol} completado: {result}")

# Fase 3: Evaluación final en todos los símbolos
logging.info("Fase 3: Evaluación final")
all_symbols = [primary_symbol] + symbols

for symbol in all_symbols:
    data = ibkr.get_historical_data(
        symbol=symbol,
        duration="1 M",  # Datos recientes para evaluación
        bar_size="1 hour",
        what_to_show="TRADES"
    )
    
    if not data.empty:
        # Preprocesar datos
        data = calculate_technical_indicators(data, include_all=True)
        data = calculate_trend_indicator(data)
        data = normalize_indicators(data)
        
        # Evaluar
        eval_result = rl_manager.evaluate_model(
            symbol=symbol,
            data=data,
            episodes=10
        )
        
        logging.info(f"Evaluación final para {symbol}: {eval_result}")

# Desconectar IBKR
ibkr.disconnect()
logging.info("Proceso de entrenamiento completado.")