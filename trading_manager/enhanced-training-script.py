#!/usr/bin/env python
"""
Enhanced RL Trading Training Script

This script implements the enhanced training process for RL trading models with:
1. Improved hyperparameter optimization
2. Enhanced reward function
3. Better neural network architecture
4. Comprehensive evaluation metrics

Usage:
    python enhanced-training-script.py --symbol MSFT --n-trials 30 --enable-optuna
"""

import csv
import os
import argparse
import logging
import pandas as pd
import numpy as np
import time
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from stable_baselines3 import PPO
from tqdm import tqdm

# Import project modules
from ibkr_api.interface import IBKRInterface
from data.processing import calculate_technical_indicators, normalize_indicators, calculate_trend_indicator
from utils.data_utils import load_config, save_config, merge_configs
from trading_manager.rl_trading_stable import RLTradingStableManager, EnhancedTradingEnv
from trading_manager.integration_helpers import (
    make_enhanced_env, 
    enhance_trading_manager, 
    create_enhanced_config,
    update_config_file,
    run_enhanced_optimization
)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from integration_helpers import DetailedTrialEvalCallback, IncrementalTrialCallback, DetailedEvalCallback, TrainingProgressCallback


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'enhanced_training.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Configure progress printing
def print_progress(message, end='\n'):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", end=end)
    sys.stdout.flush()


def fetch_and_process_data(ibkr: IBKRInterface, symbol: str, lookback_days: int = 730) -> pd.DataFrame:
    """
    Fetch and process data for a symbol.
    
    Args:
        ibkr: IBKR interface
        symbol: Symbol to fetch data for
        lookback_days: Number of days to look back
        
    Returns:
        pd.DataFrame: Processed data
    """
    try:
        print_progress(f"Fetching data for {symbol}, lookback: {lookback_days} days...")
        
        # Determine duration string based on lookback
        if lookback_days <= 30:
            duration = f"{lookback_days} D"
        elif lookback_days <= 180:
            duration = f"{lookback_days // 30 + 1} M"
        else:
            duration = f"{lookback_days // 365 + 1} Y"
        
        # Fetch hourly data
        df = ibkr.get_historical_data(
            symbol=symbol,
            duration=duration,
            bar_size="1 hour",
            what_to_show="TRADES",
            use_rth=True
        )
        
        if df.empty:
            print_progress(f"⚠️ No data returned for {symbol}")
            return pd.DataFrame()
        
        # Process data
        print_progress(f"Processing data for {symbol} ({len(df)} rows)...")
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df, include_all=True)
        
        # Calculate trend indicator
        df = calculate_trend_indicator(df)
        
        # Normalize indicators
        df = normalize_indicators(df)
        
        # Remove the first 22 rows due to incorrect values from indicator windows
        df = df.iloc[22:]
        
        # Remove NaN values
        df.dropna(inplace=True)

        # Save processed data for reference
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv(f'data/processed/{symbol}_processed.csv', index=False)
        
        print_progress(f"✅ Data processed for {symbol}: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        print_progress(f"❌ Error fetching and processing data for {symbol}: {e}")
        logger.error(f"Error fetching and processing data for {symbol}: {e}")
        return pd.DataFrame()

def load_data_from_file(filepath: str) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        filepath: Path to data file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        print_progress(f"Loading data from {filepath}...")
        
        # Determine file type from extension
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.pkl', '.pickle']:
            df = pd.read_pickle(filepath)
        else:
            print_progress(f"❌ Unsupported file type: {ext}")
            return pd.DataFrame()
        
        # Convert datetime column if exists
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        print_progress(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    except Exception as e:
        print_progress(f"❌ Error loading data from file: {e}")
        logger.error(f"Error loading data from file: {e}")
        return pd.DataFrame()

class ProgressCallback:
    """Callback to track and display optimization progress."""
    
    def __init__(self, n_trials, update_interval=60):
        self.n_trials = n_trials
        self.completed_trials = 0
        self.best_value = float('-inf')
        self.best_params = None
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.update_interval = update_interval  # seconds
        
        # Print header
        print("\n" + "=" * 80)
        print(f"OPTUNA OPTIMIZATION PROGRESS ({n_trials} trials)")
        print("=" * 80)
        print(f"{'Trial #':<8} {'Value':<10} {'Win Rate':<10} {'Profit %':<10} {'Trades':<8} {'Elapsed':<10} {'ETA':<10}")
        print("-" * 80)
    
    def __call__(self, study, trial):
        self.completed_trials += 1
        current_time = datetime.now()
        elapsed = current_time - self.start_time
        
        # Update best value
        if trial.value is not None and trial.value > self.best_value:
            self.best_value = trial.value
            self.best_params = trial.params
            
            # Get trial attributes
            win_rate = trial.user_attrs.get('win_rate', 0) * 100
            portfolio_change = trial.user_attrs.get('portfolio_change', 0) * 100
            total_trades = trial.user_attrs.get('total_trades', 0)
            
            # Print progress immediately for new best value
            time_per_trial = elapsed / self.completed_trials
            eta = time_per_trial * (self.n_trials - self.completed_trials)
            
            elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
            eta_str = str(timedelta(seconds=int(eta.total_seconds())))
            
            print(f"{self.completed_trials}/{self.n_trials:<3} {trial.value:<10.4f} {win_rate:<10.2f} {portfolio_change:<10.2f} {total_trades:<8} {elapsed_str:<10} {eta_str:<10} NEW BEST! ⭐")
            self.last_update = current_time
        
        # Regular updates based on interval
        elif (current_time - self.last_update).total_seconds() >= self.update_interval:
            # Get trial attributes
            win_rate = trial.user_attrs.get('win_rate', 0) * 100
            portfolio_change = trial.user_attrs.get('portfolio_change', 0) * 100
            total_trades = trial.user_attrs.get('total_trades', 0)
            
            # Calculate ETA
            time_per_trial = elapsed / self.completed_trials
            eta = time_per_trial * (self.n_trials - self.completed_trials)
            
            elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
            eta_str = str(timedelta(seconds=int(eta.total_seconds())))
            
            print(f"{self.completed_trials}/{self.n_trials:<3} {trial.value:<10.4f} {win_rate:<10.2f} {portfolio_change:<10.2f} {total_trades:<8} {elapsed_str:<10} {eta_str:<10}")
            self.last_update = current_time

    def finalize(self):
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))
        
        print("-" * 80)
        print(f"✅ Optimization completed: {self.completed_trials}/{self.n_trials} trials in {elapsed_str}")
        print(f"Best value: {self.best_value:.4f}")
        print("=" * 80 + "\n")


"""
Integración de Enhanced Pruning en el flujo de optimización con Optuna

Este código demuestra cómo integrar la estrategia de pruning mejorada en el
flujo de optimización existente en el script enhanced-training-script.py.
"""

import os
import logging
import numpy as np
import time
from typing import Dict
import optuna

from integration_helpers import DetailedTrialEvalCallback, create_pruning_config

# Configuramos el logging
logger = logging.getLogger(__name__)

def custom_run_enhanced_optimization(
    data, 
    config, 
    output_dir="models/rl/enhanced",
    n_trials=20,
    n_timesteps=100000,
    final_timesteps=200000,
    n_jobs=1,
    pruning_config=None  # Nueva configuración de pruning
):
    """
    Versión mejorada de la optimización con Optuna que incorpora
    estrategias avanzadas de pruning basadas en métricas financieras.
    
    Args:
        data: Datos históricos para el entrenamiento
        config: Configuración del entorno
        output_dir: Directorio de salida
        n_trials: Número de trials para Optuna
        n_timesteps: Pasos para cada trial
        final_timesteps: Pasos para el modelo final
        n_jobs: Número de jobs paralelos
        pruning_config: Configuración del pruning mejorado (opcional)
        
    Returns:
        Dict: Resultados de la optimización
    """
    # Importamos aquí para evitar problemas de dependencias cíclicas
    from stable_baselines3 import PPO
    import torch.nn as nn
    from trading_manager.integration_helpers import (
        make_enhanced_env, 
        IncrementalTrialCallback,
        TrainingProgressCallback,
        create_pruning_config
    )
    
    # Crear directorios para resultados
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Si no se proporciona configuración de pruning, usar valores predeterminados adaptados
    if pruning_config is None:
        pruning_config = create_pruning_config(
            # Configuración general más adecuada para trading
            min_eval_count=2,                     # Empezar a evaluar antes
            min_trades_for_eval=8,                # Ser más flexible con el número de trades
            
            # Umbrales adaptados para trading con RL
            max_drawdown_threshold=0.35,          # 35% máximo drawdown
            negative_portfolio_threshold=-0.10,   # -10% portfolio change
            low_win_rate_threshold=0.30,          # 30% win rate mínimo 
            
            # Permitir win rate bajo si hay buenas ganancias
            allow_low_win_rate_if_profit=True,
            min_profit_for_low_win_rate=0.08,     # 8% de ganancia compensa win rate bajo
            
            # Pruning progresivo después de más trials
            enable_progressive_pruning=True,
            progressive_pruning_start=8           # Después de 8 evaluaciones
        )
    
    # Callback para mostrar progreso de la optimización
    progress_callback = IncrementalTrialCallback(n_trials=n_trials)
    

    def make_json_serializable(obj):
        """Convierte un objeto en un formato serializable a JSON."""
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: make_json_serializable(value) for key, value in obj.items()}
        else:
            # Para otros tipos de objetos, convertir a string
            return str(obj)

    # Definir la función objetivo con pruning mejorado
    def objective(trial):
        """Función objetivo con pruning mejorado para Optuna."""
        # Muestrear hiperparámetros (igual que en la versión original)
        env_params = {
            'reward_strategy': trial.suggest_categorical(
                'reward_strategy', 
                ['balanced', 'sharpe']
            ),
            'risk_aversion': trial.suggest_float('risk_aversion', 0.5, 2.0),
            'reward_scaling': trial.suggest_float('reward_scaling', 0.5, 2.0),
            'drawdown_penalty_factor': trial.suggest_float('drawdown_penalty_factor', 5.0, 25.0),
            'holding_penalty_factor': trial.suggest_float('holding_penalty_factor', 0.05, 0.2),
            'inactive_penalty_factor': trial.suggest_float('inactive_penalty_factor', 0.01, 0.1),
            'consistency_reward_factor': trial.suggest_float('consistency_reward_factor', 0.1, 0.4),
            'trend_following_factor': trial.suggest_float('trend_following_factor', 0.1, 0.5),
            'win_streak_factor': trial.suggest_float('win_streak_factor', 0.05, 0.2)
        }
        
        # PPO hyperparámetros
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        n_epochs = trial.suggest_int("n_epochs", 5, 20)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.999)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.4, 1.0)
        
        # Arquitectura de red
        net_width = trial.suggest_categorical("net_width", [64, 128, 256, 512])
        net_depth = trial.suggest_int("net_depth", 1, 4)
        net_arch = [net_width for _ in range(net_depth)]
        
        # Función de activación
        activation_fn_name = trial.suggest_categorical(
            "activation_fn", ["tanh", "relu", "elu"]
        )
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU
        }[activation_fn_name]
        
        # Actualizar configuración del entorno con parámetros muestreados
        trial_config = config.copy()
        trial_config.update(env_params)
        
        # Crear entornos de entrenamiento y evaluación
        env = make_enhanced_env(data, trial_config, False)
        eval_env = make_enhanced_env(data, trial_config, True)
        
        # Crear kwargs de política
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fn
        }
        
        # Parámetros PPO
        ppo_params = {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "policy_kwargs": policy_kwargs
        }
        
        
        # Código para guardar el trial
        trial_config_path = os.path.join(output_dir, "trial_configs")
        os.makedirs(trial_config_path, exist_ok=True)
        trial_config_file = os.path.join(trial_config_path, 'trials_config.json')

        # Crear información del trial asegurando que sea serializable
        trial_info = {
            'trial_number': trial.number,
            'policy_kwargs': {
                'net_arch': net_arch,
                'activation_fn': activation_fn_name
            },
            'ppo_params': {
                'learning_rate': ppo_params['learning_rate'],
                'n_steps': ppo_params['n_steps'],
                'batch_size': ppo_params['batch_size'],
                'n_epochs': ppo_params['n_epochs'],
                'gamma': ppo_params['gamma'],
                'gae_lambda': ppo_params['gae_lambda'],
                'clip_range': ppo_params['clip_range'],
                'ent_coef': ppo_params['ent_coef'],
                'vf_coef': ppo_params['vf_coef']
                # Omitimos policy_kwargs para evitar referencias circulares
            },
            'env_params': env_params
        }

        # Convertir toda la estructura a formato serializable
        serializable_trial_info = make_json_serializable(trial_info)

        # Guardar configuración del trial
        import json
        existing_configs = []
        if os.path.exists(trial_config_file):
            try:
                with open(trial_config_file, 'r') as f:
                    existing_configs = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error leyendo archivo de configuraciones: {e}")
                existing_configs = []

        # Asegurarnos de que existing_configs sea una lista
        if not isinstance(existing_configs, list):
            existing_configs = []

        existing_configs.append(serializable_trial_info)

        try:
            with open(trial_config_file, 'w') as f:
                json.dump(existing_configs, f, indent=4)
            print(f"Configuración del trial {trial.number} guardada correctamente")
        except Exception as e:
            print(f"Error guardando configuración del trial: {e}")


        
        # Crear modelo
        model = PPO("MlpPolicy", env, verbose=0, **ppo_params)
        
        # Usar el callback mejorado con enhanced pruning
        eval_callback = DetailedTrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            pruning_config=pruning_config,  # Usar la configuración de pruning mejorada
            n_eval_episodes=5,
            eval_freq=5000,
            log_path=os.path.join(output_dir, "logs", f"trial_{trial.number}"),
            best_model_save_path=os.path.join(output_dir, "models", f"trial_{trial.number}"),
            deterministic=True,
            verbose=1
        )
        
        try:
            # Entrenar el modelo
            model.learn(total_timesteps=n_timesteps, callback=eval_callback)
            
            # Si el trial fue podado, lanzar excepción
            if eval_callback.is_pruned:
                raise optuna.exceptions.TrialPruned()
            
            # Ejecutar un episodio final completo para obtener métricas
            final_metrics = eval_env.run_full_episode(model, deterministic=True)
            
            # Establecer atributos finales
            for key, value in final_metrics.items():
                if key != 'total_reward':  # Ya reportado como reward
                    trial.set_user_attr(key, value)
            
            # Devolver la mejor recompensa como valor del objetivo
            return eval_callback.best_mean_reward
            
        except (optuna.exceptions.TrialPruned, Exception) as e:
            # Manejar excepciones
            if isinstance(e, optuna.exceptions.TrialPruned):
                logger.info(f"Trial {trial.number} pruned.")
                raise e
            else:
                logger.error(f"Error en trial {trial.number}: {e}")
                return float('-inf')
    
    # Crear estudio Optuna con muestreo y poda mejorados
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=config.get('seed', 42))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    study = optuna.create_study(
        study_name=f"enhanced_trading_{time.strftime('%Y%m%d_%H%M%S')}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    try:
        # Ejecutar optimización con callbacks mejorados
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            callbacks=[progress_callback],
            show_progress_bar=True
        )
        
        # Finalizar reporte de progreso
        progress_callback.finalize()
        
        # Obtener mejor trial y parámetros
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Convertir parámetros para formato esperado
        activation_fn_name = best_params.pop("activation_fn")
        net_width = best_params.pop("net_width")
        net_depth = best_params.pop("net_depth")
        net_arch = [net_width for _ in range(net_depth)]
        
        # Separar parámetros de entorno
        env_params = {
            'reward_strategy': best_params.pop('reward_strategy'),
            'risk_aversion': best_params.pop('risk_aversion'),
            'reward_scaling': best_params.pop('reward_scaling'),
            'drawdown_penalty_factor': best_params.pop('drawdown_penalty_factor'),
            'holding_penalty_factor': best_params.pop('holding_penalty_factor'),
            'inactive_penalty_factor': best_params.pop('inactive_penalty_factor'),
            'consistency_reward_factor': best_params.pop('consistency_reward_factor'),
            'trend_following_factor': best_params.pop('trend_following_factor'),
            'win_streak_factor': best_params.pop('win_streak_factor')
        }
        
        # Crear policy kwargs
        policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fn_name
        }
        
        # Empaquetar parámetros restantes como ppo_params
        ppo_params = {
            key: best_params[key] for key in [
                "learning_rate", "n_steps", "batch_size", "n_epochs",
                "gamma", "gae_lambda", "clip_range", "ent_coef", "vf_coef"
            ]
        }
        
        # Organizar resultados en formato esperado
        best_params_formatted = {
            'env_params': env_params,
            'policy_kwargs': policy_kwargs,
            'ppo_params': ppo_params
        }
        
        # Entrenar modelo final con mejores parámetros
        logger.info(f"\nEntrenando modelo final con mejores parámetros...")
        
        # Crear entorno con mejores parámetros
        final_config = config.copy()
        final_config.update(env_params)
        
        final_env = make_enhanced_env(data, final_config)
        final_eval_env = make_enhanced_env(data, final_config)
        
        # Importar función de activación correctamente
        activation_fn = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU
        }[activation_fn_name]
        
        # Crear modelo final
        final_policy_kwargs = {
            "net_arch": net_arch,
            "activation_fn": activation_fn
        }
        
        final_model = PPO(
            "MlpPolicy",
            final_env,
            verbose=1,
            policy_kwargs=final_policy_kwargs,
            **ppo_params
        )
        
        # Callback para evaluación durante entrenamiento final
        from trading_manager.integration_helpers import DetailedEvalCallback
        
        final_eval_callback = DetailedEvalCallback(
            eval_env=final_eval_env,
            best_model_save_path=os.path.join(output_dir, "final_model"),
            log_path=os.path.join(output_dir, "logs"),
            eval_freq=10000,
            deterministic=True,
            verbose=1
        )
        
        final_progress_callback = TrainingProgressCallback(
            total_timesteps=final_timesteps,
            update_interval=5000,
            verbose=1
        )
        
        # Entrenar modelo final
        start_time = time.time()
        final_model.learn(
            total_timesteps=final_timesteps,
            callback=[final_eval_callback, final_progress_callback]
        )
        training_time = time.time() - start_time
        
        # Guardar modelo final
        final_model_path = os.path.join(output_dir, "final_model", "model")
        final_model.save(final_model_path)
        
        # Evaluar modelo final
        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(
            final_model, final_eval_env, n_eval_episodes=10
        )
        
        # Obtener métricas del entorno
        final_metrics = final_eval_env.run_full_episode(final_model, deterministic=True)
        
        # Extraer métricas 
        win_rate = final_metrics.get('win_rate', 0)
        portfolio_change = final_metrics.get('portfolio_change', 0)
        max_drawdown = final_metrics.get('max_drawdown', 0)
        total_trades = final_metrics.get('total_trades', 0)
        sharpe_ratio = final_metrics.get('sharpe_ratio', 0)
        
        # Crear diccionario de resultados
        final_results = {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'win_rate': float(win_rate),
            'portfolio_change': float(portfolio_change),
            'max_drawdown': float(max_drawdown),
            'total_trades': int(total_trades),
            'sharpe_ratio': float(sharpe_ratio),
            'training_time': float(training_time),
            'training_time_formatted': time.strftime("%H:%M:%S", time.gmtime(training_time))
        }
        
        # Guardar resultados en archivo
        import json
        with open(os.path.join(output_dir, "results", "final_results.json"), "w") as f:
            json.dump(final_results, f, indent=4)
        
        # # Análisis adicional de los trials
        # trial_analysis = analyze_trials(study)
        
        # # Guardar análisis
        # with open(os.path.join(output_dir, "results", "trial_analysis.json"), "w") as f:
        #     json.dump(trial_analysis, f, indent=4)
        
        # Devolver resultados
        return {
            'status': 'success',
            'best_params': best_params_formatted,
            'final_results': final_results,
            'study': study
        }
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Optimización interrumpida por el usuario")
        progress_callback.finalize()
        
        # Recopilar resultados parciales
        if hasattr(study, 'best_trial'):
            logger.info(f"Mejor trial hasta ahora: #{study.best_trial.number}, valor: {study.best_value:.4f}")
        
        return {'status': 'interrupted'}



# def analyze_trials(study):
#     """
#     Analiza los resultados de los trials para obtener insights sobre el proceso de optimización.
    
#     Args:
#         study: Estudio de Optuna completado
        
#     Returns:
#         dict: Análisis de los trials
#     """
#     # Recopilar estadísticas de pruning
#     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
#     completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
#     # Recopilar razones de pruning
#     pruning_reasons = {}
#     for trial in pruned_trials:
#         reason = trial.user_attrs.get('prune_reason', 'Unknown')
#         if reason in pruning_reasons:
#             pruning_reasons[reason] += 1
#         else:
#             pruning_reasons[reason] = 1
    
#     # Calcular estadísticas de rendimiento
#     perf_metrics = {
#         'win_rate': [],
#         'portfolio_change': [],
#         'max_drawdown': [],
#         'total_trades': [],
#         'sharpe_ratio': []
#     }
    
#     for trial in completed_trials:
#         if 'win_rate' in trial.user_attrs:
#             perf_metrics['win_rate'].append(trial.user_attrs['win_rate'])
#         if 'portfolio_change' in trial.user_attrs:
#             perf_metrics['portfolio_change'].append(trial.user_attrs['portfolio_change'])
#         if 'max_drawdown' in trial.user_attrs:
#             perf_metrics['max_drawdown'].append(trial.user_attrs['max_drawdown'])
#         if 'total_trades' in trial.user_attrs:
#             perf_metrics['total_trades'].append(trial.user_attrs['total_trades'])
#         if 'sharpe_ratio' in trial.user_attrs:
#             perf_metrics['sharpe_ratio'].append(trial.user_attrs['sharpe_ratio'])
    
#     # Calcular estadísticas
#     stats = {}
#     for metric, values in perf_metrics.items():
#         if values:
#             stats[metric] = {
#                 'mean': float(np.mean(values)),
#                 'median': float(np.median(values)),
#                 'min': float(np.min(values)),
#                 'max': float(np.max(values)),
#                 'std': float(np.std(values))
#             }
    
#     # Análisis de los mejores trials
#     best_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:5]
#     best_trials_info = []
    
#     for i, trial in enumerate(best_trials):
#         best_trials_info.append({
#             'rank': i + 1,
#             'trial_number': trial.number,
#             'value': float(trial.value) if trial.value is not None else None,
#             'win_rate': float(trial.user_attrs.get('win_rate', 0)),
#             'portfolio_change': float(trial.user_attrs.get('portfolio_change', 0)),
#             'total_trades': int(trial.user_attrs.get('total_trades', 0)),
#             'params': trial.params
#         })
    
#     # Análisis de terminación temprana (aceleración)
#     steps_saved = 0
#     for trial in pruned_trials:
#         # Estimar pasos ahorrados (simple estimación)
#         if hasattr(trial, 'last_step'):
#             steps_saved += n_timesteps - trial.last_step
#         else:
#             # Estimación conservadora si no sabemos el paso exacto
#             steps_saved += n_timesteps // 2
    
#     return {
#         'total_trials': len(study.trials),
#         'completed_trials': len(completed_trials),
#         'pruned_trials': len(pruned_trials),
#         'pruning_reasons': pruning_reasons,
#         'completion_rate': len(completed_trials) / max(1, len(study.trials)),
#         'performance_stats': stats,
#         'best_trials': best_trials_info,
#         'estimated_steps_saved': steps_saved,
#         'pruning_efficiency': steps_saved / max(1, n_timesteps * len(pruned_trials)) if pruned_trials else 0
#     }


def simple_rl_training(
    data: pd.DataFrame, 
    config: dict, 
    symbol: str, 
    timesteps: int = 200000,
    output_dir: str = "models/rl/enhanced"
) -> dict:
    """
    Train a simple RL model without optimization, using improved callbacks
    
    Args:
        data: Historical price data
        config: Environment configuration
        symbol: Symbol being trained
        timesteps: Number of training timesteps
        output_dir: Output directory
        
    Returns:
        dict: Training results
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from trading_manager.integration_helpers import DetailedEvalCallback, TrainingProgressCallback
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print_progress(f"Setting up environments for simple training...")
    
    # Create environments
    env = make_enhanced_env(data, config)
    eval_env = make_enhanced_env(data, config)
    
    # Create model
    policy_kwargs = {
        'net_arch': [256, 128, 64]  # Default architecture
    }
    
    # Check for custom architecture in config
    if 'net_arch' in config:
        if isinstance(config['net_arch'], list):
            policy_kwargs['net_arch'] = config['net_arch']
        elif config['net_arch'] == 'large':
            policy_kwargs['net_arch'] = [512, 256, 128]
        elif config['net_arch'] == 'medium':
            policy_kwargs['net_arch'] = [256, 128, 64]
        elif config['net_arch'] == 'small':
            policy_kwargs['net_arch'] = [128, 64, 32]
    
    print_progress(f"Creating PPO model with architecture: {policy_kwargs['net_arch']}...")
    
    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log=os.path.join(output_dir, "logs"),
        policy_kwargs=policy_kwargs,
        learning_rate=config.get('learning_rate', 3e-4),
        n_steps=config.get('n_steps', 2048),
        batch_size=config.get('batch_size', 64),
        gamma=config.get('gamma', 0.99),
        ent_coef=config.get('ent_coef', 0.01)
    )
    
    # Crear DetailedEvalCallback en lugar del EvalCallback estándar
    eval_callback = DetailedEvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=10000,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "logs"),
        deterministic=True,
        verbose=1
    )
    
    # Crear TrainingProgressCallback en lugar del ProgressTracker personalizado
    progress_callback = TrainingProgressCallback(
        total_timesteps=timesteps,
        update_interval=5000,
        verbose=1
    )
    
    # # Create a simple tracker for printing progress
    # class ProgressTracker:
    #     def __init__(self, total_timesteps, update_interval=30):
    #         self.total_timesteps = total_timesteps
    #         self.start_time = time.time()
    #         self.last_update = time.time()
    #         self.update_interval = update_interval
    #         print("\nTRAINING PROGRESS")
    #         print("-" * 50)
    #         print(f"{'Timesteps':<15} {'Progress':<10} {'Elapsed':<15} {'ETA':<15}")
    #         print("-" * 50)
        
    #     def __call__(self, locals, globals):
    #         current_time = time.time()
    #         timesteps_so_far = locals['self'].num_timesteps
            
    #         # Update at regular intervals
    #         if current_time - self.last_update > self.update_interval:
    #             # Calculate progress
    #             progress = timesteps_so_far / self.total_timesteps * 100
                
    #             # Calculate time metrics
    #             elapsed = current_time - self.start_time
    #             elapsed_str = str(timedelta(seconds=int(elapsed)))
                
    #             # Calculate ETA
    #             if timesteps_so_far > 0:
    #                 time_per_step = elapsed / timesteps_so_far
    #                 remaining_steps = self.total_timesteps - timesteps_so_far
    #                 eta = time_per_step * remaining_steps
    #                 eta_str = str(timedelta(seconds=int(eta)))
    #             else:
    #                 eta_str = "Unknown"
                
    #             # Print progress
    #             print(f"{timesteps_so_far}/{self.total_timesteps:<8} {progress:<9.1f}% {elapsed_str:<15} {eta_str:<15}")
                
    #             self.last_update = current_time
                
    #         return True
    
    # # Create progress tracker
    # progress_tracker = ProgressTracker(timesteps)
    
    # Train model
    print_progress(f"Starting training for {timesteps} timesteps...")
    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=[eval_callback, progress_callback])
    training_time = time.time() - start_time
    
    # Save final model
    model_save_path = os.path.join(output_dir, f"final_model_{symbol}")
    model.save(model_save_path)
    
    print_progress(f"Evaluating model performance...")
    
    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    # Get environment metrics
    env_unwrapped = env.envs[0].unwrapped
    info = env_unwrapped._get_info()
    
    # Prepare results
    results = {
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'win_rate': float(info['win_rate']),
        'max_drawdown': float(info['max_drawdown']),
        'portfolio_change': float(info['portfolio_change']),
        'total_trades': int(info['total_trades']),
        'sharpe_ratio': float(info.get('sharpe_ratio', 0)),
        'training_time': float(training_time),
        'training_time_formatted': time.strftime("%H:%M:%S", time.gmtime(training_time)),
        'model_path': model_save_path
    }
    
    # Save results
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    with open(os.path.join(output_dir, "results", f"{symbol}_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    print_progress(f"✅ Training completed for {symbol}: mean_reward={mean_reward:.2f}, win_rate={info['win_rate']:.2f}")
    
    return results

def train_with_trading_manager(
    data: pd.DataFrame, 
    symbol: str, 
    config_path: str,
    timesteps: int = 200000,
    enhanced: bool = True
) -> dict:
    """
    Train a model using the RLTradingStableManager, optionally enhanced.
    
    Args:
        data: Historical price data
        symbol: Symbol being trained
        config_path: Path to configuration file
        timesteps: Number of training timesteps
        enhanced: Whether to use enhanced training
        
    Returns:
        dict: Training results
    """
    # Load configuration
    config = load_config(config_path)
    
    print_progress(f"Creating {'enhanced' if enhanced else 'standard'} trading manager...")
    
    # Create trading manager
    manager = RLTradingStableManager(config_path=config_path)
    
    # Enhance manager if requested
    if enhanced:
        manager = enhance_trading_manager(manager)
    
    # Initialize with data
    print_progress(f"Initializing trading manager with data...")
    success = manager._initialize_with_data(data)
    if not success:
        print_progress(f"❌ Failed to initialize trading manager with data")
        return {'status': 'error', 'message': 'Failed to initialize trading manager'}
    
    # Train model
    print_progress(f"Starting training for {timesteps} timesteps...")
    start_time = time.time()
    training_result = manager.train_model(
        symbol=symbol,
        data=data,
        epochs=timesteps,
        save_path=os.path.join("models/rl", "enhanced" if enhanced else "standard")
    )
    training_time = time.time() - start_time
    
    # Add training time
    training_result['training_time'] = training_time
    training_result['training_time_formatted'] = time.strftime("%H:%M:%S", time.gmtime(training_time))
    
    print_progress(f"✅ Training completed in {training_result['training_time_formatted']}")
    
    return training_result

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Enhanced RL Trading Training')
    parser.add_argument('--symbol', type=str, default='MSFT', help='Symbol to train on')
    parser.add_argument('--config', type=str, default='config/lstm_config.json', help='Path to configuration file')
    parser.add_argument('--data-file', type=str, default=None, help='Path to data file (optional)')
    parser.add_argument('--lookback-days', type=int, default=1095, help='Number of days to look back for data')
    parser.add_argument('--timesteps', type=int, default=200000, help='Number of timesteps for training')
    parser.add_argument('--enable-optuna', action='store_true', default=False, help='Enable Optuna optimization')
    parser.add_argument('--n-trials', type=int, default=35, help='Number of Optuna trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs for Optuna')
    parser.add_argument('--use-manager', action='store_true', default=True, help='Use trading manager for training')
    parser.add_argument('--output-dir', type=str, default='models/rl/enhanced', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/rl/enhanced', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print script configuration
    print("\n" + "=" * 50)
    print(f"ENHANCED RL TRADING TRAINING")
    print("=" * 50)
    print(f"Symbol:          {args.symbol}")
    print(f"Config file:     {args.config}")
    print(f"Data file:       {args.data_file or 'Fetch from IBKR'}")
    print(f"Lookback days:   {args.lookback_days}")
    print(f"Training steps:  {args.timesteps}")
    print(f"Optuna enabled:  {args.enable_optuna}")
    if args.enable_optuna:
        print(f"Optuna trials:   {args.n_trials}")
        print(f"Parallel jobs:   {args.n_jobs}")
    print(f"Use manager:     {args.use_manager}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Random seed:     {args.seed}")
    print("=" * 50 + "\n")
    
    # Load configuration
    print_progress("Loading configuration...")
    config = load_config(args.config)
    
    # Create enhanced configuration
    enhanced_config = create_enhanced_config(config)
    
    # Update configuration file with enhanced settings
    if not os.path.exists(args.config + '.original'):
        # Make backup of original config
        import shutil
        shutil.copy(args.config, args.config + '.original')
        print_progress(f"Created backup of original config at {args.config + '.original'}")
    
    # Update config file with enhanced settings
    update_config_file(args.config, enhanced_config)
    print_progress(f"Updated config file with enhanced settings")
    
    # Load data
    data = None
    if args.data_file:
        # Load from file
        data = load_data_from_file(args.data_file)
    else:
        # Initialize IBKR
        ibkr_config = config.get('ibkr', {})
        ibkr = IBKRInterface(
            host=ibkr_config.get('host', '127.0.0.1'),
            port=ibkr_config.get('port', 4002),
            client_id=ibkr_config.get('client_id', 1)
        )
        
        # Connect to IBKR
        print_progress(f"Connecting to IBKR at {ibkr_config.get('host', '127.0.0.1')}:{ibkr_config.get('port', 4002)}...")
        connected = ibkr.connect()
        if not connected:
            print_progress(f"❌ Failed to connect to IBKR")
            return
        
        print_progress(f"✅ Connected to IBKR")
        
        # Fetch and process data
        data = fetch_and_process_data(ibkr, args.symbol, args.lookback_days)
        
        # Disconnect from IBKR when done
        ibkr.disconnect()
        print_progress(f"Disconnected from IBKR")
    
    if data is None or data.empty:
        print_progress(f"❌ No data available for {args.symbol}")
        return
    
    # Get environment configuration from enhanced config
    rl_config = enhanced_config.get('trading_manager', {}).get('rl', {})
    
    # Create a clean environment config
    env_config = {
        'initial_balance': rl_config.get('initial_balance', 10000.0),
        'max_position': rl_config.get('max_position', 100),
        'transaction_fee': rl_config.get('transaction_fee', 0.001),
        'reward_scaling': rl_config.get('reward_scaling', 1.0),
        'window_size': rl_config.get('window_size', 60),
        'reward_strategy': rl_config.get('reward_strategy', 'balanced'),
        'risk_free_rate': rl_config.get('risk_free_rate', 0.0),
        'risk_aversion': rl_config.get('risk_aversion', 1.0),
        'drawdown_penalty_factor': rl_config.get('drawdown_penalty_factor', 15.0),
        'holding_penalty_factor': rl_config.get('holding_penalty_factor', 0.1),
        'inactive_penalty_factor': rl_config.get('inactive_penalty_factor', 0.05),
        'consistency_reward_factor': rl_config.get('consistency_reward_factor', 0.2),
        'trend_following_factor': rl_config.get('trend_following_factor', 0.3),
        'win_streak_factor': rl_config.get('win_streak_factor', 0.1),
        'seed': args.seed
    }
    
    # Add network architecture
    if 'net_arch' in rl_config:
        env_config['net_arch'] = rl_config['net_arch']
    
    # Add training parameters
    for param in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma', 
                  'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']:
        if param in rl_config:
            env_config[param] = rl_config[param]
    
    # Run training
    start_time = time.time()
    
    if args.enable_optuna:

        
        # Load pruning configuration from file
        pruning_config_path = 'config/pruning_config.json'
        try:
            if os.path.exists(pruning_config_path):
                with open(pruning_config_path, 'r') as f:
                    pruning_config = json.load(f)
                print_progress(f"Loaded pruning config from {pruning_config_path}")
            else:
            # Use default pruning configuration
                pruning_config = {
                    'min_eval_count': 2,
                    'min_trades_for_eval': 8,
                    'max_drawdown_threshold': 0.35,
                    'negative_portfolio_threshold': -0.10,
                    'low_win_rate_threshold': 0.30,
                    'allow_low_win_rate_if_profit': True,
                    'min_profit_for_low_win_rate': 0.08,
                    'enable_progressive_pruning': True,
                    'progressive_pruning_start': 8
                }
            print_progress("Using default pruning configuration")
        except Exception as e:
            logger.warning(f"Error loading pruning config: {e}, using defaults")
            pruning_config = None

        # Run Optuna optimization
        print_progress(f"Starting Optuna optimization with {args.n_trials} trials...")
        result = custom_run_enhanced_optimization(
            data=data,
            config=env_config,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            n_timesteps=args.timesteps // 2,  # Use half timesteps for trials
            final_timesteps=args.timesteps,
            n_jobs=args.n_jobs,
            pruning_config=pruning_config
        )
        
        # Extract and save best parameters to config file
        best_params = result['best_params']
        
        # Update environment parameters in config
        rl_config.update(best_params['env_params'])
        
        # Update PPO parameters in config
        for key, value in best_params['ppo_params'].items():
            rl_config[key] = value
        
        # Update network architecture in config
        rl_config['net_arch'] = best_params['policy_kwargs']['net_arch']
        
        # Save updated config
        update_config_file(args.config, enhanced_config)
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"ENHANCED OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Symbol: {args.symbol}")
        print(f"Total Trials: {args.n_trials}")
        print("\nBest Model Performance:")
        final_results = result['final_results']
        print(f"  Mean Reward:       {final_results['mean_reward']:.4f}")
        print(f"  Win Rate:          {final_results['win_rate'] * 100:.2f}%")
        print(f"  Portfolio Change:  {final_results['portfolio_change'] * 100:.2f}%")
        print(f"  Sharpe Ratio:      {final_results['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown:      {final_results['max_drawdown'] * 100:.2f}%")
        print(f"  Total Trades:      {final_results['total_trades']}")
        print(f"  Training Time:     {final_results['training_time_formatted']}")
        print("\nBest Parameters:")
        print(f"  Reward Strategy:   {best_params['env_params']['reward_strategy']}")
        print(f"  Risk Aversion:     {best_params['env_params']['risk_aversion']:.2f}")
        print(f"  Network Arch:      {best_params['policy_kwargs']['net_arch']}")
        print(f"  Learning Rate:     {best_params['ppo_params']['learning_rate']:.6f}")
        print(f"  Batch Size:        {best_params['ppo_params']['batch_size']}")
        print(f"  N Steps:           {best_params['ppo_params']['n_steps']}")
        print("=" * 80 + "\n")
        
    elif args.use_manager:
        # Train with trading manager
        print_progress(f"Training with enhanced trading manager for {args.timesteps} timesteps...")
        result = train_with_trading_manager(
            data=data,
            symbol=args.symbol,
            config_path=args.config,
            timesteps=args.timesteps,
            enhanced=True
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"TRADING MANAGER TRAINING RESULTS")
        print("=" * 80)
        print(f"Symbol: {args.symbol}")
        print(f"Status: {result.get('status', 'unknown')}")
        if 'evaluation' in result:
            eval_results = result['evaluation']
            print(f"  Mean Reward:      {eval_results.get('avg_return', 0):.4f}")
            print(f"  Win Rate:         {eval_results.get('avg_win_rate', 0) * 100:.2f}%")
            print(f"  Portfolio Change: {eval_results.get('avg_portfolio_change', 0) * 100:.2f}%")
            print(f"  Max Drawdown:     {eval_results.get('avg_max_drawdown', 0) * 100:.2f}%")
            print(f"  Total Trades:     {eval_results.get('avg_trades', 0):.0f}")
        print(f"  Training Time:    {result.get('training_time_formatted', 'unknown')}")
        print("=" * 80 + "\n")
    
    else:
        # Train without optimization
        print_progress(f"Training without optimization for {args.timesteps} timesteps...")
        result = simple_rl_training(
            data=data,
            config=env_config,
            symbol=args.symbol,
            timesteps=args.timesteps,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"SIMPLE TRAINING RESULTS")
        print("=" * 80)
        print(f"Symbol: {args.symbol}")
        print(f"Mean Reward:      {result['mean_reward']:.4f}")
        print(f"Win Rate:         {result['win_rate'] * 100:.2f}%")
        print(f"Portfolio Change: {result['portfolio_change'] * 100:.2f}%")
        print(f"Sharpe Ratio:     {result['sharpe_ratio']:.4f}")
        print(f"Max Drawdown:     {result['max_drawdown'] * 100:.2f}%")
        print(f"Total Trades:     {result['total_trades']}")
        print(f"Training Time:    {result['training_time_formatted']}")
        print("=" * 80 + "\n")
    
    total_time = time.time() - start_time
    formatted_time = time.strftime('%H:%M:%S', time.gmtime(total_time))
    logger.info(f"Total execution time: {formatted_time}")
    print_progress(f"Total execution time: {formatted_time}")


def cleanup_resources():
    """Clean up any resources before exiting."""
    # Import needed modules
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Other cleanup as needed
    print_progress("Cleaning up resources...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_progress("\n⚠️ Training interrupted by user")
        logger.info("Training interrupted by user")
    except Exception as e:
        print_progress(f"\n❌ ERROR: {str(e)}")
        logger.error(f"Unhandled exception: {e}", exc_info=True)
    finally:
        # Clean up resources before exiting
        cleanup_resources()