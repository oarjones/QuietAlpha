"""
LSTM Model Manager

Este módulo proporciona funcionalidades para gestionar modelos LSTM específicos por símbolo,
incluyendo verificación de existencia, carga de modelos y programación de entrenamientos.
"""

import os
import json
import shutil
import logging
import concurrent.futures
from datetime import datetime
import numpy as np
import joblib
from typing import Dict, Tuple, Optional, List, Any

logger = logging.getLogger(__name__)

class LSTMModelManager:
    """
    Gestor de modelos LSTM específicos por símbolo.
    
    Proporciona funcionalidades para:
    - Verificar la existencia de modelos para símbolos específicos
    - Cargar modelos y scalers
    - Programar entrenamientos para nuevos modelos
    - Gestionar metadatos de modelos
    """
    
    def __init__(self, base_path: str = None):
        """
        Inicializar el gestor de modelos LSTM.
        
        Args:
            base_path: Ruta base para almacenar modelos (por defecto: models/lstm/symbols)
        """
        if base_path is None:
            base_path = os.path.join('models', 'lstm', 'symbols')
        
        self.base_path = base_path
        self.universal_path = os.path.join('models', 'lstm', 'universal')
        self.training_queue = {}  # Symbol -> Future
        self.max_age_days = 30  # Reentrenar después de 30 días
        
        # Crear directorios si no existen
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.universal_path, exist_ok=True)
        
        logger.info(f"LSTM Model Manager initialized with base path: {self.base_path}")
    
    def get_symbol_dir(self, symbol: str) -> str:
        """
        Obtener directorio para un símbolo específico.
        
        Args:
            symbol: Símbolo para el cual obtener directorio
            
        Returns:
            str: Ruta al directorio del símbolo
        """
        symbol_dir = os.path.join(self.base_path, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        return symbol_dir
    
    def get_model_path(self, symbol: str) -> str:
        """
        Obtener ruta del modelo para un símbolo.
        
        Args:
            symbol: Símbolo para el cual obtener ruta
            
        Returns:
            str: Ruta al archivo del modelo
        """
        return os.path.join(self.get_symbol_dir(symbol), "model.keras")
    
    def get_scaler_path(self, symbol: str) -> str:
        """
        Obtener ruta del scaler para un símbolo.
        
        Args:
            symbol: Símbolo para el cual obtener ruta
            
        Returns:
            str: Ruta al archivo del scaler
        """
        return os.path.join(self.get_symbol_dir(symbol), "scaler.pkl")
    
    def get_metadata_path(self, symbol: str) -> str:
        """
        Obtener ruta del archivo de metadatos para un símbolo.
        
        Args:
            symbol: Símbolo para el cual obtener ruta
            
        Returns:
            str: Ruta al archivo de metadatos
        """
        return os.path.join(self.get_symbol_dir(symbol), "metadata.json")
    
    def model_exists(self, symbol: str) -> bool:
        """
        Verificar si existe un modelo para el símbolo.
        
        Args:
            symbol: Símbolo a verificar
            
        Returns:
            bool: True si el modelo existe, False en caso contrario
        """
        return os.path.exists(self.get_model_path(symbol))
    
    def model_is_fresh(self, symbol: str) -> bool:
        """
        Verificar si el modelo está actualizado.
        
        Un modelo se considera 'fresco' si su edad es menor que max_age_days.
        
        Args:
            symbol: Símbolo a verificar
            
        Returns:
            bool: True si el modelo está actualizado, False en caso contrario
        """
        metadata_path = self.get_metadata_path(symbol)
        
        if not os.path.exists(metadata_path):
            return False
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            trained_date = datetime.fromisoformat(metadata.get('trained_date', '2000-01-01T00:00:00'))
            days_old = (datetime.now() - trained_date).days
            
            return days_old <= self.max_age_days
        except Exception as e:
            logger.error(f"Error checking model freshness for {symbol}: {e}")
            return False
    
    def get_model_metadata(self, symbol: str) -> Dict:
        """
        Obtener metadatos del modelo para un símbolo.
        
        Args:
            symbol: Símbolo para el cual obtener metadatos
            
        Returns:
            dict: Metadatos del modelo o diccionario vacío si no existen
        """
        metadata_path = self.get_metadata_path(symbol)
        
        if not os.path.exists(metadata_path):
            return {}
        
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {symbol}: {e}")
            return {}
    
    def save_model_metadata(self, symbol: str, metadata: Dict) -> bool:
        """
        Guardar metadatos del modelo para un símbolo.
        
        Args:
            symbol: Símbolo para el cual guardar metadatos
            metadata: Diccionario con metadatos
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        metadata_path = self.get_metadata_path(symbol)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metadata for {symbol}: {e}")
            return False
    
    def load_model_and_scaler(self, symbol: str) -> Tuple[Any, Any]:
        """
        Cargar modelo y scaler para un símbolo.
        
        Args:
            symbol: Símbolo para el cual cargar modelo y scaler
            
        Returns:
            tuple: (model, scaler) o (None, None) si no existe
        """
        try:
            # Verificar si existe el modelo
            if not self.model_exists(symbol):
                logger.warning(f"Model for {symbol} does not exist")
                return None, None
            
            # Importar aquí para evitar importaciones circulares
            import keras as ke
            
            # Cargar modelo
            model_path = self.get_model_path(symbol)
            model = ke.models.load_model(model_path)
            
            # Cargar scaler
            scaler_path = self.get_scaler_path(symbol)
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                logger.warning(f"Scaler for {symbol} not found")
                scaler = None
            
            logger.info(f"Loaded model and scaler for {symbol}")
            return model, scaler
        
        except Exception as e:
            logger.error(f"Error loading model and scaler for {symbol}: {e}")
            return None, None
    
    def load_universal_model_and_scaler(self) -> Tuple[Any, Any]:
        """
        Cargar modelo y scaler universal.
        
        Returns:
            tuple: (model, scaler) o (None, None) si no existe
        """
        try:
            # Importar aquí para evitar importaciones circulares
            import keras as ke
            
            # Comprobar si existe el modelo universal
            model_path = os.path.join(self.universal_path, "model.keras")
            scaler_path = os.path.join(self.universal_path, "scaler.pkl")
            
            if not os.path.exists(model_path):
                logger.warning("Universal model does not exist")
                return None, None
            
            # Cargar modelo
            model = ke.models.load_model(model_path)
            
            # Cargar scaler
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
            else:
                logger.warning("Universal scaler not found")
                scaler = None
            
            logger.info("Loaded universal model and scaler")
            return model, scaler
        
        except Exception as e:
            logger.error(f"Error loading universal model and scaler: {e}")
            return None, None
    
    def schedule_training(self, symbol: str, force: bool = False) -> None:
        """
        Programar entrenamiento asíncrono para un símbolo.
        
        Args:
            symbol: Símbolo para el cual entrenar modelo
            force: Forzar reentrenamiento incluso si existe
        """
        # Verificar si ya existe y está actualizado
        if not force and self.model_exists(symbol) and self.model_is_fresh(symbol):
            logger.info(f"Model for {symbol} already exists and is fresh. Skipping training.")
            return
        
        # Verificar si ya está en cola de entrenamiento
        if symbol in self.training_queue:
            logger.info(f"Training for {symbol} already scheduled")
            return
        
        # Programar entrenamiento en un hilo separado
        logger.info(f"Scheduling training for {symbol}")
        
        # En esta fase inicial, solo registramos que se debería entrenar
        self.training_queue[symbol] = True
        
        # En una implementación completa, usaríamos hilos o procesos:
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     self.training_queue[symbol] = executor.submit(self.train_model, symbol)
    
    def train_model(self, symbol: str) -> Dict:
        """
        Entrenar modelo para un símbolo específico.
        
        Esta es una implementación básica inicial. En fases posteriores,
        se integrará con lstm/model.py para entrenamiento real.
        
        Args:
            symbol: Símbolo para el cual entrenar modelo
            
        Returns:
            dict: Resultado del entrenamiento
        """
        logger.info(f"Training model for {symbol} would be implemented in Phase 2")
        
        # Crear metadatos básicos para probar la infraestructura
        metadata = {
            'trained_date': datetime.now().isoformat(),
            'symbol': symbol,
            'metrics': {
                'rmse': 0.0,
                'mae': 0.0,
                'r2': 0.0
            },
            'reliability_index': 0.0,
            'status': 'pending',
            'message': 'Training not implemented in Phase 1'
        }
        
        # Guardar metadatos básicos
        self.save_model_metadata(symbol, metadata)
        
        return {
            'status': 'pending',
            'message': 'Training will be implemented in Phase 2'
        }
    
    def get_available_models(self) -> List[str]:
        """
        Obtener lista de modelos disponibles.
        
        Returns:
            list: Lista de símbolos con modelos disponibles
        """
        if not os.path.exists(self.base_path):
            return []
        
        # Buscar directorios de símbolos
        symbols = []
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "model.keras")):
                symbols.append(item)
        
        return symbols
    
    def clean_old_models(self, max_days: int = 90) -> int:
        """
        Eliminar modelos antiguos que superan cierta edad.
        
        Args:
            max_days: Edad máxima en días
            
        Returns:
            int: Número de modelos eliminados
        """
        if not os.path.exists(self.base_path):
            return 0
        
        # Buscar modelos antiguos
        removed = 0
        for symbol in self.get_available_models():
            metadata = self.get_model_metadata(symbol)
            if not metadata:
                continue
            
            trained_date = datetime.fromisoformat(metadata.get('trained_date', '2000-01-01T00:00:00'))
            days_old = (datetime.now() - trained_date).days
            
            if days_old > max_days:
                # Eliminar directorio completo
                try:
                    shutil.rmtree(self.get_symbol_dir(symbol))
                    removed += 1
                    logger.info(f"Removed old model for {symbol} ({days_old} days old)")
                except Exception as e:
                    logger.error(f"Error removing old model for {symbol}: {e}")
        
        return removed