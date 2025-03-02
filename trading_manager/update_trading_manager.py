"""
Update Trading Manager for LSTM Integration

Este script actualiza la implementación del TradingManager, sustituyendo el método
predict_price_with_lstm por una nueva versión que se integra con el sistema LSTM mejorado.

Es una solución más directa que el patch anterior, modificando directamente el código fuente.
"""

import os
import shutil
import logging
import tempfile
import re
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_trading_manager():
    """
    Actualiza el archivo trading_manager/base.py para integrar el nuevo sistema LSTM.
    """
    # Ruta al archivo TradingManager
    base_file = Path('trading_manager/base.py')
    
    if not base_file.exists():
        logger.error(f"No se encontró el archivo {base_file}")
        return False
    
    # Crear copia de seguridad
    backup_file = base_file.with_suffix('.py.bak')
    logger.info(f"Creando copia de seguridad en {backup_file}")
    shutil.copy2(base_file, backup_file)
    
    # Leer el contenido del archivo
    with open(base_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Definir el nuevo método predict_price_with_lstm
    new_method = """
    def predict_price_with_lstm(self, symbol: str, data: pd.DataFrame = None) -> Dict:
        \"\"\"
        Utiliza modelos LSTM para predecir precios futuros.
        Esta versión integra el nuevo sistema LSTM con modelos específicos por símbolo.
        
        Args:
            symbol (str): Símbolo para el cual hacer la predicción
            data (pd.DataFrame): Datos históricos (opcional, se obtienen si es None)
            
        Returns:
            Dict: Resultado de la predicción
        \"\"\"
        try:
            # Importar el servicio LSTM (importación tardía para evitar dependencias circulares)
            # Se importa aquí para no afectar al resto del módulo si no está disponible
            from lstm.integration import predict_with_lstm, request_model_training
            
            # Obtener datos si no se proporcionan
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                
            if data.empty:
                logger.warning(f"No hay datos disponibles para {symbol}")
                return {'predicted_direction': 'unknown', 'confidence': 0}
            
            # Realizar predicción usando el servicio LSTM
            prediction = predict_with_lstm(symbol, data, self.ibkr)
            
            # Si la predicción falló, registrar error y devolver un marcador de posición
            if prediction.get('status') != 'success':
                logger.warning(f"La predicción LSTM falló para {symbol}: {prediction.get('message', 'Error desconocido')}")
                return {'predicted_direction': 'unknown', 'confidence': 0}
            
            # Si se usó el modelo universal, programar entrenamiento para un modelo específico
            # Esto permite que el sistema aprenda y mejore con el tiempo
            if prediction.get('model_type') == 'universal':
                logger.info(f"Se usó el modelo universal para {symbol}, programando entrenamiento para modelo específico")
                request_model_training(symbol, priority=70)  # Prioridad alta (número más bajo)
            
            return prediction
                
        except ImportError:
            # Si el módulo de integración LSTM no está disponible, usar la implementación original
            logger.warning(f"Módulo lstm.integration no disponible, usando implementación LSTM antigua para {symbol}")
            
            # Implementación original simplificada (fallback)
            # Esta es una versión reducida del método original que sirve como respaldo
            if data is None or data.empty:
                data = self.get_market_data(symbol)
                
            if data.empty:
                return {'predicted_direction': 'unknown', 'confidence': 0}
            
            # Extraer tendencia básica de los datos
            if len(data) >= 5:
                recent_trend = data['close'].iloc[-1] / data['close'].iloc[-5] - 1
                if recent_trend > 0.01:
                    return {'predicted_direction': 'up', 'confidence': min(1.0, recent_trend * 10)}
                elif recent_trend < -0.01:
                    return {'predicted_direction': 'down', 'confidence': min(1.0, abs(recent_trend) * 10)}
            
            return {'predicted_direction': 'neutral', 'confidence': 0}
            
        except Exception as e:
            logger.error(f"Error en la predicción LSTM para {symbol}: {e}")
            return {'predicted_direction': 'error', 'confidence': 0, 'error': str(e)}
    """
    
    # Buscar el método existente usando regex
    method_pattern = re.compile(r'def predict_price_with_lstm\([^)]*\)[^:]*:(.*?)(?=\n    def|\n\n)', re.DOTALL)
    match = method_pattern.search(content)
    
    if not match:
        logger.error("No se encontró el método predict_price_with_lstm en el archivo base.py")
        return False
    
    # Reemplazar el método
    new_content = method_pattern.sub(new_method, content)
    
    # Escribir el nuevo contenido
    with open(base_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    logger.info(f"Se actualizó con éxito el método predict_price_with_lstm en {base_file}")
    return True

def restore_backup():
    """
    Restaura la copia de seguridad del archivo trading_manager/base.py si existe.
    """
    base_file = Path('trading_manager/base.py')
    backup_file = base_file.with_suffix('.py.bak')
    
    if backup_file.exists():
        logger.info(f"Restaurando copia de seguridad desde {backup_file}")
        shutil.copy2(backup_file, base_file)
        return True
    else:
        logger.error(f"No se encontró copia de seguridad en {backup_file}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Actualizar TradingManager con integración LSTM')
    parser.add_argument('--restore', action='store_true', help='Restaurar copia de seguridad')
    
    args = parser.parse_args()
    
    if args.restore:
        success = restore_backup()
        if success:
            print("Se restauró correctamente la copia de seguridad")
        else:
            print("Error al restaurar la copia de seguridad")
    else:
        success = update_trading_manager()
        if success:
            print("Se actualizó correctamente TradingManager con la nueva integración LSTM")
        else:
            print("Error al actualizar TradingManager")