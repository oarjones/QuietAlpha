"""
Script para probar el entrenamiento paralelo de modelos LSTM

Este script prueba la solución implementada para asignar instancias independientes
de IBKRInterface a cada thread de entrenamiento.
"""

import os
import sys
import logging
import time
from datetime import datetime
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Asegurar que podemos importar desde el directorio raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar componentes necesarios
from lstm.trainer import LSTMTrainer
from lstm.integration import get_lstm_service, predict_with_lstm
from ibkr_api.interface import IBKRInterface

def test_parallel_training(symbols, max_parallel=2, monitor=True):
    """
    Prueba el entrenamiento paralelo de modelos LSTM.
    
    Args:
        symbols: Lista de símbolos para entrenar
        max_parallel: Máximo número de entrenamientos paralelos
        monitor: Si se debe monitorear el progreso
        
    Returns:
        dict: Resultados del entrenamiento
    """
    try:
        # Crear trainer con el número máximo de entrenamientos paralelos
        trainer = LSTMTrainer()
        trainer.max_parallel_trainings = max_parallel
        
        # Iniciar entrenamiento por lotes
        result = trainer.run_parallel_batch_training(symbols)
        
        if result['status'] != 'success':
            logger.error(f"Error iniciando entrenamiento: {result}")
            return result
        
        # Mostrar información inicial
        logger.info(f"Iniciado entrenamiento para {len(result['scheduled_symbols'])} símbolos")
        logger.info(f"Símbolos programados: {result['scheduled_symbols']}")
        
        # Si no queremos monitorear, salir aquí
        if not monitor:
            logger.info("Entrenamiento en ejecución en segundo plano")
            return result
        
        # Monitorear progreso
        logger.info("Monitoreando progreso del entrenamiento...")
        logger.info("Presiona Ctrl+C para detener el monitoreo (el entrenamiento continuará)")
        
        try:
            while True:
                # Obtener estado de la cola
                queue_status = trainer.get_queue_status()
                active_count = len(queue_status['active_trainings'])
                queue_size = queue_status['queue_size']
                
                # Mostrar estado actual
                if active_count > 0:
                    logger.info(f"Entrenamientos activos: {queue_status['active_trainings']}, Cola: {queue_size}")
                else:
                    logger.info(f"No hay entrenamientos en curso, Cola: {queue_size}")
                
                # Mostrar estado de cada símbolo
                training_status = {}
                for symbol in result['scheduled_symbols']:
                    status = trainer.get_training_status(symbol)
                    if status:
                        training_status[symbol] = status.get('status', 'unknown')
                        
                        # Mostrar métricas para entrenamientos completados con éxito
                        if status.get('status') == 'success' and 'metrics' in status:
                            metrics = status['metrics']
                            rmse = metrics.get('rmse', 'N/A')
                            r2 = metrics.get('r2', 'N/A')
                            logger.info(f"Símbolo {symbol} completado: RMSE={rmse}, R²={r2}")
                
                logger.info(f"Estado de entrenamientos: {training_status}")
                
                # Verificar si todos los entrenamientos han terminado
                all_completed = True
                for symbol in result['scheduled_symbols']:
                    status = trainer.get_training_status(symbol)
                    if not status or status.get('status') not in ['success', 'error']:
                        all_completed = False
                        break
                
                if all_completed:
                    logger.info("Todos los entrenamientos completados!")
                    break
                
                # Si no hay entrenamientos activos y la cola está vacía, pero faltan entrenamientos por completar
                if active_count == 0 and queue_size == 0 and not all_completed:
                    logger.warning("No hay entrenamientos activos ni encolados, pero aún hay entrenamientos pendientes")
                    # Esperar un poco más y volver a verificar
                    time.sleep(10)
                    
                    # Si sigue igual, salir
                    queue_status = trainer.get_queue_status()
                    if len(queue_status['active_trainings']) == 0 and queue_status['queue_size'] == 0:
                        logger.error("Parece que ha habido un problema con el sistema de entrenamiento")
                        break
                
                # Esperar antes de la siguiente comprobación
                time.sleep(15)
                
        except KeyboardInterrupt:
            logger.info("Monitoreo interrumpido por el usuario")
        
        # Recopilar resultados finales
        final_results = {}
        for symbol in result['scheduled_symbols']:
            status = trainer.get_training_status(symbol)
            final_results[symbol] = {
                'status': status.get('status', 'unknown'),
                'message': status.get('message', ''),
                'metrics': status.get('metrics', {})
            }
        
        return {
            'status': 'monitoring_complete',
            'initial_result': result,
            'final_results': final_results
        }
        
    except Exception as e:
        logger.error(f"Error en test_parallel_training: {e}")
        return {'status': 'error', 'message': str(e)}

def main():
    """Función principal para ejecutar la prueba."""
        
    # Ejecutar prueba
    result = test_parallel_training(
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
        max_parallel=2,
        monitor=True
    )
    
    logger.info(f"Prueba completada: {result['status']}")
    
    # Mostrar resultados finales
    if 'final_results' in result:
        successful = 0
        failed = 0
        
        logger.info("Resultados por símbolo:")
        for symbol, res in result['final_results'].items():
            if res['status'] == 'success':
                successful += 1
                metrics = res['metrics']
                logger.info(f"  {symbol}: ÉXITO - RMSE: {metrics.get('rmse', 'N/A')}, R²: {metrics.get('r2', 'N/A')}")
            else:
                failed += 1
                logger.info(f"  {symbol}: FALLIDO - {res['message']}")
        
        logger.info(f"Resumen: {successful} exitosos, {failed} fallidos")

if __name__ == "__main__":
    main()