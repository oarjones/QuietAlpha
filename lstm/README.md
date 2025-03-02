# Sistema LSTM para Predicción de Precios

Este módulo implementa un sistema avanzado de predicción de precios basado en redes LSTM (Long Short-Term Memory), con capacidad para gestionar tanto modelos universales como modelos específicos por símbolo.

## Características principales

- **Modelos específicos por símbolo**: Entrenamiento y predicción con modelos adaptados a cada activo.
- **Modelo universal de respaldo**: Para símbolos sin modelo específico o como apoyo inicial.
- **Entrenamiento incremental**: Actualización de modelos existentes sin reentrenar completamente.
- **Entrenamiento en paralelo**: Procesamiento múltiple para optimizar el uso de recursos.
- **Priorización inteligente**: Los símbolos se priorizan en función de volumen, volatilidad y otros factores.
- **Cálculo de fiabilidad**: Índice de confianza para cada modelo basado en múltiples métricas.

## Estructura del módulo

- `model.py`: Funciones básicas para construir, entrenar y evaluar modelos LSTM.
- `model_manager.py`: Gestión de modelos (carga, guardado, verificación).
- `trainer.py`: Sistema de entrenamiento paralelo con priorización.
- `integration.py`: Integración con el resto del sistema (interfaz principal).
- `init_system.py`: Inicialización y configuración del sistema.
- `experiments.py`: Herramientas para experimentación y optimización de hiperparámetros.

## Directorios de modelos

```
models/
  └── lstm/
      ├── universal/          # Modelo universal (usado como respaldo)
      │   ├── model.keras     # Archivo del modelo
      │   ├── scaler.pkl      # Objeto de normalización
      │   └── metadata.json   # Metadatos y métricas
      └── symbols/            # Modelos específicos por símbolo
          ├── AAPL/
          │   ├── model.keras
          │   ├── scaler.pkl
          │   └── metadata.json
          ├── MSFT/
          └── ...
```

## Uso básico

### Inicializar el sistema

```python
# Asegúrate de ejecutar esto antes de usar cualquier funcionalidad LSTM
from lstm.init_system import main as init_lstm
init_lstm()
```

### Realizar predicciones

```python
# Método simple usando la integración
from lstm.integration import predict_with_lstm

# Con datos ya disponibles
prediction = predict_with_lstm('AAPL', data_df)

# O dejando que el sistema obtenga los datos (requiere ibkr_interface)
prediction = predict_with_lstm('AAPL', ibkr_interface=ibkr)

# Verificar resultado
if prediction['status'] == 'success':
    print(f"Predicción: {prediction['predicted_direction']}")
    print(f"Confianza: {prediction['confidence']:.2f}")
    print(f"Precio actual: {prediction['current_price']:.2f}")
    print(f"Precio predicho: {prediction['predicted_price']:.2f}")
```

### Solicitar entrenamiento de modelos

```python
from lstm.integration import request_model_training

# Solicitar entrenamiento para un símbolo con alta prioridad (número más bajo)
request_model_training('AAPL', priority=30)

# Para entrenar múltiples símbolos con priorización automática
from lstm.integration import get_lstm_service
service = get_lstm_service()
service.request_batch_training(['AAPL', 'MSFT', 'GOOGL'], auto_prioritize=True)
```

### Verificar estado de los modelos

```python
from lstm.integration import get_model_status

# Estado de un modelo específico
status = get_model_status('AAPL')
print(f"Estado: {status['status']}")
print(f"Fiabilidad: {status.get('reliability', 0):.2f}")

# Estado de todos los modelos
all_status = get_model_status()
```

## Integración con TradingManager

El sistema LSTM está integrado directamente con el `TradingManager`, reemplazando el método `predict_price_with_lstm` con una implementación que aprovecha los modelos específicos por símbolo cuando están disponibles.

Cuando un símbolo se opera usando el modelo universal, el sistema automáticamente programa el entrenamiento de un modelo específico con alta prioridad.

## Optimización y ajuste

Para experimentar con diferentes configuraciones de modelos, utiliza el módulo `experiments.py`:

```python
# Entrenar y evaluar modelos con diferentes configuraciones
from lstm.experiments import train_universal_model, evaluate_multiple_symbols

# Entrenar modelo universal
train_universal_model(['AAPL', 'MSFT', 'GOOGL'])

# Evaluar en múltiples símbolos
results = evaluate_multiple_symbols(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
```

## Ejemplos prácticos

Consulta el directorio `examples/` para ver ejemplos completos de uso, incluyendo:
- `train_and_predict.py`: Entrenamiento y predicción básicos
- Visualización de resultados
- Comparación entre modelos específicos y universal

## Requisitos

- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Desarrollo futuro

Este módulo implementa la Fase 2 (Sistema de entrenamiento) del plan de mejora LSTM. Las fases futuras incluirán:

- **Fase 3**: Integración avanzada con el Portfolio Manager
- **Fase 4**: Sistema de monitoreo y actualización automática