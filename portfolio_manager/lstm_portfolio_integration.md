# Fase 3: Integración LSTM con Portfolio Manager

## Índice
1. [Visión general](#visión-general)
2. [Componentes implementados](#componentes-implementados)
3. [Flujo de trabajo](#flujo-de-trabajo)
4. [Configuración](#configuración)
5. [Puntos clave de la implementación](#puntos-clave-de-la-implementación)
6. [Esquema de priorización](#esquema-de-priorización)
7. [Uso y ejemplos](#uso-y-ejemplos)
8. [Solución de problemas](#solución-de-problemas)
9. [Desarrollo futuro](#desarrollo-futuro)

## Visión general

La Fase 3 de la mejora del sistema LSTM implementa la integración completa con el Portfolio Manager. Este desarrollo permite que el Portfolio Manager aproveche las predicciones LSTM para mejorar la selección de símbolos, al tiempo que gestiona de forma proactiva el entrenamiento de modelos para símbolos relevantes.

Esta integración está diseñada con los siguientes objetivos:

1. **Mejorar la calidad de la selección de portafolio** utilizando insights de los modelos LSTM
2. **Priorizar el entrenamiento de modelos** para símbolos que forman parte o podrían formar parte del portafolio
3. **Proporcionar una degradación elegante** en caso de que los modelos LSTM no estén disponibles
4. **Mantener la compatibilidad** con las implementaciones existentes

## Componentes implementados

Esta fase incluye los siguientes componentes principales:

### 1. `portfolio_manager/lstm_integration.py`

Módulo de integración que actúa como puente entre el Portfolio Manager y el sistema LSTM. Proporciona métodos para:
- Solicitar entrenamiento de modelos con priorización
- Obtener predicciones LSTM con caché
- Analizar múltiples símbolos eficientemente
- Integrar puntuaciones LSTM con las puntuaciones tradicionales
- Crear programas de entrenamiento optimizados

### 2. `portfolio_manager/lstm_enhanced_manager.py`

Extensión del PortfolioManager base que incorpora la funcionalidad LSTM. Sobrescribe métodos clave para:
- Mejorar el proceso de escaneo de mercado
- Incorporar predicciones LSTM en el análisis de símbolos
- Utilizar insights LSTM en la selección de portafolio
- Solicitar entrenamiento para símbolos seleccionados
- Incluir métricas LSTM en las estadísticas del portafolio

### 3. `config/lstm_config.json`

Archivo de configuración específico para la integración LSTM que permite ajustar:
- Pesos de la influencia LSTM en las decisiones
- Límites de entrenamiento de modelos
- Umbrales de fiabilidad y calidad
- Intervalos de reentrenamiento
- Configuración de caché y priorización

### 4. `main_lstm_integration.py`

Modificación del punto de entrada principal para soportar la integración LSTM, permitiendo:
- Cargar configuraciones LSTM específicas
- Inicializar el servicio LSTM cuando está habilitado
- Utilizar el PortfolioManager mejorado con LSTM
- Mantener compatibilidad con la implementación original

### 5. `tests/test_lstm_portfolio_integration.py`

Script de prueba para verificar la correcta integración entre el sistema LSTM y el Portfolio Manager, que comprueba:
- La inicialización correcta de los componentes
- La incorporación de predicciones LSTM en el proceso de selección
- La solicitud de entrenamiento para símbolos del portafolio
- La inclusión de insights LSTM en las estadísticas

## Flujo de trabajo

El flujo de trabajo de la integración LSTM con el Portfolio Manager sigue estos pasos:

1. **Inicialización**:
   - El sistema carga la configuración LSTM específica
   - Se inicializa el servicio LSTM
   - Se crea una instancia del LSTMEnhancedPortfolioManager

2. **Escaneo de mercado**:
   - El Portfolio Manager escanea el mercado para obtener símbolos candidatos
   - Los símbolos candidatos se programan para entrenamiento LSTM con prioridad normal

3. **Análisis de símbolos**:
   - Para cada símbolo, se obtienen predicciones LSTM
   - Las predicciones se integran con el análisis técnico tradicional
   - La puntuación final incorpora tanto el análisis técnico como las predicciones LSTM

4. **Selección de portafolio**:
   - Los símbolos se clasifican según puntuaciones combinadas
   - El portafolio se forma con los símbolos de mayor puntuación
   - Los símbolos seleccionados se programan para entrenamiento LSTM con alta prioridad

5. **Actualización del portafolio**:
   - El portafolio se actualiza periódicamente
   - Los nuevos símbolos se programan para entrenamiento LSTM
   - Las estadísticas del portafolio incluyen insights LSTM

## Configuración

La configuración del sistema se realiza a través del archivo `lstm_config.json`. Los principales parámetros son:

### Configuración del Portfolio Manager

```json
"lstm": {
    "enabled": true,             // Activar/desactivar integración LSTM
    "weight": 0.3,               // Peso de LSTM en decisiones (0-1)
    "max_training_models": 5,    // Máximo de modelos a entrenar por actualización
    "max_initial_symbols": 50,   // Máximo de símbolos para análisis inicial
    "training_interval": 86400,  // Intervalo de entrenamiento en segundos
    "prediction_cache_ttl": 3600,// Tiempo de vida de caché de predicciones
    "reliability_threshold": 0.6,// Umbral mínimo de fiabilidad
    "force_retrain_days": 30,    // Días para forzar reentrenamiento
    "model_quality_threshold": 0.7 // Umbral de calidad del modelo
}
```

### Configuración del servicio LSTM

```json
"lstm_service": {
    "training_threads": 2,      // Hilos de entrenamiento paralelo
    "enable_monitoring": true,  // Activar monitoreo
    "schedule_priority": {      // Prioridades por categoría de símbolo
        "portfolio_symbols": 20,
        "watchlist_symbols": 40,
        "scanning_symbols": 60,
        "on_demand_symbols": 30
    },
    "retraining": {             // Configuración de reentrenamiento
        "auto_retrain": true,
        "min_reliability_for_update": 0.5,
        "force_retrain_age_days": 45
    }
}
```

## Puntos clave de la implementación

### 1. Integración LSTM en el análisis de símbolos

```python
# De portfolio_manager/lstm_enhanced_manager.py
def analyze_symbol(self, symbol: str, data: pd.DataFrame = None) -> Dict:
    # Primero usamos el análisis base
    base_analysis = super().analyze_symbol(symbol, data)
    
    # Obtenemos la predicción LSTM
    lstm_result = self.lstm_integration.get_lstm_prediction(symbol, data)
    
    if lstm_result.get('status') == 'success':
        # Extraemos información útil
        direction = lstm_result.get('predicted_direction', 'neutral')
        confidence = lstm_result.get('confidence', 0)
        
        # Calculamos ajuste de puntuación basado en LSTM
        lstm_score_adj = 0
        if direction == 'up':
            lstm_score_adj = confidence * 30  # Hasta +30 puntos para alta confianza alcista
        elif direction == 'down':
            lstm_score_adj = -confidence * 30  # Hasta -30 puntos para alta confianza bajista
        
        # Ajustamos la puntuación base con insights LSTM
        base_score = base_analysis.get('score', 50)
        adjusted_score = max(0, min(100, base_score + lstm_score_adj))
        
        # Actualizamos la recomendación basada en la nueva puntuación
        # ...
```

### 2. Solicitud de entrenamiento priorizado

```python
# De portfolio_manager/lstm_integration.py
def request_model_training(self, symbols: List[str], priority_override: Dict[str, int] = None) -> Dict:
    # Inicializamos resultados
    results = {
        'requested': 0,
        'already_training': 0,
        'details': {}
    }
    
    # Calculamos prioridades predeterminadas
    base_priority = 50
    default_priorities = {}
    for i, symbol in enumerate(symbols):
        position_adjustment = min(30, 5 * i)
        default_priorities[symbol] = base_priority + position_adjustment
    
    # Procesamos cada símbolo
    for symbol in symbols:
        # Obtenemos estado del modelo para verificar si se necesita entrenamiento
        status = get_model_status(symbol)
        
        # Determinamos prioridad - usamos override si se proporciona, de lo contrario predeterminado
        priority = priority_override.get(symbol, default_priorities[symbol]) if priority_override else default_priorities[symbol]
        
        # Ajustamos prioridad basada en estado del modelo
        if status.get('status') == 'not_available':
            # Mayor prioridad (número más bajo) para modelos faltantes
            priority = max(10, priority - 20)
        
        # Solicitamos entrenamiento con prioridad determinada
        success = request_model_training(symbol, priority)
        # ...
```

### 3. Selección de portafolio mejorada con LSTM

```python
# De portfolio_manager/lstm_enhanced_manager.py
def select_portfolio(self, num_symbols: int = None) -> Dict[str, float]:
    # Obtenemos símbolos candidatos
    candidate_symbols = self.scan_market()[:num_symbols * 2]
    
    # Obtenemos análisis LSTM por lotes para todos los símbolos
    lstm_batch_analysis = self.lstm_integration.analyze_symbols_with_lstm(
        [s for s in candidate_symbols if s in symbol_data],
        symbol_data
    )
    
    # Analizamos cada símbolo incorporando insights LSTM
    for symbol in candidate_symbols:
        if symbol in symbol_data:
            analysis = self.analyze_symbol(symbol, symbol_data[symbol])
            symbol_analyses[symbol] = analysis
            
            # También guardamos análisis LSTM por separado
            if symbol in lstm_batch_analysis:
                lstm_analysis_results[symbol] = lstm_batch_analysis[symbol]
    
    # Filtramos símbolos con recomendación positiva
    # ...
    
    # Programamos entrenamiento para los símbolos seleccionados (alta prioridad)
    self.lstm_integration.request_model_training(
        list(portfolio.keys()),
        {s: 30 for s in portfolio.keys()}  # Mayor prioridad (30 en lugar de 50 predeterminado)
    )
    # ...
```

### 4. Estadísticas de portafolio con insights LSTM

```python
# De portfolio_manager/lstm_enhanced_manager.py
def get_portfolio_stats(self) -> Dict:
    # Obtenemos estadísticas base
    base_stats = super().get_portfolio_stats()
    
    # Si no hay portafolio activo, devolvemos estadísticas base
    if not self.current_portfolio:
        return base_stats
    
    # Obtenemos predicciones LSTM para símbolos del portafolio actual
    lstm_predictions = {}
    
    for symbol in self.current_portfolio.keys():
        prediction = self.lstm_integration.get_lstm_prediction(symbol)
        if prediction.get('status') == 'success':
            lstm_predictions[symbol] = {
                'direction': prediction.get('predicted_direction', 'unknown'),
                'confidence': prediction.get('confidence', 0),
                'price_change_pct': prediction.get('price_change_pct', 0),
                'model_type': prediction.get('model_type', 'unknown')
            }
    
    # Calculamos perspectiva de nivel de portafolio LSTM
    if lstm_predictions:
        # Promedio ponderado de cambios de precio previstos
        weighted_change = 0
        total_weight = 0
        
        for symbol, pred in lstm_predictions.items():
            allocation = self.current_portfolio.get(symbol, 0)
            confidence = pred.get('confidence', 0)
            price_change = pred.get('price_change_pct', 0)
            
            # Ponderamos tanto por asignación como por confianza
            weight = allocation * confidence
            weighted_change += price_change * weight
            total_weight += weight
        
        # Normalizamos
        if total_weight > 0:
            portfolio_outlook = weighted_change / total_weight
        else:
            portfolio_outlook = 0
        
        # Añadimos insights LSTM a estadísticas
        base_stats['lstm_insights'] = {
            'predictions': lstm_predictions,
            'portfolio_outlook': portfolio_outlook,
            'symbols_with_predictions': len(lstm_predictions),
            'timestamp': datetime.datetime.now().isoformat()
        }
    # ...
```

## Esquema de priorización

El sistema utiliza un sofisticado esquema de priorización para asignar recursos de entrenamiento de forma óptima:

### Prioridades base por categoría

| Categoría de símbolo | Nivel de prioridad | Valor numérico |
|----------------------|-------------------|----------------|
| Símbolos en portafolio | Muy alta | 20 |
| Símbolos de demanda específica | Alta | 30 |
| Símbolos en lista de seguimiento | Media | 40 |
| Símbolos de escaneo general | Baja | 60 |

### Ajustes de prioridad

La prioridad base se ajusta según varios factores:

1. **Estado del modelo**:
   - Modelo no disponible: -20 (mayor prioridad)
   - Modelo desactualizado: -10 (mayor prioridad)
   - Modelo fiable y actualizado: +20 (menor prioridad)

2. **Rendimiento del modelo**:
   - Baja fiabilidad (<0.5): -15 (mayor prioridad)
   - Alta fiabilidad (>0.7): +15 (menor prioridad)

3. **Estado de entrenamiento**:
   - Ya en entrenamiento/programado: +40 (mucha menor prioridad)

4. **Posición en la lista**:
   - Se aplica un ajuste gradual basado en la posición (+5 por posición)

### Ejemplo de cálculo de prioridad

Para un símbolo en portafolio (prioridad base 20):
- Sin modelo disponible: 20 - 20 = 0 (prioridad extremadamente alta)
- Con modelo desactualizado y baja fiabilidad: 20 - 10 - 15 = -5 (prioridad muy alta)
- Con modelo fiable y actualizado: 20 + 20 = 40 (prioridad media)
- Ya en entrenamiento: 20 + 40 = 60 (prioridad baja)

## Uso y ejemplos

### Habilitar la integración LSTM

Para habilitar la integración LSTM, ejecute la aplicación con el archivo de configuración LSTM:

```bash
python main_lstm_integration.py --lstm-config config/lstm_config.json
```

O use el indicador de activación LSTM para usar la configuración predeterminada:

```bash
python main_lstm_integration.py --enable-lstm
```

### Ejemplo de análisis de símbolo mejorado con LSTM

```python
# Inicializar el gestor de portafolio mejorado con LSTM
portfolio_manager = LSTMEnhancedPortfolioManager(config_path='config/lstm_config.json')

# Analizar un símbolo
analysis = portfolio_manager.analyze_symbol('AAPL')

# Verificar contribución LSTM
if 'lstm' in analysis:
    lstm_info = analysis['lstm']
    print(f"Dirección LSTM: {lstm_info['direction']}")
    print(f"Confianza: {lstm_info['confidence']:.2f}")
    print(f"Contribución a puntuación: {lstm_info['contribution']:.2f}")

# Verificar puntuación y recomendación finales
print(f"Puntuación final: {analysis['score']}")
print(f"Recomendación: {analysis['recommendation']}")
```

### Ejemplo de estadísticas de portafolio con insights LSTM

```python
# Obtener estadísticas de portafolio
stats = portfolio_manager.get_portfolio_stats()

# Verificar insights LSTM
if 'lstm_insights' in stats:
    insights = stats['lstm_insights']
    
    # Mostrar perspectiva general del portafolio
    print(f"Perspectiva LSTM del portafolio: {insights['portfolio_outlook']:.2f}%")
    
    # Mostrar predicciones individuales
    for symbol, pred in insights['predictions'].items():
        allocation = portfolio_manager.current_portfolio.get(symbol, 0) * 100
        print(f"{symbol} ({allocation:.1f}%): {pred['direction']}, "
              f"{pred['confidence']:.2f} confianza, {pred['price_change_pct']:.2f}% cambio")
```

## Solución de problemas

### Problemas comunes y soluciones

| Problema | Posible causa | Solución |
|----------|---------------|----------|
| No hay predicciones LSTM en análisis | Servicio LSTM no inicializado | Verificar que la integración LSTM esté habilitada en la configuración |
| Baja influencia LSTM en decisiones | Peso LSTM configurado bajo | Aumentar el parámetro `weight` en la configuración LSTM |
| Errores de modelo no disponible | Modelos no entrenados | Ejecutar entrenamiento inicial o aumentar `max_training_models` |
| Caché de predicciones antigua | TTL de caché demasiado largo | Reducir `prediction_cache_ttl` o usar `force_refresh=True` |
| Todos los modelos son universales | Entrenamiento en segundo plano no completo | Esperar a que los entrenamientos programados se completen |

### Verificación del sistema

Para verificar la correcta integración LSTM:

```bash
python -m tests.test_lstm_portfolio_integration
```

Este script de prueba validará:
- La inicialización correcta del sistema
- La obtención de predicciones LSTM
- La integración de las predicciones en el análisis
- La solicitud de entrenamiento para símbolos relevantes
- Las estadísticas de portafolio mejoradas con LSTM

## Desarrollo futuro

La integración LSTM con el Portfolio Manager establece las bases para desarrollos adicionales:

### 1. Fase 4: Sistema de monitoreo y actualización automática

- Implementación de un dashboard para visualizar el estado y rendimiento de los modelos LSTM
- Sistema automatizado para detectar y reentrenar modelos degradados
- Optimización continua de hiperparámetros para cada símbolo

### 2. Mejoras de integración

- Integración más profunda con el Trading Manager para usar predicciones LSTM en decisiones de salida
- Incorporación de predicciones LSTM en la gestión dinámica de riesgo
- Adaptación del tamaño de posición basada en la confianza del modelo LSTM

### 3. Refinamientos adicionales

- Sistema de votación combinando múltiples arquitecturas LSTM
- Adaptación de los pesos LSTM basada en el desempeño histórico
- Incorporación de características macroeconómicas en los modelos LSTM