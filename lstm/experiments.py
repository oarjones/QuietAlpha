import random
from model import run_lstm_training_with_score, run_lstm_training

def random_search(config_base, n_iter=10):
    best_config = None
    best_score = float('inf')

    for _ in range(n_iter):
        config = config_base.copy()
        config['units'] = random.choice([50, 100, 150, 200])
        config['dropout'] = random.uniform(0.2, 0.5)
        config['learning_rate'] = random.choice([0.001, 0.0005, 0.0001])
        config['num_layers'] = random.choice([2, 3, 4])
        config['batch_size'] = random.choice([64, 128, 256])
        config['bidirectional'] = random.choice([True, False])
        config['l2_reg'] = random.choice([0, 0.01, 0.001])
        config['lr_factor'] = random.choice([0.1, 0.2, 0.3])
        
        # Ejecutar el entrenamiento con la configuración actual
        score = run_lstm_training_with_score(config['symbol'], config)

        # Actualizar la mejor configuración si el score es mejor
        if score < best_score:
            best_score = score
            best_config = config

    return best_config, best_score



#Aqui definimos las distintas configuraciones que queremos probar.
if __name__ == '__main__':
    # --- Configuración Base (Experimento 1) ---
    config_base = {
        'label': 'experimento_base',  # Etiqueta para identificar el experimento
        'symbol': 'AAPL',  # Símbolo de la acción a predecir
        'seq_length': 60,
        'epochs': 50,
        'batch_size': 64,
        'units': 50,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'patience': 10,
        'num_layers': 3,  # Número de capas LSTM
        'bidirectional': False,  # Usar LSTM bidireccional o no
        'l2_reg': 0, #Regularización L2
        'lr_factor': 0.2,  # Factor de reducción del learning rate
        'min_lr': 0.00001,  # Mínimo learning rate

    }

    # Realizar búsqueda aleatoria de hiperparámetros
    # best_config, best_score = random_search(config_base, n_iter=40)
    # print(f"Mejor configuración: {best_config}")
    # print(f"Mejor score: {best_score}")

    experimento_final_AMZN = {'label': 'experimento_final_AMZN', 'symbol': 'AMZN', 'seq_length': 60, 'epochs': 50, 'batch_size': 128, 'units': 200, 'dropout': 0.40973999440459574, 'learning_rate': 0.0001, 'patience': 10, 'num_layers': 2, 'bidirectional': True, 'l2_reg': 0, 'lr_factor': 0.3, 'min_lr': 1e-05}

    # Ejecutar el entrenamiento con la mejor configuración
    run_lstm_training(experimento_final_AMZN['symbol'], experimento_final_AMZN)