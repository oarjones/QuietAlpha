import joblib
import tensorflow as tf
import keras as ke
import sys
import os
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

Sequential = ke.models.Sequential
LSTM = ke.layers.LSTM
Dense = ke.layers.Dense
Dropout = ke.layers.Dropout
Bidirectional = ke.layers.Bidirectional
Adam = ke.optimizers.Adam
EarlyStopping = ke.callbacks.EarlyStopping
ReduceLROnPlateau = ke.callbacks.ReduceLROnPlateau


from data_processing import load_and_preprocess

def build_lstm_model(input_shape, config):
    """Construye el modelo LSTM, ahora usando un diccionario de configuración."""
    #model = Sequential()
    model = Sequential([
        ke.Input(shape=input_shape)  # Explicit input layer as first layer
    ])

    # Capa LSTM Bidireccional (si está habilitada en la configuración)
    # if config.get('bidirectional', False):  # Valor por defecto: False
    #     model.add(Bidirectional(LSTM(units=config['units'], return_sequences=True, input_shape=input_shape,
    #                                  kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
    #                                  )))
    # else:
    #     model.add(LSTM(units=config['units'], return_sequences=True, input_shape=input_shape,
    #                    kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
    #                    ))

    # Capa LSTM Bidireccional (si está habilitada en la configuración)
    if config.get('bidirectional', False):  # Valor por defecto: False
        model.add(Bidirectional(LSTM(units=config['units'], return_sequences=True,
                                   kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                                   )))
    else:
        model.add(LSTM(units=config['units'], return_sequences=True,
                      kernel_regularizer=ke.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                      ))
        

    model.add(Dropout(config['dropout']))

    # Capas LSTM adicionales (según la configuración)
    for _ in range(config.get('num_layers', 2) - 1):  # Valor por defecto: 2 capas
        if config.get('bidirectional', False):
            model.add(Bidirectional(LSTM(units=config['units'], return_sequences=True,
                                         kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                                         )))
        else:
            model.add(LSTM(units=config['units'], return_sequences=True,
                           kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                           ))
        model.add(Dropout(config['dropout']))

    # Ultima capa LSTM (no devuelve secuencias)
    if config.get('bidirectional', False):
        model.add(Bidirectional(LSTM(units=config['units'],
                                     kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                                     )))
    else:
        model.add(LSTM(units=config['units'],
                       kernel_regularizer=tf.keras.regularizers.l2(config.get('l2_reg', 0)) #Regularizacion L2
                       ))
    model.add(Dropout(config['dropout']))

    model.add(Dense(units=1))

    optimizer = Adam(learning_rate=config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train, x_test, y_test, config):
    """Entrena el modelo LSTM, usando la configuración."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=config['patience'], restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config.get('lr_factor', 0.2), patience=config['patience'] // 2, min_lr=config.get('min_lr', 0.00001))

    history = model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'],
                        validation_data=(x_test, y_test), shuffle=False,
                        callbacks=[early_stopping, reduce_lr])
    return history

def evaluate_model(model, x_test, y_test, scaler=None, y_train=None):
    """Evalúa el modelo LSTM."""
    y_pred = model.predict(x_test)

    if scaler:
        y_test_descaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :-1],y_test.reshape(-1, 1)], axis=1))[:, -1]
        y_pred_descaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :-1],y_pred.reshape(-1, 1)], axis=1))[:, -1]
    else:
        y_test_descaled = y_test
        y_pred_descaled = y_pred

    rmse = np.sqrt(mean_squared_error(y_test_descaled, y_pred_descaled))
    mae = mean_absolute_error(y_test_descaled, y_pred_descaled)
    r2 = r2_score(y_test_descaled, y_pred_descaled)

    return rmse, mae, r2

def plot_results(history, y_test_descaled, y_pred_descaled, experiment_label):
    """Genera y guarda los gráficos de resultados."""
    # Gráfico de Pérdida
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title(f"Pérdida - {experiment_label}")
    plt.savefig(f"loss_{experiment_label}.png")  # Guarda el gráfico
    plt.close()

    # Gráfico de Predicciones vs. Reales
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_descaled, label='Real')
    plt.plot(y_pred_descaled, label='Predicción')
    plt.legend()
    plt.title(f"Predicciones vs. Reales - {experiment_label}")
    plt.savefig(f"predictions_{experiment_label}.png")  # Guarda el gráfico
    plt.close()


def save_experiment_results(experiment_label, rmse, mae, r2, results_df_path="experiment_results.csv"):
    """Guarda los resultados del experimento en un DataFrame y un archivo CSV."""
    new_results = pd.DataFrame({
        'experiment': [experiment_label],
        'rmse': [rmse],
        'mae': [mae],
        'r2': [r2]
    })

    # Carga resultados anteriores (si existen)
    if os.path.exists(results_df_path):
        results_df = pd.read_csv(results_df_path)
        results_df = pd.concat([results_df, new_results], ignore_index=True)
    else:
        results_df = new_results

    # Guarda en CSV
    results_df.to_csv(results_df_path, index=False)
    return results_df

def load_experiment_results(results_df_path="experiment_results.csv"):
    """Carga el DataFrame de resultados desde un archivo CSV."""
    if os.path.exists(results_df_path):
        return pd.read_csv(results_df_path)
    else:
        return pd.DataFrame()

def run_lstm_training(symbol, config, data_dir="data"):
    """Función principal para cargar, entrenar, evaluar y guardar, con configuración."""

    # 1. Cargar y Preprocesar Datos
    try:
        x_train, y_train, x_test, y_test, _, _, _, feature_cols = load_and_preprocess(symbol, data_dir, config['seq_length'])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return

    # 2. Escalar datos
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    # 3. Construir el Modelo
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_lstm_model(input_shape, config)

    # 4. Entrenar el Modelo
    history = train_model(model, x_train, y_train, x_test, y_test, config)

    # 5. Evaluar el Modelo
    rmse, mae, r2 = evaluate_model(model, x_test, y_test, scaler)

    # 6. Desescalar para graficos
    y_pred = model.predict(x_test)
    y_test_descaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :-1],y_test.reshape(-1, 1)], axis=1))[:, -1]
    y_pred_descaled = scaler.inverse_transform(np.concatenate([x_test[:, -1, :-1],y_pred.reshape(-1, 1)], axis=1))[:, -1]

    # 7. Generar gráficos
    plot_results(history, y_test_descaled, y_pred_descaled, config['label'])

    # 8. Guardar Resultados (DataFrame)
    results_df = save_experiment_results(config['label'], rmse, mae, r2)
    print(results_df)

    # 9. Guardar Modelo y Escalador
    #model.save(f"lstm_model_{config['label']}.keras")
    ke.saving.save_model(model, 'lstm_model.keras')

    joblib.dump(scaler, f"scaler_{config['label']}.pkl")
    print(f"Modelo y escalador guardados para {config['label']}.")


def run_lstm_training_with_score(symbol, config, data_dir="data"):
    """Función para cargar, entrenar, evaluar y devolver el score del modelo."""

    # 1. Cargar y Preprocesar Datos
    try:
        x_train, y_train, x_test, y_test, _, _, _, feature_cols = load_and_preprocess(symbol, data_dir, config['seq_length'])
    except (FileNotFoundError, ValueError) as e:
        print(e)
        return float('inf')  # Devolver un score alto en caso de error

    # 2. Escalar datos
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train.reshape(-1, x_train.shape[-1])).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(-1, x_test.shape[-1])).reshape(x_test.shape)

    # 3. Construir el Modelo
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_lstm_model(input_shape, config)

    # 4. Entrenar el Modelo
    history = train_model(model, x_train, y_train, x_test, y_test, config)

    # 5. Evaluar el Modelo
    rmse, mae, r2 = evaluate_model(model, x_test, y_test, scaler)

    return rmse  # Devolver el RMSE como score


# El bloque if __name__ == '__main__':  se moverá a un script separado (ver abajo)