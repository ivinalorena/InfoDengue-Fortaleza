# lstm_dengue_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.metrics import RootMeanSquaredError, mean_absolute_error
from sklearn.metrics import mean_squared_error

def carregar_dados(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df_modificado = df.loc[:, ['casos_est']].copy()
    datas = df.loc[:, ['data_iniSE']].copy()
    return df_modificado, datas

def normalizar_dados(df):
    normalizador = MinMaxScaler(feature_range=(0,1))
    df_normalizado = normalizador.fit_transform(df)
    return df_normalizado, normalizador

def criar_janelas(df_normalizado, window_size):
    previsao, valor_real = [], []
    for i in range(window_size, len(df_normalizado)):
        janela = df_normalizado[i-window_size:i, 0]
        previsao.append(janela)
        valor_real.append(df_normalizado[i, 0])
    previsao = np.array(previsao)
    valor_real = np.array(valor_real)
    previsao = np.reshape(previsao, (previsao.shape[0], previsao.shape[1], 1))
    return previsao, valor_real

def dividir_dados(previsao, valor_real, proporcao_treino=0.8):
    tam_treinamento = int(len(previsao) * proporcao_treino)
    X_treinamento = previsao[:tam_treinamento]
    x_teste = previsao[tam_treinamento:]
    y_treinamento = valor_real[:tam_treinamento]
    y_teste = valor_real[tam_treinamento:]
    return X_treinamento, x_teste, y_treinamento, y_teste, tam_treinamento

def criar_modelo(input_shape):
    modelo = Sequential()
    modelo.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=50, return_sequences=True))
    modelo.add(Dropout(0.3))
    modelo.add(LSTM(units=50))
    modelo.add(Dropout(0.3))
    modelo.add(Dense(units=1))
    modelo.compile(optimizer='adam', loss='mean_squared_error',
                   metrics=['mean_absolute_error', RootMeanSquaredError()])
    return modelo

def avaliar_modelo(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    return mae, rmse, mape

def plotar_resultados(df_normalizado, previsao_treinamento, previsao_teste, tam_treinamento, window_size, normalizador):
    plt.figure(figsize=(14, 6))
    plt.plot(normalizador.inverse_transform(df_normalizado), color='blue', alpha=0.3, label='Dados Reais')
    train_range = range(window_size, window_size + len(previsao_treinamento))
    plt.plot(train_range, previsao_treinamento, color='orange', label='Previsão (Treino)')
    test_start = window_size + tam_treinamento
    test_range = range(test_start, test_start + len(previsao_teste))
    plt.plot(test_range, previsao_teste, color='red',  label='Previsão (Teste)')
    plt.axvline(x=tam_treinamento + window_size, color='black', linestyle='--', label='Divisão Treino/Teste')
    plt.title('Comparacão: Previsões no Treino e Teste')
    plt.xlabel('Período')
    plt.ylabel('Casos de Dengue')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    df_modificado, datas = carregar_dados("Fortaleza-Dengue.csv")
    print('\nComo estão os dados:\n', datas)

    df_normalizado, normalizador = normalizar_dados(df_modificado)
    window_size = 4  # 4 semanas (~1 mês)
    previsao, valor_real = criar_janelas(df_normalizado, window_size)
    X_treinamento, x_teste, y_treinamento, y_teste, tam_treinamento = dividir_dados(previsao, valor_real)

    modelo = criar_modelo((previsao.shape[1], 1))
    modelo.fit(X_treinamento, y_treinamento, batch_size=32, epochs=200, verbose=1)

    previsao_treinamento = modelo.predict(X_treinamento)
    previsao_teste = modelo.predict(x_teste)

    previsao_treinamento = normalizador.inverse_transform(previsao_treinamento)
    y_treinamento_desnormalizado = normalizador.inverse_transform(y_treinamento.reshape(-1, 1))
    previsao_teste = normalizador.inverse_transform(previsao_teste)
    y_teste_desnormalizado = normalizador.inverse_transform(y_teste.reshape(-1, 1))

    mae_train, rmse_train, mape_train = avaliar_modelo(y_treinamento_desnormalizado, previsao_treinamento)
    mae_test, rmse_test, mape_test = avaliar_modelo(y_teste_desnormalizado, previsao_teste)

    print(f"\nTotal de amostras: {len(df_normalizado)}")
    print(f"Treinamento: {len(X_treinamento)} amostras ({len(X_treinamento)/len(df_normalizado):.1%})")
    print(f"Teste: {len(x_teste)} amostras ({len(x_teste)/len(df_normalizado):.1%})")

    print("\nMétricas de TREINAMENTO:")
    print(f"MAPE (Treino): {mape_train:.2f}%")
    print(f"RMSE (Treino): {rmse_train:.2f}")

    print("\nMétricas de TESTE:")
    print(f"MAPE (Teste): {mape_test:.2f}%")
    print(f"RMSE (Teste): {rmse_test:.2f}")

    plotar_resultados(df_normalizado, previsao_treinamento, previsao_teste, tam_treinamento, window_size, normalizador)

if __name__ == "__main__":
    main()
