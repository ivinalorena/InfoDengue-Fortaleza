#previsões até 106 semanas (2 anos)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("Caminho_do_CSV")
df=df.sort_values(by='data_iniSE')

df_modificado = df[['casos_est']].copy()
datas = df[['data_iniSE']].copy()


#print(df)

normalizador = MinMaxScaler(feature_range=(0, 1))
df_normalizado = normalizador.fit_transform(df_modificado)


window_size = 4  # 4 semanas (~1 mês)
previsao = []
valor_real = []

for i in range(window_size, len(df_normalizado)):
    janela = df_normalizado[i-window_size:i, 0]
    previsao.append(janela)
    valor_real.append(df_normalizado[i, 0])


previsao = np.array(previsao)
valor_real = np.array(valor_real)


previsao = np.reshape(previsao, (previsao.shape[0], previsao.shape[1], 1))


tam_treinamento = int(len(previsao) * 0.8)
X_treinamento = previsao[:tam_treinamento]
y_treinamento = valor_real[:tam_treinamento]
x_teste = previsao[tam_treinamento:]
y_teste = valor_real[tam_treinamento:]


modelo = Sequential()
modelo.add(LSTM(units=100, return_sequences=True, input_shape=(previsao.shape[1], 1)))
modelo.add(Dropout(0.3))

modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.3))

modelo.add(LSTM(units=50))
modelo.add(Dropout(0.3))
modelo.add(Dense(units=1))
modelo.add(acti)
modelo.compile(optimizer='adam',
               loss='mean_squared_error',
               metrics=[MeanAbsoluteError(), RootMeanSquaredError()])

modelo.fit(X_treinamento, y_treinamento, batch_size=32, epochs=100, verbose=1)


keras.saving.save_model(modelo, 'LSTM_dengue.keras')


previsao_treinamento_lstm = modelo.predict(X_treinamento)
previsao_treinamento_desnormalizada = normalizador.inverse_transform(previsao_treinamento_lstm)
y_treinamento_desnormalizado = normalizador.inverse_transform(y_treinamento.reshape(-1, 1))


previsao_lstm = modelo.predict(x_teste)
previsao_teste_desnormalizada = normalizador.inverse_transform(previsao_lstm)
y_teste_desnormalizado = normalizador.inverse_transform(y_teste.reshape(-1, 1))


previsao_futuro = modelo.predict(previsao)



mae_treinamento = mean_absolute_error(y_treinamento_desnormalizado, previsao_treinamento_desnormalizada)
rmse_treinamento = np.sqrt(mean_squared_error(y_treinamento_desnormalizado, previsao_treinamento_desnormalizada))
mape_treinamento = np.mean(np.abs((y_treinamento_desnormalizado - previsao_treinamento_desnormalizada) / y_treinamento_desnormalizado)) * 100

mae_teste = mean_absolute_error(y_teste_desnormalizado, previsao_teste_desnormalizada)
rmse_teste = np.sqrt(mean_squared_error(y_teste_desnormalizado, previsao_teste_desnormalizada))
mape_teste = np.mean(np.abs((y_teste_desnormalizado - previsao_teste_desnormalizada) / y_teste_desnormalizado)) * 100

print(f"\nTotal de amostras: {len(df_normalizado)}")
print(f"Treinamento: {len(X_treinamento)} amostras ({len(X_treinamento)/len(previsao):.1%})")
print(f"Teste: {len(x_teste)} amostras ({len(x_teste)/len(previsao):.1%})")

print("\nMétricas de TREINAMENTO:")
print(f"MAPE: {mape_treinamento:.2f}% | RMSE: {rmse_treinamento:.2f} | MAE: {mae_treinamento:.2f}")

print("\nMétricas de TESTE:")
print(f"MAPE: {mape_teste:.2f}% | RMSE: {rmse_teste:.2f} | MAE: {mae_teste:.2f}")

n = 52 # 1 ano
janela_atual = df_normalizado[-window_size:].reshape(1, window_size, 1)
previsoes_futuras = []

for _ in range(n):
    proxima_pred = modelo.predict(janela_atual, verbose=0)[0][0]
    previsoes_futuras.append(proxima_pred)
    janela_atual = np.append(janela_atual[:, 1:, :], [[[proxima_pred]]], axis=1)

previsoes_futuras = np.array(previsoes_futuras).reshape(-1, 1)
previsoes_futuras_desnormalizadas = normalizador.inverse_transform(previsoes_futuras)

ultima_data = pd.to_datetime(datas.iloc[-1, 0])
datas_futuras = [ultima_data + datetime.timedelta(weeks=i+1) for i in range(n)]

plt.figure(figsize=(16, 6))


plt.plot(normalizador.inverse_transform(df_normalizado), color='blue', alpha=0.3, label='Dados Reais')


train_range = range(window_size, window_size + len(previsao_treinamento_desnormalizada))
plt.plot(train_range, previsao_treinamento_desnormalizada, color='orange', label='Previsão (Treino)')

test_start = window_size + tam_treinamento
test_range = range(test_start, test_start + len(previsao_teste_desnormalizada))
plt.plot(test_range, previsao_teste_desnormalizada, color='red', label='Previsão (Teste)')


futuro_start = len(df_normalizado)
futuro_range = range(futuro_start, futuro_start + len(previsoes_futuras_desnormalizadas))
plt.plot(futuro_range, previsoes_futuras_desnormalizadas, color='green', linestyle='--', label='Previsão Futura (106 semanas)')

plt.axvline(x=tam_treinamento + window_size, color='black', linestyle='--', label='Divisão Treino/Teste')

plt.axvline(x=len(df_normalizado)-1, color='gray', linestyle='--', label='Início Previsão Futura')

plt.title('Casos de Dengue em Fortaleza: Histórico e Previsões (LSTM)')
plt.xlabel('Semanas')
plt.ylabel('Casos Estimados')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

previsoes_futuras_df = pd.DataFrame({
    'Data':datas_futuras,
    'Previsão': previsoes_futuras_desnormalizadas.flatten()
})
previsoes_futuras_df.to_csv('previsoes_futuras_dengue.csv', index=False)
