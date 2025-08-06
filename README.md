# InfoDengue-Fortaleza
Previsão de Casos de Dengue em Fortaleza-CE (2025) usando LSTM

* 🔍 Descrição
Projeto de previsão de casos de dengue em Fortaleza-CE para 2025, utilizando redes neurais LSTM (Long Short-Term Memory) para modelagem de séries temporais. O modelo foi treinado com dados históricos para capturar padrões sazonais e tendências, gerando projeções com métricas de avaliação robustas.

# Tecnologias e Bibliotecas
Python 3

* Bibliotecas Principais:

🧠 TensorFlow/Keras (implementação da LSTM).

📊 Pandas (manipulação de dados).

📈 Matplotlib/Seaborn (visualização).

⚙️ Scikit-learn (pré-processamento e validação).

# Metodologia
1. Coleta e Pré-processamento:

Dados históricos de casos de dengue em Fortaleza.

2. Tratamento de valores faltantes e normalização (ex.: MinMaxScaler).

3. Modelagem LSTM:

Divisão dos dados: Treino (79.9%) e Teste (20.1%).

Arquitetura da rede:

4. Camadas LSTM com ativação tanh.

Dropout para evitar overfitting.

Otimizador: Adam.

#Avaliação:

Métricas calculadas para treino e teste (MAPE, RMSE, MAE).

#Resultados
📉 Gráficos
Série histórica vs. prevista (treino e teste).

Projeção para 2025 (com intervalo de confiança, se aplicável).

#📊 Métricas
Conjunto  |	MAPE  | RMSE	| MAE
--------  |--------|------|-----  
Treino	|  25.36%  |	160.31	| 88.68
--------| --------|---------|-------
Teste	  | 23.81%	|69.99	| 49.77

*Dados do Modelo:
Total de amostras: 575
Treinamento: 456 amostras (79.9%).
Teste: 115 amostras (20.1%).

