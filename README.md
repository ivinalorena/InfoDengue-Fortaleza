# InfoDengue-Fortaleza
##Previsão de Casos em Fortaleza-CE usando LSTM (Séries Temporais)
🔍 Descrição
Este repositório contém um projeto de previsão de casos de dengue no município de Fortaleza-CE para o ano de 2025, utilizando redes neurais LSTM (Long Short-Term Memory) para modelagem de séries temporais. O modelo foi treinado com dados históricos para capturar padrões sazonais e tendências, gerando projeções futuras com métricas de avaliação.
### Tecnologias e Bibliotecas
Python 3

*Bibliotecas:

1.TensorFlow/Keras (implementação da LSTM)

2.Pandas (manipulação de dados)

3.Matplotlib (visualização)

4.Scikit-learn (pré-processamento)

###📉 Gráficos:

Série histórica vs. prevista.
###📊 Métricas:
Total de amostras: 575|
-------------------------
Treinamento: 456 amostras (79.9%)
Teste: 115 amostras (20.1%)
----------------------------
Métricas de TREINAMENTO:
MAPE: 25.36% | RMSE: 160.31 | MAE: 88.68
-----------------------------------------
Métricas de TESTE:
MAPE: 23.81% | RMSE: 69.99 | MAE: 49.77
