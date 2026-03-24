Com certeza! Preparei uma documentação completa e profissional para o seu repositório no GitHub. Ela está organizada para que qualquer pessoa (ou você mesmo no futuro) consiga entender, instalar e rodar o projeto rapidamente.
📈 Stock Predictor 5-Day API

Este projeto utiliza Redes Neurais Recorrentes do tipo LSTM (Long Short-Term Memory) para prever o preço de fechamento de ações (por padrão BBAS3.SA) com base nos últimos 5 dias de negociação. A solução é composta por um pipeline de coleta de dados, treinamento de modelo e uma API para servir as previsões.
🚀 Estrutura do Projeto

    config.py: Centraliza as configurações do projeto, como o Ticker da ação, datas e caminhos de arquivos.

    preprocessing.py: Gerencia a extração de dados via yfinance, normalização com MinMaxScaler e a criação de sequências temporais.

    treinamento.py: Define a arquitetura da rede neural, treina o modelo e salva os artefatos (.keras e .joblib).

    app.py: Interface de API construída com FastAPI para realizar predições em tempo real.

🛠️ Tecnologias Utilizadas

    Python 3.x

    TensorFlow/Keras: Para construção e treinamento do modelo de Deep Learning.

    FastAPI / Uvicorn: Para a criação da API de alto desempenho.

    Scikit-Learn: Para o escalonamento e normalização dos dados.

    yfinance: Para obtenção de dados históricos do Yahoo Finance.

    Pandas/NumPy: Para manipulação de dados numéricos.

⚙️ Como Executar
1. Instalação das Dependências

Primeiro, instale as bibliotecas necessárias:
Bash

pip install tensorflow fastapi uvicorn yfinance scikit-learn joblib pandas numpy

2. Treinamento do Modelo

Antes de rodar a API, você precisa gerar o modelo treinado. Execute o script de treinamento:
Bash

python treinamento.py

Isso criará os arquivos lstm_stock_model.keras e scaler.joblib.
3. Iniciando a API

Com o modelo treinado, suba o servidor:
Bash

python app.py

A API estará disponível em http://127.0.0.1:8000.
📡 Documentação da API
Endpoint de Predição

POST /predict

Corpo da Requisição (JSON):
Você deve enviar uma lista com, no mínimo, os últimos 5 preços de fechamento.
JSON

{
  "prices": [25.50, 25.80, 26.10, 25.90, 26.05]
}

Exemplo de Resposta:
JSON

{
  "ticker": "BBAS3.SA",
  "input_used": [25.50, 25.80, 26.10, 25.90, 26.05],
  "prediction_next_day": 26.25
}

📝 Notas de Implementação

    Arquitetura do Modelo: O modelo utiliza duas camadas LSTM com 50 unidades cada e camadas de Dropout(0.2) para evitar overfitting.

    Escalonamento: É fundamental que os dados de entrada na API passem pelo mesmo scaler utilizado no treinamento para garantir a precisão.

    Configuração: Para trocar a ação ou o intervalo de dias, basta alterar as variáveis no arquivo config.py.
