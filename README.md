# ObesityPredict — Tech Challenge Fase 4 · POSTECH

Sistema preditivo hospitalar para classificação de níveis de obesidade.

## Setup rápido (local)

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Configurar ambiente
cp .env.example .env
# edite .env com suas credenciais Supabase

# 3. Criar tabelas no Supabase
# Execute db/create_tables.sql no SQL Editor do Supabase

# 4. Ingerir dados CSV
python -m db.ingest_csv          # requer data/Obesity.csv

# 5. Treinar o modelo
python -m ml.train               # gera ml/model.pkl, ml/label_encoder.pkl e ml/metrics.json

# 6. Rodar o front-end
python -m streamlit run app/Home.py
```

## Docker

```bash
# Build da imagem
docker build -t obesity_predict .

# Rodar com docker-compose (lê credenciais do .env automaticamente)
docker-compose up --build

# Acessar em: http://localhost:8501
```

> O arquivo `.env` **não é copiado para dentro da imagem** — as variáveis são
> injetadas em runtime pelo `docker-compose` via `env_file`.

## Estrutura

```
obesity_predict/
├── .github/
│   └── workflows/
│       └── ci.yml             # pipeline de CI/CD (GitHub Actions)
├── app/
│   ├── pages/
│   │   ├── 0_Analise.py       # análise exploratória dos dados (EDA)
│   │   ├── 1_Previsao.py      # formulário + IMC em tempo real + gauge + log
│   │   ├── 2_Logs.py          # tabela + gráficos de histórico
│   │   └── 3_Sobre.py         # arquitetura, métricas, limitações
│   └── Home.py                # entry point Streamlit
├── ml/
│   ├── train.py               # pipeline completo de treino
│   ├── predict.py             # helpers de inferência
│   ├── model.pkl              # artefato gerado (git-ignored)
│   ├── label_encoder.pkl      # encoder das classes (git-ignored)
│   └── metrics.json           # métricas de avaliação (git-ignored)
├── data/
│   └── Obesity.csv            # dataset original (git-ignored)
├── db/
│   ├── supabase_client.py     # singleton do cliente Supabase
│   ├── ingest_csv.py          # carga do CSV para obesity_data
│   └── create_tables.sql      # DDL das tabelas
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

## Deploy (Streamlit Community Cloud)

1. Faça push do repositório para o GitHub
2. Em [share.streamlit.io](https://share.streamlit.io), conecte o repo
3. Configure `SUPABASE_URL` e `SUPABASE_KEY` em **Secrets**
4. Defina o main file como `app/Home.py`

> `model.pkl` e `label_encoder.pkl` devem ser commitados ou gerados via
> script de build antes do deploy.
