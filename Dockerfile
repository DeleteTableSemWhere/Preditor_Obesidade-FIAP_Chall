# Imagem base leve com Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Instala dependências antes de copiar o código (aproveita cache de camadas)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY app/  ./app/
COPY db/   ./db/
COPY ml/   ./ml/

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Variáveis de ambiente lidas em runtime via docker-compose ou -e
ENV SUPABASE_URL=""
ENV SUPABASE_KEY=""

# Streamlit precisa de --server.address=0.0.0.0 para ser acessível fora do container
CMD ["python", "-m", "streamlit", "run", "app/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
