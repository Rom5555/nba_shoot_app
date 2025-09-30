FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV STREAMLIT_SERVER_HEADLESS=true
ENV TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Installer PyTorch CPU compatible Python 3.12
RUN pip install torch==2.2.2


# Installer le reste des dépendances
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models deep_models

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
