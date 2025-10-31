FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# 1) Mettre pip à jour
RUN python -m pip install --upgrade pip

# 2) Installer Torch CPU depuis l’index PyTorch (en plus de PyPI)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.4.*

# 3) Installer le reste depuis PyPI
RUN pip install --no-cache-dir -r requirements.txt

# 4) Copier le code (tick_trade.py, data.py, ppo_intraday_longflat.zip, etc.)
COPY . /app

ENTRYPOINT ["python","tick_trade.py"]
