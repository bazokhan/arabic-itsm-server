FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    HOST=0.0.0.0 \
    DEVICE=cpu \
    MODEL_DIRS=/data/models \
    HF_HOME=/data/hf_cache \
    DEFAULT_MODEL_ID=marbert-arabic-itsm-l3-categories \
    HF_MODEL_REPOS=albaz2000/marbert-arabic-itsm-l3-categories,albaz2000/arabert-arabic-itsm-l3-categories,albaz2000/byt5-arabic-itsm-l3-categories,albaz2000/egybert-arabic-itsm-l3-categories \
    DB_PATH=/data/db/itsm.db

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip && \
    pip install torch --index-url "${TORCH_INDEX_URL}" && \
    pip install -r /app/requirements.txt

COPY . /app

RUN chmod +x /app/scripts/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]
