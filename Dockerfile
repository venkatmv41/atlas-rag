FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/atlas/.cache/huggingface

WORKDIR /app

RUN useradd --create-home --uid 10001 atlas

COPY requirements.txt .
RUN python -u -m pip install \
    --no-cache-dir \
    --progress-bar on \
    --verbose \
    -r requirements.txt

COPY app/ ./app/
COPY data/sops/ ./data/sops/

RUN mkdir -p /app/chroma_db \
    && chown -R atlas:atlas /app /home/atlas

USER atlas

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]