FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    P3_MODELS=/weights \
    HF_HOME=/weights/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_IMPLICIT_TOKEN=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt
COPY p3/ ./p3/
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "p3.app:app", "--host", "0.0.0.0", "--port", "7860"]
