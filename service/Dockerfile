FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

COPY service/ ./service/
COPY quant/ ./quant/
COPY pipeline/ ./pipeline/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app
EXPOSE 8088

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8088", "--workers", "1"]
