FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV HOST=0.0.0.0
ENV PORT=8080
ENV MODEL_PATH=models/webcam-model/best.keras
ENV OPEN_HAND_THRESHOLD=1.55
ENV SERVICE_VERSION=local

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
COPY requirements/deploy.txt requirements/deploy.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements/deploy.txt

COPY src src
COPY models/webcam-model/best.keras models/webcam-model/best.keras
COPY README.md README.md

EXPOSE 8080

CMD ["sh", "-c", "python -m uvicorn src.webapp.app:app --host ${HOST} --port ${PORT}"]

