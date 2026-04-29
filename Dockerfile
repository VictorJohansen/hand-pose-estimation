FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PORT=7860

WORKDIR /app

COPY requirements/base.txt requirements/base.txt
COPY requirements/deploy.txt requirements/deploy.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements/deploy.txt

COPY src src
COPY models models
COPY README.md README.md

EXPOSE 7860

CMD uvicorn src.webapp.app:app --host 0.0.0.0 --port ${PORT}

