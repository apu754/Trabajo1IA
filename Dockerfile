FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# montaremos artifacts/ como volumen al correr
ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host","0.0.0.0", "--port","8000"]