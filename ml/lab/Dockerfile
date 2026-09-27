FROM python:3.12-slim
WORKDIR /app/ml
COPY ml/requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt
COPY ml/generated ./generated
COPY ml/service.py ml/constants.py ml/forecast_bundle.json ./
COPY ml/submissions/002_chronos_daily_zero_shot.csv ./submissions/
ENV PYTHONPATH=/app/ml/generated PYTHONUNBUFFERED=1
USER 65534:65534
EXPOSE 50051
CMD ["python", "service.py", "--bind", "0.0.0.0:50051"]
