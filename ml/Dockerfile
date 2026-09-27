FROM python:3.12-slim AS dependencies
WORKDIR /app/ml/runtime
COPY ml/runtime/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app/ml/runtime/generated PYTHONUNBUFFERED=1
EXPOSE 50051

FROM dependencies AS serving
COPY ml/runtime/*.py ./
COPY ml/runtime/generated ./generated
COPY ml/bundles/002 ../bundles/002
COPY ml/bundles/030 ../bundles/030
COPY ml/recipes/tabpfn-030 ../recipes/tabpfn-030
USER 65534:65534
CMD ["python", "service.py", "--bind", "0.0.0.0:50051", "--bundle", "/app/ml/bundles/002/forecast_bundle.json"]

FROM dependencies AS inference
COPY ml/runtime/requirements-inference.txt ./
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements-inference.txt
COPY ml/runtime/*.py ./
COPY ml/runtime/generated ./generated
COPY ml/recipes/tabpfn-030 ../recipes/tabpfn-030
RUN mkdir -p /app/ml/cache && chown 65534:65534 /app/ml/cache
USER 65534:65534
CMD ["python", "service.py", "--bind", "0.0.0.0:50051"]
