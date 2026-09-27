# syntax=docker/dockerfile:1

FROM node:24.14.1-bookworm-slim AS frontend-build
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM golang:1.26.8-bookworm AS backend-build
ENV CGO_ENABLED=0 GOTOOLCHAIN=local
WORKDIR /src/backend
COPY backend/go.mod backend/go.sum ./
RUN go mod download
COPY backend/ ./
RUN go build -trimpath -ldflags="-s -w" -o /out/tramcast ./cmd/tramcast \
    && go build -trimpath -ldflags="-s -w" -o /out/import-catalog ./cmd/import-catalog \
    && go build -trimpath -ldflags="-s -w" -o /out/register-forecast-version ./cmd/register-forecast-version \
    && go build -trimpath -ldflags="-s -w" -o /out/enqueue-prediction-jobs ./cmd/enqueue-prediction-jobs

# Every Compose application service uses this image with the sanitized catalog:
# the server, the catalog import, version registration and job admission.
FROM debian:bookworm-slim AS app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=backend-build /out/tramcast /out/import-catalog /out/register-forecast-version /out/enqueue-prediction-jobs ./
COPY --from=frontend-build /src/frontend/dist ./frontend/dist
COPY data/catalog/catalog.xlsx ./data/catalog/catalog.xlsx
COPY data/osm/tram_routes.json ./data/osm/tram_routes.json
ENV GIN_MODE=release \
    HTTP_ADDR=:8080 \
    WEB_DIR=/app/frontend/dist \
    CATALOG_FILE=/app/data/catalog/catalog.xlsx \
    CATALOG_OSM_FILE=/app/data/osm/tram_routes.json
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/app/tramcast"]
