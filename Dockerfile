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
    && go build -trimpath -ldflags="-s -w" -o /out/import-catalog ./cmd/import-catalog

# Both Compose application services use this image. Only catalog-init receives
# the external workbook mount; the workbook never enters the build context.
FROM debian:bookworm-slim AS app
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates curl tzdata \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=backend-build /out/tramcast /out/import-catalog ./
COPY --from=frontend-build /src/frontend/dist ./frontend/dist
COPY data/osm/tram_routes.json ./data/osm/tram_routes.json
ENV GIN_MODE=release \
    HTTP_ADDR=:8080 \
    WEB_DIR=/app/frontend/dist \
    CATALOG_FILE=/run/catalog/catalog.xlsx \
    CATALOG_OSM_FILE=/app/data/osm/tram_routes.json
USER 10001:10001
EXPOSE 8080
ENTRYPOINT ["/app/tramcast"]
