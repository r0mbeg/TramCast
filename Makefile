.DEFAULT_GOAL := help

DOCKER_COMPOSE ?= docker compose
GOOSE ?= goose
SQLC ?= sqlc
PROTOC ?= protoc
GO ?= go
NPM ?= npm
ENV_FILE ?= ../.env
PYTHON ?= python3

.PHONY: help env-up env-down env-port-forward env-port-close ps logs \
	migrate-up migrate-status migrate-down migrate-create migrate-validate \
	sqlc-compile sqlc-generate proto-generate-go \
	tidy-backend tidy-backend-check run-backend \
	proto-generate-python prepare-ml-model run-ml test-ml test-ml-service \
	import-catalog import-catalog-dry-run \
	frontend-install frontend-dev frontend-build frontend-test

help:
	@echo TramCast commands
	@echo   make env-up                 Start PostgreSQL and the local port-forwarder
	@echo   make env-down               Stop the environment and keep database volumes
	@echo   make env-port-forward       Open the PostgreSQL port for local development
	@echo   make env-port-close         Close the forwarded PostgreSQL port
	@echo   make ps                     Show Compose services
	@echo   make logs                   Follow PostgreSQL logs
	@echo   make migrate-up             Apply pending migrations
	@echo   make migrate-status         Show migration status
	@echo   make migrate-down           Roll back the last migration
	@echo   make migrate-create name=x  Create a local SQL migration template
	@echo   make migrate-validate       Validate local migration files
	@echo   make sqlc-compile           Check SQL queries and sqlc configuration
	@echo   make sqlc-generate          Generate Go database code
	@echo   make proto-generate-go      Generate Go messages and gRPC interfaces
	@echo   make tidy-backend           Sync backend go.mod and go.sum with imports
	@echo   make tidy-backend-check     Show required go.mod and go.sum changes without writing
	@echo   make run-backend            Run the Go backend with the root .env file
	@echo   make proto-generate-python  Generate Python messages and gRPC interfaces
	@echo   make prepare-ml-model       Download and verify the pinned TabPFN checkpoint
	@echo   make run-ml                 Serve recipe 030 with persistent inference cache
	@echo   make test-ml-service        Check the real gRPC service without ML dependencies
	@echo   make test-ml                Run serving and client checks

	@echo   make import-catalog         Import the workbook from CATALOG_FILE and the OSM snapshot from CATALOG_OSM_FILE
	@echo   make import-catalog-dry-run Check the workbook and OSM snapshot and roll the import back
	@echo   make frontend-install       Install frontend dependencies from package-lock.json
	@echo   make frontend-dev           Run the Vite dev server with /api proxied to :8080
	@echo   make frontend-build         Build frontend/dist, which the backend serves
	@echo   make frontend-test          Type-check and run frontend unit tests

env-up:
	@$(DOCKER_COMPOSE) up -d --wait postgres env-port-forwarder

env-down:
	@$(DOCKER_COMPOSE) --profile dev --profile tools down

env-port-forward:
	@$(DOCKER_COMPOSE) up -d --wait env-port-forwarder

env-port-close:
	@$(DOCKER_COMPOSE) stop env-port-forwarder

ps:
	@$(DOCKER_COMPOSE) ps --all

logs:
	@$(DOCKER_COMPOSE) logs --follow postgres

migrate-up:
	@$(DOCKER_COMPOSE) run --rm --name tramcast-migrate migrate up

migrate-status:
	@$(DOCKER_COMPOSE) run --rm --name tramcast-migrate migrate status

migrate-down:
	@$(DOCKER_COMPOSE) run --rm --name tramcast-migrate migrate down

migrate-create:
	$(if $(strip $(name)),,$(error Set name, for example: make migrate-create name=init_schema))
	@$(GOOSE) -dir backend/migrations create "$(strip $(name))" sql

migrate-validate:
	@$(GOOSE) -dir backend/migrations validate

sqlc-compile:
	@$(SQLC) -f backend/sqlc.yaml compile

sqlc-generate:
	@$(SQLC) -f backend/sqlc.yaml generate

proto-generate-go:
	@$(PROTOC) --proto_path=proto --go_out=backend --go_opt=module=github.com/r0mbeg/TramCast/backend --go-grpc_out=backend --go-grpc_opt=module=github.com/r0mbeg/TramCast/backend proto/tramcast/forecast/v1/forecast.proto

tidy-backend:
	@$(GO) -C backend mod tidy

tidy-backend-check:
	@$(GO) -C backend mod tidy -diff

run-backend:
	@$(GO) -C backend run ./cmd/tramcast -env-file "$(ENV_FILE)"

proto-generate-python:
	@mkdir -p ml/runtime/generated
	@$(PYTHON) -m grpc_tools.protoc --proto_path=proto --python_out=ml/runtime/generated --pyi_out=ml/runtime/generated --grpc_python_out=ml/runtime/generated proto/tramcast/forecast/v1/forecast.proto

prepare-ml-model:
	@cd ml/runtime && PYTHONPATH=generated $(PYTHON) recipe.py

run-ml:
	@cd ml/runtime && PYTHONPATH=generated $(PYTHON) service.py

test-ml-service:
	@cd ml/runtime && PYTHONPATH=generated $(PYTHON) -m tests

test-ml: test-ml-service

# The workbook path comes from CATALOG_FILE in the env file: a Cyrillic path
# passed through make is mangled in PowerShell.
import-catalog:
	@$(GO) -C backend run ./cmd/import-catalog -env-file "$(ENV_FILE)"

import-catalog-dry-run:
	@$(GO) -C backend run ./cmd/import-catalog -env-file "$(ENV_FILE)" -dry-run

frontend-install:
	@$(NPM) --prefix frontend ci

frontend-dev:
	@$(NPM) --prefix frontend run dev

frontend-build:
	@$(NPM) --prefix frontend run build

frontend-test:
	@$(NPM) --prefix frontend run typecheck
	@$(NPM) --prefix frontend test
