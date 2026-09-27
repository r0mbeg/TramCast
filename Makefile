.DEFAULT_GOAL := help

DOCKER_COMPOSE ?= docker compose
GOOSE ?= goose
SQLC ?= sqlc
PROTOC ?= protoc
GO ?= go
NPM ?= npm
ENV_FILE ?= ../.env
PYTHON ?= python3

# Standalone ML delivery from ml/docs/HANDOFF.md. ML_MODE=gpu recomputes recipe
# 030 on an NVIDIA GPU; ML_MODE=replay serves the saved 030 result on CPU.
ML_COMPOSE ?= $(DOCKER_COMPOSE) -f ml/compose.yaml
ML_MODE ?= gpu
ML_ROUTE ?= 1
ML_FROM ?= 2025-11-01T00:00:00+03:00
ML_TO ?= 2026-01-01T00:00:00+03:00
ML_TIMEOUT ?= 930
ML_OTHER_MODE = $(if $(filter gpu,$(ML_MODE)),replay,gpu)

# Git Bash would otherwise rewrite container paths such as /tmp/... for Docker.
export MSYS_NO_PATHCONV := 1

ifneq ($(filter ml-up ml-check ml-logs ml-start ml-warm ml-describe,$(MAKECMDGOALS)),)
ifeq ($(filter gpu replay,$(ML_MODE)),)
$(error ML_MODE must be gpu or replay, got "$(ML_MODE)")
endif
endif
ifneq ($(filter ml-start ml-warm ml-describe,$(MAKECMDGOALS)),)
ifneq ($(ML_MODE),gpu)
$(error $(filter ml-start ml-warm ml-describe,$(MAKECMDGOALS)) needs ML_MODE=gpu: replay serves a saved result)
endif
endif

.PHONY: help app-build app-up app-down app-logs \
	env-up env-down env-port-forward env-port-close ps logs \
	migrate-up migrate-status migrate-down migrate-create migrate-validate \
	sqlc-compile sqlc-generate proto-generate-go \
	tidy-backend tidy-backend-check run-backend \
	proto-generate-python prepare-ml-model run-ml test-ml test-ml-service \
	ml-install ml-up ml-warm ml-start ml-check ml-describe ml-logs ml-down \
	import-catalog import-catalog-dry-run forecast-version-register \
	frontend-install frontend-dev frontend-build frontend-test

help:
	@echo TramCast commands
	@echo   make app-build              Build the application and migration images
	@echo   make app-up                 Build and start the app after migrations and initial catalog import
	@echo   make app-down               Stop the app environment and keep database files
	@echo   make app-logs               Follow backend, catalog initialization and migration logs
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
	@echo   make ml-install             Download and verify the pinned TabPFN weights into a Docker volume
	@echo   make ml-up                  Start the ML gRPC container on 127.0.0.1:50051, ML_MODE=gpu or replay
	@echo   make ml-warm                Compute the full forecast into the ML cache, gpu only
	@echo   make ml-start               Run ml-install, ml-up and ml-warm, gpu only
	@echo   make ml-check               Call Predict for ML_ROUTE over ML_FROM..ML_TO with the bundled client
	@echo   make ml-describe            Show the model and dataset versions, gpu only
	@echo   make ml-logs                Follow the logs of the ML container
	@echo   make ml-down                Stop both ML modes and keep weights and cache
	@echo   make import-catalog         Import the workbook from CATALOG_FILE and the OSM snapshot from CATALOG_OSM_FILE
	@echo   make import-catalog-dry-run Check the workbook and OSM snapshot and roll the import back
	@echo   make forecast-version-register METADATA=x  Register ML version metadata, ACTIVATE=1 DRY_RUN=1 ALLOW_REPLAY=1 VERIFY=0
	@echo   make frontend-install       Install frontend dependencies from package-lock.json
	@echo   make frontend-dev           Run the Vite dev server with /api proxied to :8080
	@echo   make frontend-build         Build frontend/dist, which the backend serves
	@echo   make frontend-test          Type-check and run frontend unit tests

app-build:
	@$(DOCKER_COMPOSE) build backend migrate

app-up:
	@$(DOCKER_COMPOSE) up -d --build --wait backend

app-down:
	@$(DOCKER_COMPOSE) --profile dev down

app-logs:
	@$(DOCKER_COMPOSE) logs --follow backend catalog-init migrate

env-up:
	@$(DOCKER_COMPOSE) up -d --wait postgres env-port-forwarder

env-down:
	@$(DOCKER_COMPOSE) --profile dev down

env-port-forward:
	@$(DOCKER_COMPOSE) up -d --wait env-port-forwarder

env-port-close:
	@$(DOCKER_COMPOSE) stop env-port-forwarder

ps:
	@$(DOCKER_COMPOSE) ps --all

logs:
	@$(DOCKER_COMPOSE) logs --follow postgres

migrate-up:
	@$(DOCKER_COMPOSE) run --rm migrate up

migrate-status:
	@$(DOCKER_COMPOSE) run --rm migrate status

migrate-down:
	@$(DOCKER_COMPOSE) run --rm migrate down

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

ml-install:
	@$(ML_COMPOSE) run --rm --build ml-install

# Both modes publish the same port, so the other mode is stopped first.
ml-up:
	@$(ML_COMPOSE) --profile $(ML_OTHER_MODE) stop ml-$(ML_OTHER_MODE)
	@$(ML_COMPOSE) --profile $(ML_MODE) up -d --build --wait ml-$(ML_MODE)

ml-warm:
	@$(ML_COMPOSE) --profile gpu exec -T ml-gpu python service.py --warm-cache

# The ml-start steps depend on each other. Make 3.81 applies .NOTPARALLEL to
# the whole file, so make -j still runs them in order.
.NOTPARALLEL:
ml-start: ml-install ml-up ml-warm

ml-check:
	@$(ML_COMPOSE) --profile $(ML_MODE) exec -T ml-$(ML_MODE) python client.py --route $(ML_ROUTE) --from $(ML_FROM) --to $(ML_TO) --timeout $(ML_TIMEOUT) --output /tmp/prediction.json

ml-describe:
	@$(ML_COMPOSE) --profile gpu exec -T ml-gpu python service.py --describe

ml-logs:
	@$(ML_COMPOSE) --profile $(ML_MODE) logs --tail 100 --follow ml-$(ML_MODE)

ml-down:
	@$(ML_COMPOSE) --profile replay --profile gpu down

# The workbook path comes from CATALOG_FILE in the env file: a Cyrillic path
# passed through make is mangled in PowerShell.
import-catalog:
	@$(GO) -C backend run ./cmd/import-catalog -env-file "$(ENV_FILE)"

import-catalog-dry-run:
	@$(GO) -C backend run ./cmd/import-catalog -env-file "$(ENV_FILE)" -dry-run

# METADATA is the JSON of ml-describe or of a replay bundle. go -C backend
# changes the working directory, so a relative path is prefixed with the
# repository root here; $(abspath) breaks drive letters in make 3.81.
METADATA_PATH = $(if $(filter /%,$(METADATA))$(findstring :,$(METADATA)),$(strip $(METADATA)),$(CURDIR)/$(strip $(METADATA)))

forecast-version-register:
	$(if $(strip $(METADATA)),,$(error Set METADATA, for example: make forecast-version-register METADATA=out/ml-describe.json))
	@$(GO) -C backend run ./cmd/register-forecast-version -env-file "$(ENV_FILE)" -metadata "$(METADATA_PATH)"$(if $(filter 1,$(ACTIVATE)), -activate)$(if $(filter 1,$(DRY_RUN)), -dry-run)$(if $(filter 1,$(ALLOW_REPLAY)), -allow-replay)$(if $(filter 0,$(VERIFY)), -verify=false)

frontend-install:
	@$(NPM) --prefix frontend ci

frontend-dev:
	@$(NPM) --prefix frontend run dev

frontend-build:
	@$(NPM) --prefix frontend run build

frontend-test:
	@$(NPM) --prefix frontend run typecheck
	@$(NPM) --prefix frontend test
