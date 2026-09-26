.DEFAULT_GOAL := help

DOCKER_COMPOSE ?= docker compose
GOOSE ?= goose
SQLC ?= sqlc

.PHONY: help env-up env-down env-port-forward env-port-close ps logs \
	migrate-up migrate-status migrate-down migrate-create migrate-validate \
	sqlc-compile sqlc-generate

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
