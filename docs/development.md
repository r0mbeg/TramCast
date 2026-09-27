# Локальная разработка TramCast

PostgreSQL запускается в Docker Compose, Go-backend — локально через `make run-backend`, фронтенд собирается локально через npm. Контейнер backend добавляется следующим этапом. Python ML-сервис уже доступен отдельно; инструкции — в [ml/README.md](../ml/README.md). Redis и Caddy в текущем окружении не нужны.

## Настройка

Нужны Go версии из `backend/go.mod` (1.26.8), Docker с Compose и для фронтенда Node.js 24 с npm 11 (проверено на Node 24.14.1). Make удобен для коротких команд; для SQL-генерации и создания миграций используются установленные `sqlc` и `goose`.

При первом запуске скопируйте `.env.example` в `.env` и задайте непустой пароль PostgreSQL. В PowerShell:

```powershell
Copy-Item .env.example .env
```

Существующий `.env` сохраняйте. Он исключён из Git. Compose сам читает этот файл; `make run-backend` передаёт его Go-приложению через `-env-file`. `PROJECT_ROOT` и отдельный экспорт переменных не требуются. При запуске из WSL инструменты Go/Make должны быть доступны в `PATH` этого окружения.

## Запуск

Из корня проекта:

```text
make env-up
make migrate-up
make migrate-status
make import-catalog
make run-backend
```

Без Make те же действия:

```text
docker compose up -d --wait postgres env-port-forwarder
docker compose run --rm --name tramcast-migrate migrate up
docker compose run --rm --name tramcast-migrate migrate status
go -C backend run ./cmd/import-catalog -env-file ../.env
go -C backend run ./cmd/tramcast -env-file ../.env
```

`make import-catalog` загружает маршруты, остановки и порядок остановок из книги справочников по пути `CATALOG_FILE` и из снимка OpenStreetMap по пути `CATALOG_OSM_FILE` для маршрутов 17, 25, 26, 28, 50, которых в книге нет. Книгу из `dataset/` каждый участник размещает сам. Снимок `data/osm/tram_routes.json` хранится в репозитории, и значение `CATALOG_OSM_FILE` по умолчанию указывает на него; на сервере файл можно положить в любое место, например рядом с книгой, и задать путь в `.env`. Импорт нужен один раз на базу и повторяется при новой версии книги или снимка. `make import-catalog-dry-run` проверяет оба источника и показывает итог, откатывая изменения. При ошибках команда перечисляет их — в книге по листам и строкам, в снимке по relation и участникам — и ничего не записывает. После импорта обновите открытую страницу интерфейса: справочники кешируются в браузере до перезагрузки, перезапуск backend не нужен. Подробности — в [backend/README.md](../backend/README.md#импорт-справочников), порядок обновления снимка — в [его README](../data/osm/README.md).

| Контейнер | Назначение |
| --- | --- |
| `tramcast-postgres` | Отдельная PostgreSQL 18.1 для проекта; постоянные данные в `out/pgdata` |
| `tramcast-env-port-forwarder` | Необязательный доступ с хоста через `127.0.0.1:5433`; профиль `dev` |
| `tramcast-migrate` | Одноразовый запуск goose; профиль `tools`, удаляется после команды |

`make env-up` поднимает PostgreSQL и локальный forwarder вместе. Для повторного открытия порта после `make env-port-close` используйте `make env-port-forward`. При явном запуске сервиса его Compose-профиль включается автоматически. Миграции обращаются к PostgreSQL по внутренней сети и не требуют проброса порта. PostgreSQL должен пройти healthcheck перед запуском зависимых сервисов.

Для клиента БД на хосте: адрес `127.0.0.1`, порт из `POSTGRES_FORWARD_PORT` (по умолчанию `5433`), пользователь и база из `.env`. Для контейнеров проекта адрес БД — `postgres:5432`. Пароль передаётся мигратору отдельной переменной окружения, а не аргументом командной строки.

У PostgreSQL нет опубликованных портов. Forwarder слушает только loopback, поэтому не открывает БД в локальную сеть. После реализации приложения наружу будет опубликован только HTTP-порт `tramcast-backend`; Python gRPC останется внутренним.

## Конфигурация backend

`-env-file` загружает только указанный файл. Уже установленные переменные окружения имеют приоритет, включая пустые значения: пустой обязательный параметр будет ошибкой. Без этого флага приложение читает окружение процесса. В Make путь можно заменить: `make run-backend ENV_FILE=../.env.local`; он считается относительно `backend`.

| Переменная | Значение по умолчанию / назначение |
| --- | --- |
| `HTTP_ADDR` | `:8080` |
| `HTTP_READ_HEADER_TIMEOUT` / `HTTP_READ_TIMEOUT` | `5s` / `15s` |
| `HTTP_WRITE_TIMEOUT` / `HTTP_IDLE_TIMEOUT` | `30s` / `60s` |
| `HTTP_SHUTDOWN_TIMEOUT` / `HTTP_PROBE_TIMEOUT` | `10s` / `2s` |
| `LOGGER_LEVEL` / `LOGGER_FORMAT` | `INFO` / `text`; формат также может быть `json` |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `127.0.0.1` / `5433` для локального forwarder |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Обязательные значения из `.env` |
| `POSTGRES_SSLMODE` | `disable` для локальной PostgreSQL |
| `POSTGRES_MAX_CONNS` / `POSTGRES_MIN_CONNS` | `10` / `0` |
| `POSTGRES_CONNECT_TIMEOUT` / `POSTGRES_STARTUP_TIMEOUT` | `5s` / `10s` |
| `WEB_DIR` | `../frontend/dist`: сборка фронтенда; относительный путь считается от рабочего каталога процесса и при старте переводится в абсолютный |
| `CATALOG_FILE` | `../dataset/spravochniki/Хакатон_справочники_трамвай_10_маршрутов.xlsx`: книга для `make import-catalog`; путь разрешается так же, как `WEB_DIR`. Книга содержит персональные данные, в Git её не кладут |
| `CATALOG_OSM_FILE` | `../data/osm/tram_routes.json`: снимок OpenStreetMap для `make import-catalog`, хранится в Git; путь разрешается так же, как `WEB_DIR`. Пустое значение — ошибка: импорт без снимка очистил бы географию маршрутов 17, 25, 26, 28, 50 |

Если изменили `POSTGRES_FORWARD_PORT`, задайте такое же значение `POSTGRES_PORT` для локального backend. В будущем контейнер backend должен использовать `POSTGRES_HOST=postgres`, `POSTGRES_PORT=5432`. Конфигурация проверяется до подключения; ошибки не выводят пароль или содержимое dotenv-файла.

После запуска доступны `http://localhost:8080/healthz` и `http://localhost:8080/readyz`. Первый маршрут проверяет HTTP, второй — доступность БД с ограничением времени. Справочники читаются через `/api/routes`, `/api/routes/geometry` (схемы маршрутов в GeoJSON для карты), `/api/routes/{route_id}/stops` и `/api/stops`; до `make import-catalog` списки пустые. API прогнозов пока не подключён. Завершение — Ctrl+C; приложение ждёт активные запросы до настроенного таймаута и закрывает пул.

Исходники фронтенда лежат в корневом `frontend/`, а Go отдаёт результат сборки из `WEB_DIR`. `make run-backend` запускает процесс в `backend/`, поэтому значение по умолчанию `../frontend/dist` указывает на `TramCast/frontend/dist`. Абсолютный путь выводится в лог при старте. Если `index.html` нет, в логе будет предупреждение, а страницы вернут `404`; API при этом работает. Новая сборка подхватывается без перезапуска.

## Фронтенд

```text
make frontend-install
make frontend-build
make frontend-dev
make frontend-test
```

`frontend-install` ставит зависимости строго по `frontend/package-lock.json` (`npm ci`); повторяйте его после изменения lock-файла. `frontend-build` проверяет типы и собирает `frontend/dist`, который отдаёт Go из `WEB_DIR`: после сборки интерфейс открывается на `http://localhost:8080`. `frontend-dev` запускает Vite на `http://localhost:5173` с горячей перезагрузкой и проксирует `/api` на backend `127.0.0.1:8080`, поэтому backend должен быть запущен. `frontend-test` проверяет типы и запускает модульные тесты Vitest.

Прогноз в интерфейсе — демо-данные, API прогнозов пока не подключён. Сценарий демо переключается в строке контекста («Демо-данные»). Подложка карты загружается с OpenFreeMap и требует интернета; при ошибке загрузки интерфейс сообщает, что подложка недоступна. Без импортированных справочников карта и список маршрутов пустые.

## Остановка и данные

```text
make env-port-close
make env-down
```

Первая команда закрывает локальный доступ к БД. Вторая останавливает сервисы всех используемых профилей. Файлы БД сохраняются в `out/pgdata`, который исключён из Git; команды автоматического удаления данных нет.

PostgreSQL 18 хранит данные внутри `/var/lib/postgresql`; сюда смонтирован каталог проекта. Для обновления на другую основную версию нужен перенос данных, а не простая замена тега образа. Параметры `POSTGRES_USER`, `POSTGRES_PASSWORD` и `POSTGRES_DB` инициализируют пустой каталог: изменение `.env` не меняет существующие роли и пароли в уже созданной БД.

## SQL и генерация

```text
make migrate-create name=add_example
make migrate-validate
make sqlc-compile
make sqlc-generate
```

`migrate-create` создаёт заготовку для следующего изменения схемы. `migrate-up` применяет миграции, `migrate-status` показывает состояние. `migrate-down` откатывает последнюю миграцию; откат начальной схемы удаляет её таблицы и данные.

Генерация sqlc не применяет миграции. Запросы и необходимые транзакции описаны в [backend/README.md](../backend/README.md#границы-транзакций).

## Protobuf

Общий контракт Go и Python описан в [proto/README.md](../proto/README.md). `make proto-generate-go` генерирует Go-код; нужны `protoc`, `protoc-gen-go` и `protoc-gen-go-grpc` в `PATH`. Компилятор проверяет синтаксис и импорты схемы при генерации. Подробности генерации приведены рядом с контрактом.

## Зависимости Go

```text
make tidy-backend
make tidy-backend-check
```

`tidy-backend` выполняет `go mod tidy` в `backend`: добавляет недостающие и удаляет неиспользуемые зависимости в `go.mod` и `go.sum`, при необходимости скачивая модули. Запускайте его после изменения импортов, `make sqlc-generate` или `make proto-generate-go` и коммитьте `go.mod` и `go.sum` вместе с кодом. `tidy-backend-check` ничего не записывает: если нужны изменения, выводит diff и завершается с ошибкой.

`make run-backend` не вызывает `tidy`. `go run` и `go build` не изменяют `go.mod` сами и сообщают о недостающей зависимости ошибкой сборки.

## Caddy и развёртывание

В MVP Go будет раздавать и API, и собранный фронтенд. Отдельный веб-сервер для статики не требуется. HTTPS при размещении в интернете обеспечит либо облачная платформа, либо обратный прокси — например Caddy, если размещаем приложение на собственном сервере. Выбор зависит от места развёртывания; локальное окружение не требует Caddy.

## Python ML

`make run-ml` запускает gRPC на `127.0.0.1:50051`; `make test-ml-service` проверяет реальный сетевой вызов и контракт. `make test-ml` запускает тот же набор сервиса и клиента; исследовательские проверки перенесены в архив. Переменная `PYTHON` выбирает интерпретатор; для venv передавайте абсолютный путь, поскольку цели запуска работают из `ml/runtime/`. Зависимости сервиса с пересчётом — `ml/runtime/requirements-inference.txt`, быстрых тестов — `ml/runtime/requirements.txt`, исследований — `ml/lab/requirements.txt`, генератора — `ml/runtime/requirements-proto.txt`.

`make proto-generate-python` генерирует Python из того же `.proto`, что используется Go. Сгенерированный код хранится в `ml/runtime/generated/`.

`docker compose up -d --build ml` запускает только ML. Внутри сети Compose адрес — `ml:50051`; порт не публикуется на хост. Сервис не зависит от PostgreSQL. Текущий Go-клиент ещё не реализован: добавление сервиса не означает готовности полного пользовательского сценария.

Перед запуском ML с пересчётом установите веса: `make prepare-ml-model PYTHON=<абсолютный путь>`. Основной рецепт — 030; Compose требует NVIDIA GPU и NVIDIA Container Toolkit для холодного расчёта, локальный SQLite-кэш сохраняется в volume `ml-cache`. Выдача кэша выполняется на CPU; руководство — [ML README](../ml/README.md).
