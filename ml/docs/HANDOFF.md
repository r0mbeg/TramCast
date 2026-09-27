# Запуск ML-сервиса из репозитория

Для новой истории используйте [подготовку нового пакета](PREPARATION.md), затем
[порядок подключения обновления к backend](INTEGRATION.md#как-подключить-обновление-данных).
Подготовка запускается отдельной командой на ML-хосте, не через `Predict`.
Ниже описан запуск сервиса с поставочным пакетом ноября–декабря 2025.

Код и инструкция передаются через GitHub; отдельный архив не нужен.
Для остановочной функции добавлен `PredictStops` и клиент `--stops`.
Компактный пакет профилей входит в оба Docker-образа, `lab` для работы не нужен.
[Подробная интеграция Go: запуск, версии, поля ответа, проверки и хранение](STOPS_INTEGRATION.md).
После получения ветки с ML-изменениями выполняйте команды **из корня TramCast**.
Для самостоятельного запуска нужны Docker и Compose v2; PostgreSQL, Go,
локальный Python и корневой `.env` не требуются.

## Если запуск останавливается на `Input checksum mismatch: base.csv`

Git с `core.autocrlf=true` может заменить LF на CRLF при скачивании на Windows.
Числа в CSV остаются прежними, но SHA256 меняется. `ml/.gitattributes` запрещает
такое преобразование для входов и сохранённых прогнозов; исходники runtime
получают LF, поскольку их байты также участвуют в ID модели.

Для ранее скачанной копии нужно восстановить исходные байты и пересобрать образ.
Если перечисленные файлы не содержат ваших правок, выполните из корня проекта
(работает и до получения обновлённого `.gitattributes`):

```sh
git -c core.autocrlf=false restore --source=HEAD --worktree -- ml/recipes/tabpfn-030/base.csv ml/bundles/002/forecast.csv ml/bundles/030/forecast.csv ml/runtime/model_030.py ml/runtime/constants.py
docker compose -f ml/compose.yaml --profile gpu up -d --build --wait
docker compose -f ml/compose.yaml --profile gpu exec ml-gpu python service.py --warm-cache
```

Веса и кэш сохраняются; заново запускать `ml-install` не требуется.
Не заменяйте ожидаемые SHA256 в `recipe.json`: они проверяют целостность входов.

## Рабочий режим: расчёт на GPU, выдача кэша на CPU

Целевое окружение — Linux/x86_64 с NVIDIA RTX 3070 8 ГБ, драйвером NVIDIA
и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Сборка образа и первая установка весов требуют интернета.

```sh
docker compose -f ml/compose.yaml run --rm --build ml-install
docker compose -f ml/compose.yaml --profile gpu up -d --build --wait
docker compose -f ml/compose.yaml --profile gpu exec ml-gpu python service.py --warm-cache
```

Первая команда скачивает закреплённые веса и проверяет SHA256; повторный запуск
проверяет уже установленный файл. Вторая запускает gRPC на **127.0.0.1:50051**.
Третья вызывает модель и сохраняет полный прогноз в SQLite. Повторный прогрев
и последующие запросы используют кэш, в том числе после перезапуска.
Без прогрева расчёт запустит первый запрос ненулевого маршрута.

Текущий рецепт **030** пересчитывает финальную поправку TabPFN поверх заранее
подготовленного прогноза 029. Входы уже лежат в репозитории; исходный `dataset/`
и исследовательский `lab/` для запуска не нужны. Поддержан ноябрь–декабрь 2025.
Обновление истории и весь исследовательский поиск не выполняются при запросе.

Расчёт проверен на RTX 2080 Ti: около 51 секунды и 1 ГиБ видеопамяти.
На RTX 3070 сам Compose ещё не проверен. Healthcheck проверяет только gRPC;
успешный `--warm-cache` при первом запуске подтверждает также работу вычислителя.
Если CUDA недоступна, проверить доступ контейнера к GPU:

```sh
docker compose -f ml/compose.yaml --profile gpu exec ml-gpu python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

## Проверочный запрос и подключение backend

```sh
docker compose -f ml/compose.yaml --profile gpu exec ml-gpu python client.py --route 1 --from 2025-11-01T00:00:00+03:00 --to 2026-01-01T00:00:00+03:00 --timeout 930 --output /tmp/prediction.json
docker compose -f ml/compose.yaml --profile gpu exec ml-gpu python service.py --describe
```

Клиент проверяет ответ и печатает `hours: 1464` и сумму. Полный JSON сохраняется
внутри контейнера; уберите `--output`, чтобы вывести его в терминал.
`--describe` возвращает фактические ID модели и данных, период и часовой пояс —
их backend использует при регистрации версии прогноза.

Backend на том же хосте подключается к `127.0.0.1:50051`; контейнер backend
в общей сети `tramcast-ml_default` — к `ml-gpu:50051`. `localhost` внутри
контейнера указывает на сам контейнер. Порт опубликован только локально;
сервис использует gRPC без TLS для локального хоста/закрытой сети.

Метод — `tramcast.forecast.v1.ForecastService/Predict`,
[исходный контракт](../../proto/tramcast/forecast/v1/forecast.proto).
Интеграция Go пока не реализована; её правила описаны в
[INTEGRATION.md](INTEGRATION.md) и [контракте backend](../../proto/README.md).

## Проверка API без GPU

На macOS или компьютере без NVIDIA можно запустить выдачу сохранённого
результата 030. Это режим `replay`: он не вызывает модель и не проверяет кэш.
Оба режима занимают один порт, поэтому сначала остановите работающий ML:

```sh
docker compose -f ml/compose.yaml --profile replay --profile gpu down
docker compose -f ml/compose.yaml --profile replay up -d --build --wait
docker compose -f ml/compose.yaml --profile replay exec ml-replay python client.py --route 1 --from 2025-11-01T00:00:00+03:00 --to 2026-01-01T00:00:00+03:00 --output /tmp/prediction.json
```

Адрес тот же: `127.0.0.1:50051`; в общей Docker-сети — `ml-replay:50051`.
Ожидаемый ответ: 1 464 часа, сумма 1 103 274. Версии replay отличаются от
GPU-режима и находятся в `ml/bundles/030/forecast_bundle.json`.
Перед возвращением к GPU выполните команду `down` выше, затем рабочий запуск.

## Логи, остановка и пересчёт

```sh
docker compose -f ml/compose.yaml --profile gpu logs --tail 50 ml-gpu
docker compose -f ml/compose.yaml --profile replay --profile gpu down
```

Для логов replay замените профиль `gpu` на `replay`, а сервис `ml-gpu` на
`ml-replay`. Обычный `down` сохраняет веса и кэш в Docker volumes; `down -v`
удаляет их. Если порт занят, задайте `ML_PORT=50052` в окружении при запуске.

Ручной принудительный пересчёт создаёт новое поколение конфигурации через
`service.py --refresh`, затем выполняет `--config … --warm-cache`.
[Порядок смены версии](INTEGRATION.md#неизменяемые-версии) сохраняет прежние
результаты. Для запуска контейнера с новым конфигом задайте `ML_CONFIG`
в окружении Compose; файл должен находиться внутри контейнера, например
в постоянном `/app/ml/cache/`. Обычный повторный `--warm-cache` с прежним
конфигом использует кэш и не является принудительным пересчётом.
