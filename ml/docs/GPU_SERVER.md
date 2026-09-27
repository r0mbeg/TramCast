# GPU-сервер TramCast

Развёртывание под пользователем `bazevv` на `89.111.142.222`, SSH-порт `1337`.
Каталог `/home/bazevv/tramcast`, отдельный Python 3.12 и пользовательская служба
`tramcast-ml.service`, без Docker и sudo. GPU 0 (RTX 2080 Ti) используется при холодном расчёте;
выдача SQLite-кэша выполняется на CPU. Built with PriorLabs-TabPFN;
[лицензия весов](../recipes/tabpfn-030/LICENSE.txt).

## Проверено 27 сентября 2026

- Реальный холодный расчёт внутри защищённой службы: **55,6 с**, GPU 0
  RTX 2080 Ti 11 ГБ, пик общей занятой видеопамяти **1 042 МиБ**
  при исходных 6 МиБ. GPU 1 не использовалась сервисом.
- Проверены все **14 640 часов**, нули, версии, повторный запрос и коды ошибок.
  После перезапуска ответы из кэша занимали до 0,21 с на маршрут.
- Полный контракт прошёл через SSH-туннель с Mac. После принудительного
  завершения основного процесса systemd восстановил службу, результат сохранился.
  Автозапуск и linger включены; перезагрузка общего сервера не выполнялась.
- После расчёта дочерний процесс освободил GPU; отдельного вычисления при чтении
  кэша не было. 26 хешей переданных исходников, входов и весов проверены.
- С прежним replay 030 различаются 1 620 часов из 14 640, максимум на
  3 валидации. Среда PyTorch/CUDA отличается; причина каждого численного
  расхождения отдельно не исследовалась. Это отдельный model_version,
  не побитовое воспроизведение прежнего результата и не новое измерение качества.

[Метаданные версии](../deploy/verified-20260927/metadata.json),
[холодный запрос](../deploy/verified-20260927/check-cold.json),
[перезапуск](../deploy/verified-20260927/check-restart.json),
[автовосстановление](../deploy/verified-20260927/check-recovery.json),
[SSH-проверка](../deploy/verified-20260927/check-tunnel.json),
[сравнение с replay](../deploy/verified-20260927/reference-comparison.json).

## Доступ backend

На ML-сервере gRPC слушает только Unix-сокет
`/home/bazevv/tramcast/run/forecast.sock` в закрытом каталоге.
TCP-порт ML на сервере не открывается. Для Go на другом хосте
поднимите SSH-туннель **на хосте backend**:

```sh
ssh -a -NT -i ~/.ssh/tramcast_gpu -p 1337 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:50051:/home/bazevv/tramcast/run/forecast.sock bazevv@89.111.142.222
```

Команда работает в переднем плане; пока она запущена, адрес Go-клиента —
`127.0.0.1:50051`. Для постоянного backend туннель должен запускаться своим
менеджером служб с перезапуском при ошибке. Для другого компьютера заведите
отдельный SSH-ключ и подтвердите ключ сервера; приватный ключ с Mac не включайте
в репозиторий или Docker-образ. Транспорт gRPC внутри туннеля без TLS,
внешнее соединение шифрует SSH. Публичного порта gRPC нет.

Если Go находится в контейнере, его `localhost` относится к контейнеру.
Туннель должен быть доступен в той же сетевой области: например, на Linux
backend и туннель могут использовать host network. Не публикуйте туннель
на `0.0.0.0` без отдельной защиты. Локальный TCP-конец туннеля доступен другим
пользователям хоста backend: для общего хоста используйте также локальный
Unix-сокет в каталоге `0700` (`-L /закрытый/путь/ml.sock:/home/bazevv/tramcast/run/forecast.sock`).
На самом GPU-сервере Go под `bazevv` может подключиться напрямую к
`unix:///home/bazevv/tramcast/run/forecast.sock`.

## Контракт Go

Используйте уже сгенерированный `forecastv1.NewForecastServiceClient`
из `backend/internal/gen/tramcast/forecast/v1`. Метод — unary
`/tramcast.forecast.v1.ForecastService/Predict`.

Пример запроса (Protobuf JSON; сервис не является HTTP JSON API):

```json
{
  "routeNumber": 1,
  "forecastFrom": "2025-10-31T21:00:00Z",
  "forecastTo": "2025-12-31T21:00:00Z"
}
```

Это полный ноябрь–декабрь 2025 по Москве: 1 464 часа на маршрут.
Поддержаны `1, 5, 7, 11, 12, 17, 25, 26, 28, 50`.
`routeNumber` — публичный номер, не внутренний `routes.id`.
Маршрут 5 и часы 1–4 имеют обязательные нули. Час 5 включает период с 05:30.
Новые даты требуют нового офлайн-пакета; запрос не обновляет историю.

Последовательность backend:

1. Для этого развёртывания взять [проверенные метаданные](../deploy/verified-20260927/metadata.json)
   и закрепить их в `forecast_versions`. При новом развёртывании получить
   метаданные заново через `service.py --describe`.
2. Из фонового обработчика вызвать `Predict` для полного горизонта версии;
   deadline RPC не связывать с HTTP-контекстом. Для холодного расчёта оставить
   до 900 секунд; прогретый кэш обычно отвечает значительно быстрее.
3. Сверить `model_version`, `dataset_version`, присутствие `hour_start` и
   `boardings` (`point.Boardings != nil`), диапазон и каждый ожидаемый час.
   Ноль — готовое значение. Повторное округление не нужно.
4. С актуальной арендой и номером попытки атомарно записать все точки и
   `succeeded` в PostgreSQL. Очередь и публикация принадлежат Go.

`UNAVAILABLE` и `DEADLINE_EXCEEDED` — ограниченные повторы через очередь;
`INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION` и несовпадение версий —
неповторяемые ошибки. Остальные правила — [контракт](../../proto/README.md)
и [интеграция](INTEGRATION.md). При смене комплекта согласуйте новую версию
до выдачи задач: RPC не принимает ID модели для выбора старого комплекта.

HTTP API прогнозов, очередь и вызов Python из Go пока не реализованы.
Развёрнутый Python-сервис сам их не добавляет.

## Защита и её границы

Конфигурация — [tramcast-ml.service](../deploy/tramcast-ml.service).
Перед каждым запуском [check_sandbox.py](../deploy/check_sandbox.py) проверяет
фактическое применение основных ограничений; при провале служба не стартует.

- Каталог приложения и каталоги состояния имеют `0700`, новые файлы — `UMask=0077`.
  Другие обычные пользователи не могут читать файлы или подключаться к сокету.
- `NoNewPrivileges`, пустые capabilities, запрет SUID/SGID и core dumps.
- `ProtectHome=tmpfs`: домашние каталоги скрыты, включая `.ssh`. Только приложение
  и интерпретатор Python подключены для чтения; запись разрешена в кэш, каталог
  сокета и отдельный временный каталог. Системные файлы доступны только для чтения.
- Проверено отсутствие обхода доступа к домашнему каталогу через чужие процессы
  в /proc; пользовательские сокеты управления systemd/DBus скрыты.
- Разрешены только сокеты `AF_UNIX`; процесс не может создавать IPv4/IPv6-соединения.
  Запросы и загрузка библиотек не скачивают модели из интернета.
- Лимит RAM 8 ГиБ, 128 задач; привязка к CPU 0 и 1. Cgroup-квота CPU пользователю
  не делегирована, поэтому используется проверяемый CPU affinity.
- Выбрана одна GPU через `CUDA_VISIBLE_DEVICES`. Это настройка вычисления,
  не аппаратная изоляция GPU и не жёсткий лимит её видеопамяти.
- Веса проверяются по закреплённой SHA256 перед запуском и после расчёта,
  входы также проверяются. RPC не принимает файлы, модели, Python-код или матрицы.

Это **не гарантия отсутствия уязвимостей** и не аудит всего общего сервера.
Процессы того же пользователя `bazevv` и администратор остаются в доверенной зоне.
Отдельная системная учётная запись, обновления ядра/драйвера и аппаратная
изоляция ресурсов требуют администратора. Ключ для SSH-доступа к аккаунту
даёт доступ ко всему аккаунту, поэтому не передавайте его третьим лицам.

`pip-audit` обнаружил 24 записи (с повторениями) для закреплённого PyTorch 2.6.0.
Setuptools обновлён до 83.0.0; в остальных 46 пакетах этот аудит известных
уязвимостей замечаний не нашёл. [Полный отчёт](../deploy/verified-20260927/dependency-audit.json)
и [точные версии](../deploy/verified-20260927/requirements.txt) сохранены.
Для поиска PyTorch в базе PyPI суффикс +cu118 нормализован до 2.6.0;
бинарные CUDA-библиотеки и весь сервер этим аудитом не проверяются.
Среди замечаний к PyTorch — обработка недоверенных checkpoints и ошибки отдельных операторов.
Фиксированные проверенные веса, отсутствие загрузки пользовательских данных и
изоляция сокращают поверхность атаки, но **не исправляют библиотеку**.
Обновление PyTorch требует отдельной проверки совместимости с драйвером,
TabPFN и результатом 030, а также нового `model_version`.

## Управление на сервере после запуска

```sh
systemctl --user status tramcast-ml
journalctl --user -u tramcast-ml -n 50 --no-pager
systemctl --user restart tramcast-ml
systemctl --user stop tramcast-ml
```

Метаданные и проверочный запрос:

```sh
cd ~/tramcast
export PYTHONPATH="$PWD/ml/runtime/generated"
.venv/bin/python ml/runtime/service.py --config ml/recipes/tabpfn-030/recipe.json --describe
.venv/bin/python ml/runtime/client.py \
  --address unix:/home/bazevv/tramcast/run/forecast.sock --route 1 \
  --from 2025-11-01T00:00:00+03:00 --to 2026-01-01T00:00:00+03:00 \
  --timeout 900 --output /tmp/tramcast-route-1.json
```

Кэш — `~/tramcast/ml/cache/forecasts.sqlite3`, веса — `~/tramcast/ml/models/`.
Остановка и перезапуск сохраняют их.
Ручное новое поколение — через `--refresh`,
как описано в [INTEGRATION.md](INTEGRATION.md#неизменяемые-версии).

## Воспроизведение окружения

Перенесите `ml/runtime`, `ml/recipes/tabpfn-030`, `ml/deploy` в `~/tramcast/ml`.
Для быстрых тестов нужны также `ml/bundles/002` и `030`.
Веса устанавливаются отдельно или переносятся с проверкой SHA256.
На сервере уже есть `uv`; системный Python и драйвер не изменяются.

```sh
cd ~/tramcast
umask 077
chmod 700 ~/tramcast
uv venv --python 3.12.12 .venv
uv pip install --python .venv/bin/python torch==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r ml/deploy/verified-20260927/requirements.txt
uv pip check --python .venv/bin/python
export PYTHONPATH="$PWD/ml/runtime/generated"
.venv/bin/python ml/runtime/recipe.py
mkdir -p ml/cache run ~/.config/systemd/user
chmod 700 ml/cache run
cp ml/deploy/tramcast-ml.service ~/.config/systemd/user/
loginctl enable-linger "$USER"
systemctl --user daemon-reload
systemctl --user enable --now tramcast-ml
.venv/bin/python ml/runtime/service.py --config ml/recipes/tabpfn-030/recipe.json --describe > run/metadata.json
.venv/bin/python ml/deploy/check_service.py \
  --address "unix:$HOME/tramcast/run/forecast.sock" --metadata run/metadata.json
```

Первый запрос проверки запускает реальный GPU-расчёт всех маршрутов внутри
защищённой службы. Следующие запросы используют SQLite. `linger` обеспечивает
работу без SSH-сеанса и запуск пользовательского systemd при загрузке.
Служба включена в `default.target`, при сбое перезапускается с задержкой.
На другом хосте проверьте UUID GPU, номера CPU, путь Python и поддержку
пользовательских namespaces/cgroups перед применением unit-файла.
Изменение версии PyTorch/CUDA меняет ID модели: повторно
получите `--describe`, не подставляйте ID другого развёртывания.
