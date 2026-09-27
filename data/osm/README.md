# Снимок OpenStreetMap для импорта справочников

В книге справочников организаторов нет маршрутов № 17, 25, 26, 28 и 50. Их остановки и схемы движения команда `backend/cmd/import-catalog` берёт из этого снимка OpenStreetMap. Снимок хранится в Git в каталоге `data/osm/` корня репозитория и читается командой импорта по пути из конфигурации, как книга справочников. Основной сервер снимок не читает.

| Файл | Содержимое |
| --- | --- |
| `tram_routes.json` | Ответ Overpass API без изменений |
| `tram_routes.overpassql` | Запрос, которым получен снимок |

## Путь к снимку

- Путь задаёт переменная `CATALOG_OSM_FILE`. По умолчанию — `../data/osm/tram_routes.json`: относительный путь разрешается от рабочего каталога процесса, а `make import-catalog` и `make import-catalog-dry-run` запускают команду в `backend/`, поэтому значение по умолчанию указывает на этот файл.
- Флаг `-osm-file` команды импорта заменяет путь на один запуск, как `-file` для книги.
- На сервере файл можно разместить в любом каталоге, например рядом с книгой справочников, и указать путь в `.env`.

## Источник и состав

Источник — OpenStreetMap через зеркало Overpass API `https://maps.mail.ru/osm/tools/overpass/api/interpreter`. Запрос выбирает по идентификаторам 10 relation `type=route` (по два направления на маршрут), их узлы-участники и родительские relation `route_master`. Те же идентификаторы с направлениями перечислены в `catalog_service.OSMRoutes` (`backend/internal/features/catalog/service/service.go`).

Запрос использует `out body`: снимок содержит теги и координаты без метаданных правок — имён и ID участников OpenStreetMap, версий и наборов правок. Переходить на `out meta` нельзя: это персональные данные. Тест `catalog_osm_repository` проверяет, что таких полей в снимке нет.

Импорт берёт узлы с ролями `stop`, `stop_entry_only`, `stop_exit_only` в порядке участников relation; платформы и пути рельсов не используются. Идентификаторы в БД получают префикс: `osm:node/<id>` у остановок, `osm:relation/<id>` у вариантов движения и у маршрута (`route_master`). Если маршрут появится в книге организаторов, книга имеет приоритет, а данные снимка для него пропускаются с предупреждением.

## Актуальность

- Состояние базы OSM: `timestamp_osm_base` = `2026-09-26T22:39:54Z`.
- Последняя проверка маршрутов по тегу `check_date`: 2023-05-11 у № 17, 25, 26, 28 и 2025-12-19 у № 50.
- Соответствие маршрутов периоду ноябрь–декабрь 2025 года не проверено.

## Обновление

Все команды выполняются из корня репозитория.

1. Сохранить новый ответ во временный файл. В PowerShell 5.1 вызывайте `curl.exe`: `curl` там — псевдоним `Invoke-WebRequest`.

   ```text
   curl --fail --max-time 120 -A "TramCast/0.1" https://maps.mail.ru/osm/tools/overpass/api/interpreter --data-urlencode "data@data/osm/tram_routes.overpassql" -o data/osm/tram_routes.json.tmp
   ```

   С `--fail` тело ошибки HTTP не записывается. При сбое запроса Overpass может ответить кодом 200 с полем `remark` и неполными данными; такой снимок импорт отклонит.
2. Переместить `data/osm/tram_routes.json.tmp` на место `data/osm/tram_routes.json`.
3. Выполнить `go -C backend test ./internal/features/catalog/...`. Тесты читают `data/osm/tram_routes.json` и сверяют дату `timestamp_osm_base`, число элементов, маршрутов, остановок и позиций, названия маршрутов и идентификаторы relation. После каждого нового запроса проверка даты **всегда** перестаёт проходить, потому что `timestamp_osm_base` меняется при любой выгрузке: обновите дату в `backend/internal/features/catalog/repository/osm/repository_test.go` и в константе `committedSnapshotBase` в `backend/internal/features/catalog/service/osm_test.go`. Если изменились сами данные OSM, после проверки изменений обновите и остальные ожидания: число элементов, идентификаторы, названия, состав маршрутов и счётчики отчёта в `backend/internal/features/catalog/service/service_test.go`.
4. Обновить дату и значения `check_date` в разделе «Актуальность» этого файла и в `AGENTS.md`: в таблице источников, в абзаце о датировке географии и в правиле импорта «Не сохраняются».
5. Выполнить `make import-catalog-dry-run`: нужны запущенная PostgreSQL и книга справочников.
6. Просмотреть изменения (`git diff`) и закоммитить снимок.

Откат: `git checkout -- data/osm/tram_routes.json` и удалить оставшийся `data/osm/tram_routes.json.tmp`, если он есть. Если в OSM сменились идентификаторы relation, обновите их в `data/osm/tram_routes.overpassql` и в `catalog_service.OSMRoutes`.

## Лицензия

Данные © участники OpenStreetMap, доступны по лицензии [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/); подробности — https://www.openstreetmap.org/copyright. Строка `osm3s.copyright` в JSON сохраняется. Строки `routes`, `stops` и `routes_stops`, полученные из снимка, остаются под ODbL; на лицензию кода это не влияет. Карта с этими данными показывает подпись «© участники OpenStreetMap» со ссылкой на https://www.openstreetmap.org/copyright.
