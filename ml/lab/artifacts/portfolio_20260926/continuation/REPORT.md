# Продолжение P20260926 — промежуточный результат

27.09.2026 МСК. **Поиск остановлен по просьбе пользователя; goal на паузе:** исходный gate ≥0,90 на каждом прежнем
61-дневном окне. Лучший минимум текущего набора031=0,88392, его mean0,88857.
Самое высокое среднее029=0,88875.
Главная цель — закрытый конкурсный score; официальной оценки новых CSV нет.
Данные, очистка, метрика, сетка и half-up сохранены.

## Текущий набор после P72

**031/029/025/024/023/026/005**. 031 первый по правилу близкого среднего
и лучшего минимума; все4его окна<0,90. 029 первый по среднему.
025 сохраняет сильный альтернативный дневной объём.
Финальные CSV уникальны; ближайшая пара029/024 отличается на1,31%.
Сильный исходный021 остаётся в архиве; его объём представлен в новых вариантах.
[Рейтинг](../../../SOLUTIONS.md), [ошибки и смещения](../finalist_window_metrics.csv),
[маршруты](../finalist_routes.csv), [различия](../finalist_diversity.csv).
1323 сохранённых окон/105 внутренних final проверены независимо;
17 защищённых файлов и осенний контрольChronos неизменны. Это повторно
используемые перекрывающиеся окна, не1309 независимых проверки.
Старые результаты ниже сохранены как хронология; актуальные P57–P71 — в конце.

## Посылки этапа P20/P21

| CSV | W1 | W2 | W3 | W4 | Среднее | Худшее |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [011 Bayesian volume](../../../submissions/011_bayesian_volume_operations.csv) | 0,87221 | 0,85741 | 0,88768 | 0,90395 | 0,88031 | 0,85741 |
| [010 Weather + operations + seasons](../../../submissions/010_weather_operations_season.csv) | 0,87786 | 0,84827 | 0,88897 | 0,90517 | 0,88006 | 0,84827 |

Разница среднего <0,001: по закреплённому правилу предпочтён 011 с лучшим
минимумом. Между ними relative L1 =3,22% финального объёма. После P21 набор
из семи был **011, 010, 004, 005, 006, 009, 002**; минимальная разница любой
пары 3,01%. Рейтинг может измениться в следующих исследованиях. 007/008 и
остальные архивы сохранены. 002 остаётся пакетом сервиса.

### 010: объём, операции и часовая форма

Дневная Ridge-регрессия log1p на календарь и дождь ERA5: alpha=1,
route_season=False, recent residual_strength=0. Нормальная история исключает
известные фазы изменения 7/50. Ноябрьское восстановление задано первичным
объявлением от 15 ноября. Коэффициент июля при недостатке истории:
7=0,45; 50=1,25. При ≥3 доступных днях вместо prior используется медианная
оценка по прошлому: падение 7 ограничено [0,1], рост 50 допускается до 2.
Осенью 7 prior=0,6, 50=0. Эти числа выбраны 36-trial GridSampler на W1/W2,
не являются измерениями неизвестного будущего.

Часовая форма — средний маршрут×weekday×hour по максимум 32 наблюдениям
выбранного сезона: Dec–Feb, Mar–May, Jun–Aug, Sep–Nov. Не наблюдавшийся
сезон получает ближайший доступный по ERA5 температуре/длине дня. Недостающий
тип дня получает профиль из другой доступной истории. Форма нормируется,
после чего умножается на дневной объём. Праздники — 0,95 среднего Sat/Sun.

Ошибка W2 по маршрутам: 7 score=0,75234, 25=0,75491, 26=0,76911,
28=0,75271, 50=0,77059. Завышение суммы W1–W4 соответственно
4,77% / 8,85% / 2,99% / 0,62%. Локальное среднее против контроля Chronos
выше на 0,04411; это не измерение скрытого улучшения.

### 011: байесовская поправка

Base=010. BayesianRidge учится на log((actual+100)/(base+100)), ограниченном
±log(2), только по завершённым прошлым 61-дневным прогнозам с origins на
концах месяцев. Повторные route-date получают вес 1/count. Признаки:
маршрут, тип дня, сезон, горизонт, объём base, расхождение P10/P13,
температура, осадки, длина дня, праздничность. StandardScaler обучается
на этих же прошлых данных; BayesianRidge оценивает precision prior/noise.

Выбор 12-trial GridSampler W1/W2: strength=1, uncertainty=1,
history_days=224. Mean log correction ослабляется в
1+(predictive_sd/0,1)^2 раз. Posterior, факторы каждого cutoff, inner raw,
SQLite и trial-прогнозы сохранены в [bayes_volume](bayes_volume/).
Условная дисперсия не является доверительным интервалом скрытого score.
W2 улучшился на 0,00914; другие окна ухудшились на 0,00121–0,00564.

## Проверенные идеи и ограничения

P13 отдельная режимная статистика: 0,87010/0,80789/0,87661/0,88586.
P14 EWMA level×shape и P15/P16 автоматические аналоговые окна перенесли
режимы хуже; отрицательные результаты сохранены. P17 Dirichlet/Gibbs
ансамбль: 0,84356/0,82461/0,86924/0,88551; expected-quality proxy отличается
от фактического score на 2–4 пункта. P18 календарь+дождь полезен как компонент,
температура/длина дня не улучшили объём. P19 квантильная и погодная адаптация
Chronos ухудшила поздние окна; выбранные q=0,25 не обобщились.

Диагностика с подставленным фактическим суточным объёмом даёт 0,91–0,93
на текущих формах. Это **не доступный прогноз и не строгая верхняя граница**,
только разбор роли ошибки объёма; файлы diagnostic_true_volume_* исключены
из кандидатов. Факты не подставлялись в модельные посылки.

GitHub-проекты [asastyy](https://github.com/asastyy/moscow_transport/tree/4899aafc5031709c400dabd28009de1c9ca8b7a9)
и [EhimenNathan](https://github.com/EhimenNathan/tramcast-moscow/tree/2547767e749cde9b09de3718be29d10109147d40)
изучены без исполнения чужих скриптов/весов. Их высокие числа относятся
к другим окнам, очистке либо оценщикам с исключением нерегулярных дат.
Ссылки, commits, хеши в [public_sources](public_sources.json).

ERA5 — разрешённые ретроспективные внешние данные 2025, полученные в 2026;
это не тогда доступный архив прогноза. Движение также известно
ретроспективно. Нулевой fallback 5 сохранён по контракту проекта.
Обучающих месяцев всего десять: годовую сезонность нельзя надёжно выучить.
W1/W2 и перекрывающиеся W3/W4 повторно используются; независимого теста нет.

## Проверка, ресурсы и воспроизведение

Независимый аудит после P23: 618 окон, 40 финальных прогнозов,
17 защищённых ML-файлов неизменны. Все CSV 010/011 перечитаны при архивации,
имеют точные 14 640 ключей шаблона, целые неотрицательные значения,
half-up из raw, нули маршрута5/часов1–4. SHA в [archives](archives_010_011.json).
Новых отправок на конкурс, замены пакета сервиса и коммита не было.

Завершены jobs8476945/47/48/49/50/54/55, все COMPLETED 0:0:
29/106/23/19/64/42/18 секунд соответственно, по 4 CPU.
GPU только job8476950, 1 карта, 64 секунды; вместе с первым этапом
GPU114 секунд =0,03167 часа, CPU0,53444 core-hour. P23 завершён:
job8476956 COMPLETED 33 с, 0,87428/0,85326/0,89358/0,87848;
среднее хуже P20/P21. Скорость забывания выбрана из прошлого,
posterior/evidence studies сохранены в particle_level/levels. P22 ждёт явного разрешения на перенос raw:
автопроверка отклонила передачу train/test с хешами карт/временем/тарифами.

Команды из собственного remote-каталога `tram-portfolio-20260926`:

```sh
sbatch jobs/zhores_portfolio_operations.sbatch
sbatch jobs/zhores_portfolio_bayes_volume.sbatch
```

Это возобновляет те же studies с прежними общими лимитами. Точные исходники
в source_operations/source_bayes_volume, параметры и hashes — run_started.json.
Для воспроизведения в новой копии использовать отдельный output и те же версии
пакетов из сохранённых метаданных. Проверка локально из `ml/`:

```sh
LOKY_MAX_CPU_COUNT=4 OPENBLAS_NUM_THREADS=4 ../.venv/bin/python -m tests.test_portfolio_experiment
../.venv/bin/python -m experiments.portfolio_verify
```

Общий потолок сохраняется: до 27.09.2026 20:40:40 UTC и ≤24 GPU-часов,
резерв начинается 15:40:40 UTC. Порог завершения ещё не выполнен.


### P24/P25 — дополнительные проверки

P24 route×temperature/daylight Ridge: 0,81765/0,81234/0,87521/0,88943,
alpha1/spline/full history. Job8476958 COMPLETED0:0,19с. Не включён в набор.
P25 прямой дневной origin×horizon HGB: 0,85168/0,86240/0,83751/0,86999,
leaves15/l2=1/max_iter200. Job8476960 COMPLETED0:0,36с.
W2 лучше текущих 010/011, но общий результат хуже; отдельная посылка
не архивировалась, прогнозы сохранены для проверки сочетаний. Восемь trials
закончены, новые факты внутри горизонта не поступают. Проверка изменения
будущих target значений подтвердила неизменность causal context и fit.

Первичный источник дополнительно подтверждает выходное укорочение17
5–30 апреля и восстановление1 мая: external/route17_april_sources.json.
Прямое получение страниц не удалось; сохранён результат поиска по первичному
официальному тексту с датами/URL/ограничением получения. P26 проверяет
исключение этих дней из normal-fit при неизменных параметрах P20.

Через P25 GPU114с=0,03167часа; CPU0,59556core-hour.
Потолок ≥0,90 на каждом окне пока не достигнут.

P26 завершён: job8476961 COMPLETED0:0,6с;
0,87709/0,84828/0,88936/0,90543. Изменение каждого score <0,001,
почти одинаковый вариант не включён в набор. Через P26 CPU2168core-секунд
=0,60222core-hour; GPU114секунд. Учёт — accounting_through_P26.json.

Аудит после P26: 672 окна и 43 final-прогноза проверены из raw/CSV;
17 защищённых файлов совпали со снимком, текущие семь архивов неизменны.


### P28: сохранён новый012

75% rawP21 +25% rawP25, выбран GridSampler10 по среднему W1/W2.
Score0,86833/0,86136/0,89124/0,90339, mean0,88108, worst0,86136.
Изменение среднего к011 меньше0,001: выбор по лучшему минимуму.
012/011 final relativeL1=1,03% — одновременно не входят в семёрку.
ПослеP28 набор был012/010/004/005/006/009/002; все прежние архивы сохранены.
W1–W4 bias012: +7,24%/+5,55%/−2,65%/−1,37%.
Детализация маршрутов — ../verified_routes.csv, отличие от контроля
Chronos по среднему +0,04513. Это не измерение скрытого score.

Независимый аудит послеP28: 696 окон/44final,17защищённых файлов неизменны.
012 сохранён штатным архиватором с точной сеткой14640/half-up/forced zeros;
SHA a82c1b76c19b4f5037e973e7ea05ffeb6c6170b5c2ac0f7b78332c3dce52cda5.
Воспроизведение: sbatch jobs/zhores_portfolio_direct_blend.sbatch,
данные/компоненты фиксируются SHA в direct_blend/run_started.json.
Job8476965 COMPLETED0:0,16с на2CPU/2GiB.

### P27: завершённое дообучение с ковариатами

Старые full200/LoRA200 уже учитываются как отрицательные исторические
эксперименты на двух иных окнах. Содержательная новая гипотеза: обучение
с разрешённой ERA5 и календарём; часть без ковариат — сравнение на текущем
протоколе, не новое семейство. Два GridSampler по4trials steps200/800×lr1e-6/1e-5,
по1800с; выбранные params проверяются на4окнах и сseed73, finalseed42.
Минимальное прошлое28дней позволяет W1; обучающие61дней полностью≤cutoff.
Chronos случайно выбирает непрерывные исторические origins, порядок
внутри рядов сохраняется; случайного train/test разбиения нет.
Pilot8476963 COMPLETED0:0,21с выделения,fit2,463с/peakGPU2,428GB.
Studyjob8476964 завершён1069с COMPLETED0:0, одно gpu_develGPU/4CPU/8GiB≤90мин.
Исходники source_finetune_pilot/source_finetune, checkpoints на Zhores;
результат ниже, кандидат P27 отвергнут.

Завершённые jobs черезP28 без ещё активного8476964: GPU135с=0,0375часа,
CPU2284core-секунды=0,63444core-hour. Общий потолок24GPU-часа сохраняется.


### P29/P30: опубликован013

HGB условных часовых долей: MAE на hourly/day share, весday volume,
leaves63/l2=10/max_iter150/learning_rate0,05/min_leaf40, seed42.
Из normal-fit исключены7/50 и апрельские выходные17, низкие дни
<max(500,0,35 median route/daytype). Это ограничения обучения,
оценочные данные остаются прежними. Календарь и ERA5 определяют форму,
дневные объёмы012 сохраняются точно до округления. 50% learned shape
+50% прежней формы, выбран GridSampler8 только наW1/W2.

Score0130,87093/0,86468/0,89169/0,90135, mean0,88216, worst0,86468.
Среднее выше012на0,00108; W1/W2/W3 лучше, W4 хуже на0,00204.
Mean против Chronos-контроля выше на0,04621; скрытый результат неизвестен.
13/12 relativeL1=1,87%,13/10=3,39%; набор13/10/4/5/6/9/2,
012 и011 сохранены как близкие архивы. Ближайшая пара семёрки005/006=3,01%.

P30 проверяет ненормированные marginal hourly medians:4trials,
нормированный контрольP29 byte-equivalent по raw всех5cutoff.
NormalizeFalse оказался хуже dev; не создавать дубликат посылки.
Jobs8476966/8476967 COMPLETED0:0,49с/35с,4CPU/4GiB.
Воспроизведение: sbatch jobs/zhores_portfolio_conditional_shape.sbatch;
сравнение нормирования: sbatch jobs/zhores_portfolio_marginal_shape.sbatch.
Точные исходники source_conditional_shape/source_marginal_shape.
013 SHA74d5e837aab8bcf7c54bf86c523479e4d51879e1d1e3bc422716b432dcb4caaa.

### ИтогP27 и устойчивость

Daily200/lr1e-6:0,79815/0,81791/0,84836/0,85438;
seed73:0,79704/0,81883/0,84950/0,85743.
Daily_weather200/lr1e-6:0,81298/0,78664/0,80997/0,85906;
seed73:0,81208/0,78911/0,80889/0,85810.
Оба хуже mean контрольного zero-shot; seed меняет score≤0,0031.
Время программы1054,64с, allocation1069с, peakGPU2,432GB;
веса остаются на Zhores/models, метаданные/SHA/прогнозы/SQLite сохранены локально.
Результат не обосновывает расширение подбора тех же lr/steps.

Следующая содержательная гипотезаP31: маскировать известные ограничения
в normal-demand target series (NaN при сохранении ежедневных дат),
прогнозировать обычный спрос и отдельно применять операции;
сравнить отдельные и совместные9route targets. Она пока идея, не результат.
ЧерезP30 завершённые исследования: GPU1204с=0,33444GPUчаса,
CPU6896core-секунд=1,91556core-hour. Смотри accounting_checkpoint_P30.json.

Аудит после завершенияP27/P30:760окон/48finalпрогнозов,
17защищённых файлов неизменны, семь архивов с точной сеткой14640 проверены.
Все 48 внутренних forecasts не объявляются уникальным конкурсным набором.
Тесты causal-input isolation, forced zeros, conservation и контрольного
нормирования прошли; P22 по-прежнему ждёт разрешения на raw transfer.


### P31: маскирование операций и совместные маршруты

Job8476968 COMPLETED0:0,111с выделения/4CPU/8GiB/1GPU, фактическое исследование96,657с, peakGPU2,436GB. Обе studies завершили по2trials steps0/200,lr1e-6; выбранsteps0. Normal_daily:0,81136/0,87375/0,86406/0,88075; normal_joint_weather:0,84802/0,86742/0,85651/0,88677. Дообучение не победило свой zero-shot контроль; seeds не дублируются для детерминированного steps0. Кандидаты не добавляются в конкурсный набор. Веса не копировались локально; checkpoint SHA и метаданные сохранены.

P32 запущен job8476969: Bayesian volume correction с direct/base disagreement и causal ratios7/14/28. GridSampler6trials strength0/0,5/1 × history112/224, uncertainty1, meanW1/W2; selected4windows+final. ais-cpu4CPU/4GiB/30мин. Полный результат пока не измерен.


### P32: Bayesian direct disagreement

Job8476969 COMPLETED0:0,15с/4CPU/4GiB;6trials завершены. Выбранstrength1/uncertainty1/history224,0,874152/0,863692/0,889338/0,900076, mean0,881814/worst0,863692. По сравнению013 mean−0,000348 (внутри tie0,001), worst хуже на0,000987; finalrelativeL1=1,384%. Сохраняем013, новый близкий архив не создаём. Bayesian correction обучается только на законченных inner61day прогнозах; источником надёжности являются наблюдаемые прошлые ошибки. По одним будущим прогнозам скрытое качество установить нельзя.

Независимый аудит послеP31/P32:795окон/51внутренний final,17защищённых файлов неизменны;7уникальных архивов остаются прежними. ThroughP32 allocationGPU1315с=0,365278GPU-hour,CPU7400core-секунд=2,055556core-hour.


### P33: августовский режим7 и кандидат014

Новое официальное сообщение15августа:7 укорачивается по выходным с16августа; https://transport.mos.ru/mostrans/all_news/125846. С16августа по5сентября моделируется отдельный regime, затем прежний autumn с6сентября. Отдельная normal-fit маска; ratio median по>=3 affected observations≤cutoff, иначе selectedprior. Уровни0,6/0,8/1 не следуют из текста объявления, это фиксированные development-гиперпараметры. Контроль вообще не использует новый режим; extended17=True и P29shape одинаковы для всех trials.

Job8476970 COMPLETED0:0,10с/4CPU/4GiB;4Gridtrials завершены, prior0,8. Scores0,880159/0,852886/0,890996/0,903347, mean0,881847/worst0,852886. RelativeL1final к013=2,9035%,к010=1,9050%. Относительно010 mean+0,001783/worst+0,004620; W4 ниже на0,001820. Выбран для замены010 в семёрке,013 остаётся первым по worst при близкомmean. 010 сохранён в архиве.

Dev control0,866015,selected0,866523: чистое добавление августовской операции даёт всего+0,000508dev. Большая часть прироста над010 связана с уже найденнойP29shape. Не приписывать весь выигрыш новой дате операции. Новый набор013/014/004/005/006/009/002. Проверка архива014 выполняется штатным архиватором и независимым оценщиком. Воспроизведение: sbatch jobs/zhores_portfolio_august_operations.sbatch; source_august_operations и external/route7_august_sources.json.

ЧерезP33:GPU1315с=0,365278GPU-hour;CPU7440core-секунд=2,066667core-hour, активных заданий нет. Цель>=0,90накаждомокне по-прежнему не достигнута; исследования продолжаются в пределах24часов/24GPU-hour.

Аудит послеP33:807окон/52внутренних final,17защищённых файлов неизменны. Архив014 проверен:14640точных ключей,integer/nonnegative/half-up/forcedzeros;SHAe3071ee946da8d35a76d0e085386573e42d72db4c946b2083b3a707e4b518c3d. МинимальныйrelativeL1семёрки=2.90349964%. Goalactive,>=0,90не достигнуто.


### P34: положительный результат сезонной модели ошибок

Job8476971 COMPLETED0:0,8с/4CPU/4GiB,3trials;control exactP32raw на обоих development окнах. Shareddev0,876453,route-specificdev0,879659. Selectedroute annual gives0,883298/0,876019/0,887855/0,899770,mean0,886735/worst0,876019. RelativeL1finalк0132,9095%,к0143,6484%. Архив015 заменил013 в семёрке,013 сохранён. SHA015b28052c4f7f7938f327b0bd25a11de34a278a2aa53f3b5c9bfa19676c9745559. Mean противChronosC1+0,050790; не измерение скрытого score.

Bayesianerror features расширены первой календарной гармоникой и phasefuture-origin, без выборачастоты/числагармоник. Shared/route effects обучены наpastcompleted61dayerrors, posterior prior/noise автоматически оцениваются. Strength1/uncertainty1/history224 фиксированыP32, hourly shapeP29 одинаковая. Сезонная повторяемость по неполному году не доказана; conditional predictive variance не покрывает весь риск смены режима. Это смысловая гипотеза сезонности ошибок, не доказательство скрытого преимущества. BiasW1–W4+3,6284%/+2,7879%/−3,6581%/+0,4784%. Основные остаточные ошибки W2:50 score0,80196;7 0,81461;26 0,85849;28 0,80682. Воспроизведение:sbatch jobs/zhores_portfolio_bayes_annual.sbatch,source_bayes_annual.

P35 запущенSlurm8476972 наais-cpu4CPU/4GiB/30мин;8trials sharedNMF+Ridge, selected+seed73. Результат будет внесён после сохранения/audit.


### P35: общие NMFфакторы

Job8476972 COMPLETED0:0,14с/4CPU/4GiB;8trials. Rank6/alpha1/rain selected0,878738/0,847949/0,891457/0,903479,mean0,880406/worst0,847949. Seed73 scores exactlysame. 16из23fits достиглиmaxiter500, это измеренное ограничение solverbudget, а не полная сходимость/глобальный optimum. Не улучшает015/014; не добавлять близкую посылку. Imputednormalinputs не подменяют targets. Воспроизведение:sbatch jobs/zhores_portfolio_factor.sbatch, source_factor и factor/*fit*.json.

ЧерезP35: GPU1315с=0,365278GPU-hour,CPU7528core-секунд=2,091111core-hour. Все завершённые задания0:0, активных заданий нет. Бюджет24GPUчаса/24чwall сохраняется, reserve15:40:40UTC27сентября.

Аудит послеP34/P35:841окно/54внутренних final,17защищённых файлов неизменны;7архивов с точными14640строками иhalf-up/forcedzeros проверены. Finalist_window_metrics.csv содержит28строк, finalist_routes.csv280строк, таблица всех7кандидатов актуальна. Скрытый score неизвестен, цель>=0,90накаждомокне остаётся активной.


### P36: подготовка TimesFM

Закреплены officialwheel TimesFM2.0.2 (44,770bytes,SHA c7bde94beb1651e1251cdf1e9d09cf6f015e0218d038a4a836a170dc70b08071) и publicweightsrevision1d952420fba87f3c6dee4f240de0f1a0fbc790e3 (925,181,104bytes,SHA2f776efe6245e42b24bc4153ffdf61810140210e4bd3b01fb21f7aa779ab6ce8). Primarymetadataexternal/timesfm_2_0_2_pypi.json и timesfm_model_metadata.json; аудит точного APIисходника выполнен,requirementsPython>=3.10/Torch>=2.0 совпадаютс окружением3.12/2.6. PubliclicenseApache2.0. Separatetimesfm_deps вownZhoresroot не меняетmainenvironment/service. PublicdownloadtokenFalse выполняетсяSlurm8476973ais-cpu2CPU/2GiB/30мин. Inferenceещёнеизмерен; новое семейство не объявляется кандидатом до сопоставимых окон.


### P36: TimesFM и сопоставимый контроль

Публичные веса и точный wheel проверены по SHA; download8476973 завершён
за45с (2CPU), pilot8476974 за16с (4CPU/1GPU), inference0,778с и peakGPU
983662592bytes. Study8476975 не запустилаPython: старая bash отвергла
пустой массив подset-u. Передача аргументов исправлена, study8476976
COMPLETED0:0 за28с, actual12,666с; все9trials завершены. Выбрана
normal_ratio/q0,3. Scores0,881436/0,872694/0,837621/0,859945,
mean0,862924/worst0,837621. Современные pretrainedweights не были доступны
на исторических cutoff, исходные валидации ограничены cutoff.

Контроль Chronos объём с точно такой же P29формой:0,824305/0,844025/
0,829896/0,871123. TimesFM сильнее на первых трёх, слабее на четвёртом;
отдельный016 не улучшает лидера015, сохранён как отрицательный архив.
Нормализация weekday и входной normal-demand imputer обучены только на
прошлом; futurecalendar и известные operations допустимы. API мутирует
список inputs, поэтому передаются копии. Quantilecache имеет хеш контекста,
весов, wheel/config и самого файла. Веса остаются на Zhores.
Воспроизведение: sbatch jobs/zhores_portfolio_timesfm.sbatch;
точные исходники source_timesfm, snapshots external/timesfm*.

### P37: прямая байесовская модель горизонта

Восемь GridSamplertrials завершены в8476977; выбран route_specific=True,
strength1/uncertainty0,history224. В конце W3 возникло деление на нулевую
часовую форму отменённого50. Исправлен allocator: только подтверждённая
отмена50 и route5 получают0, отсутствие формы/объёма активного маршрута
отклоняется. Тест закрытого/активного дня и сохранения дневного объёма прошёл.
Study сохранена, trials не повторялись:8476978 завершил W3/W4/final.
8476977 FAILED14с/4CPU,8476978 COMPLETED8с/4CPU. Старыйrun_started и
исходник до исправления сохранены отдельно.

Scores0,869823/0,874400/0,767839/0,807545,mean0,829902/worst0,767839.
Увеличение числа origin×horizon примеров не обеспечило переноса;017
отклонён. Ratio target clipped±log2 и Gaussian log-loss могут не совпадать
с почасовой WAPE-потерей. Повторные targetdays имеют weight1/count;
перекрывающиеся примеры не независимы. Не расширять текущую сетку.
Проверки training target<=cutoff, context<=origin и неизменности признаков
при подмене будущей цели прошли. Воспроизведение:
sbatch jobs/zhores_portfolio_bayes_horizon.sbatch; source_bayes_horizon.

### P38: ансамбль с лидером и кандидат018

Slurm8476979 COMPLETED0:0,7с/2CPU/2GiB. Пять общих весов 0/0,25/0,5/
0,75/1, selected0,5 на meanW1/W2. Scores0,884668/0,877437/0,870160/
0,895432,mean0,881924/worst0,870160. Оба devокна лучше015 и016,
поздние слабее015. Вес0,75 имеет почти равныйdev (разница5,6e-8);
выбран формальный максимум, это не доказательство превосходства точного
веса50%. Отдельные маршруты не подбирались по outertruth.

Mean018 и014 различается на0,000077, внутри закреплённогоtie0,001.
Worst018 выше014 на0,017274:018 заменяет014 вторым в семёрке.
FinalrelativeL1 к015=4,7095%,к014=3,7634%. 014 сохранён;016/017
отрицательные архивы, не добавляют искусственные места в рекомендуемом
наборе. Лидер015 прежний, ни один кандидат не достиг0,90 на каждом окне.
Воспроизведение: sbatch jobs/zhores_portfolio_timesfm_blend.sbatch;
source_timesfm_blend и timesfm_blend/* с компонентными SHA.

ЧерезP38:GPU1359с=0,377500GPU-hour,CPU7896core-секунд=2,193333core-hour.
Учитываются оба FAILED задания. Активных заданий нет; общий предел24GPUчаса
и срок27сентября20:40:40UTC сохранены, резерв начинается15:40:40UTC.
P22 по-прежнему ждёт разрешения на конкретныйrawpayload; файлы не переданы.


Аудит послеP38:901 сохранённое окно/58внутренних final,17защищённых файлов
неизменны. Семь архивов015/018/004/005/006/009/002 сверены сraw, метрикой,
точной сеткой14640, integer/half-up/forcedzeros. Минимальный finalrelativeL1
семёрки=3.01326231%. Finalist_window_metrics.csv28строк,finalist_routes.csv280.
016/017 также сверены с их сохранённымиfinal и контрактом архивации.
Цельактивна, закрытыйрезультатнеизмерен. СтарыеCSVнеперезаписаны.


### P39: обучающая цель под часовую потерю

Slurm8476981 COMPLETED0:0,30с/4CPU/4GiB,3trials. Контроль dailyactual
точно повторяет015devraw; новаяhourlyцель selected0,88421160/0,87514707/
0,88826997/0,89952064,mean0,88678732/worst0,87514707. Meanвыше015
на0,00005183, внутриtie0,001; worstниже на0,00087229, лидер015 прежний.
Архив019 SHA90e0c6dd17d7407ff19e22f72bfa78cbbb143ecafa2cd2d67732d15c785d71b0,
finalrelativeL1 к0150,2765%. Это близкая альтернатива, не новое место в семёрке.

Новаяsurrogate optimal_volume — weighted median(actual_hour/share) завершённых
innerпрогнозов. Каждая P29форма обучена доinnerorigin, medianцели доступны
доoutercutoff. Исходныеboardings остаются отдельнымстолбцом, точно совпадают
с frozen dailyactuals;10новыхокон независимо пересчитаны. Эталон/метрика
не менялись. ContinuousL1-optimal dailytarget всё ещё обучается Gaussian
clippedlogмоделью: это суррогат, не прямое решение конкурснойпотери.
Взвешивание basevolume оказалосьхужеdev. P40 проверяет lossнакаждомчасе.
Воспроизведение:sbatch jobs/zhores_portfolio_bayes_hourly_target.sbatch;
source_bayes_hourly_target и bayes_hourly_target/independent_verification.json.

### P40: прямая медианная поправка — выполнение

Cost/APIpilot8476982 COMPLETED22с/4CPU/4GiB,actual20,075с.
Пилотпоместилсяв4GiB; sacctMaxRSS3524K не отражает памятьPython,
поэтому не используем это число. Кодstudy добавляет resource.ru_maxrss,
исходникпилота сохраняетсявsource_hourly_quantile/pilot_source.

Передstudy закреплены3Gridtrials alpha0,001/0,01/0,1,seed42,
meanW1/W2,общийtimeout1800с,per-fitHiGHStime_limit120с.
Предикторы annualP34 прежние, completedinner61days иhistory224.
QuantileRegressor q0,5 учит actual_hour/base_hour−1 с весомbase_hour/count:
необрезанный fit loss пропорционален непрерывнойпочасовойабсолютнойошибке;
L1коэффициентоврегуляризует. Futurefactor clipped0,5–2; integer half-up
только приэкспорте. Это pointмодель, безBayesianposterior.
НеполнаясходимостьLPотклоняется; testsconstantfactor0,8 иfuturetargetguard
прошли. Полныйрезультатещёнеизмерен, источникневыдаётсязаулучшение.


### P40: прямая часовая L1 — итог

Первые3trials исчерпаны. Alpha0,001: второе devрешение не сошлось за120с;
alpha0,01 выбран поdev0,87765633; alpha0,1 слабее. Первое study8476983
FAILED441с: selectedalpha0,01 не завершилW3 за120с. Неполное решение
не опубликовано. Единственный backstop8476986 использует тот же LP
и selectedalpha0,01 с native solverhighs-ipm, без новыхtrials. Первый
familydeadline started_at+1800с сохранён. Пять refit записаны отдельно
в hourly_quantile/ipm_refit, первая study/source сохранены.

IPM COMPLETED100с/4CPU/4GiB, actual97,410с, peak990604KiB≈0,945GiB.
W1/W2 raw и опубликованные значения точно совпадают с первым решателем.
Сумма двух основных allocation541с укладывается в объявленныйsolverbudget1080с.
Результат:0,87930209/0,87601056/0,81427953/0,89894310,
mean0,86713382/worst0,81427953. W3 bias−1 899 037. Это отрицательный
перенос, сетка не расширяется. 11 новых сохранённых окон пересчитаны,
final14640 проверен и сохранён как020_hourly_quantile.
SHA cf8c6d36999a12a385b9f3fffa486dad476d0d296508effeeceacc5c9c82154c.

Воспроизведение: sbatch jobs/zhores_portfolio_hourly_quantile.sbatch pilot,
затем тот же job без аргументов; из исходной завершённой study —
sbatch jobs/zhores_portfolio_hourly_quantile.sbatch finish-ipm.
Точные фазы кода: source_hourly_quantile/pilot_source,
first_study_source и ipm_source; solver_spec/ipm_started/completed/fit JSON,
SQLite, ошибки trials и независимая проверка сохранены.

### P41: подтверждённая дата восстановления движения

Первичный [Московский метрополитен](https://www.mosmetro.ru/news/details/7570)
от11августа сообщает восстановление с11августа; публикация
[МосТранса](https://transport.mos.ru/mostrans/all_news/125800) от13августа
подтверждает событие. Исходное объявление от9июля ожидало до28дней,
старый featureperiod заканчивался6августа. Новый период10июля–10августа
включительно. Дата не подбиралась по целевым значениям. Источники и
ограничение прямого fetch сохранены external/july_restoration_sources.json;
ретроспективный внешний режим разрешён организаторами.

Два GridtrialsFalse/True, прочие P34параметры фиксированы. УчительP20,
causalinner/directcontexts и Bayesianfit пересчитаны в отдельных
bayes_volume_verified_july, direct_daily/july_verified и verified_july;
старыеcache не перезаписаны. Часовая формаP29 фиксирована.
Slurm8476985 COMPLETED34с/4CPU/4GiB, selectedTrue.
КонтрольFalse exact015devraw; W1True exact015. Новые scores:
0,88329779/0,87967559/0,89061998/0,90007588,
mean0,88841731/worst0,87967559. Против015 mean+0,00168183,
worst+0,00365622; closedscore не измерен. 11 окон/innerdirect иfinal
проверены; архив021 заменяет близкий015, differencefinal0,3466%.
SHA aa88d13509b1c1ca141df86c47cb13be68ae5fbdcd2eca308e49e0c3c61dfe98.
Воспроизведение:sbatch jobs/zhores_portfolio_verified_july.sbatch;
source_verified_july, study/inner/posterior/factors и independent_verification.

### P42: обновлённый Bayesian × TimesFM

Grid5общихweights0/0,25/0,5/0,75/1, seed42, meanW1/W2,
timeout1800с, без подбора отдельных маршрутов поoutertruth.
Slurm8476987 COMPLETED7с/2CPU/2GiB. Endpointsraw точно021 и016.
Selectedbase_weight0,75:0,88460637/0,88071294/0,88418579/0,90159810,
mean0,88777580/worst0,88071294. Среднее слабее021 на0,00064151,
внутриtie0,001; минимум выше на0,00103736, поэтому022 первый.
Три окна лучше021, W3 слабее на0,00643419. 14 новых сохранённых окон
иfinal независимо пересчитаны. Difference022/0212,3596%,10 418 часов;
022/0182,4759%. Это близкая альтернативная смесь, не новая отдельная
архитектура. 021 и022 сохранены в семёрке, старый018 остаётся архивом.

TimesFM использует прежний P36normalinput со старой Julyграницей;
исправлен только компонент021. Новые weightsmodern иERA5 ретроспективны.
Оба ограничения сохранены, этот ансамбль не объявляется оперативно
доступным в исторический cutoff. Nohiddenqualityconfidence:
условныйBayesianposterior не учитывает всю неизвестность новых режимов.
Воспроизведение: sbatch jobs/zhores_portfolio_verified_timesfm_blend.sbatch;
source_verified_timesfm_blend и verified_timesfm_blend/*.
Архив022 SHA88bfdfe5c116b4bae7867feca2377e2dd4303ff2d7d9b4b75ce30dad90f0104d.

### Аудит и ресурсы после P42

955 окон/62 внутреннихfinal,7 рекомендуемых архивов с точными14640ключами,
integer/half-up/forcedzeros. 17 исходных защищённых файлов иконтрольChronos
неизменны. Finalist_window_metrics28строк,finalist_routes280.
Минимальный relativeL1финалистов2,35963048%, между021/022.
Новых конкурсных отправок, замены пакета сервиса и коммитов нет.

CPU10418core-секунд=2,893889core-hour, GPU1359с=0,377500GPU-hour.
FAILED8476983 учтён; все jobsP39–P42терминальны, активных заданий нет.
Checkpoint accounting_checkpoint_P42.json. Ресурсный предел24GPUh,
deadline27сентября20:40:40UTC, резерв15:40:40UTC прежние. P22 ждёт
явного разрешения на перенос двухrawCSV: их не передавали.


### P43: неопределённость шума и коэффициентов

Общий predictstd sklearn включает коэффициентную variance и1/alpha шум.
[Официальная документация](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
описывает predictiveStd/sigma/alpha; точная centeredSigma формула проверена
по installed sklearn/_bayes.py. Проверены3фиксированных механизмаattenuation:
predictive(контроль),epistemic sqrt(max(0,sd²−1/alpha)),none.
Обучение, нормализация, годовые признаки маршрутов, история 224 дня,
подтверждённые операции июля и часовая форма сохранены.
Использованы готовыеcausalP41innerdaily иrawP20, parentP21 не переписывается.
Эпистемическаяvariance условна относительноfit, не учитываетunknownregime,
modelmisspecification и не являетсяclosedscoreconfidence; библиотека не
оценивает posteriorvariance intercept отдельно.

GridSampler3trialsseed42,meanW1/W2,1800с общийfamilylimit. Slurm8476989
COMPLETED7с/2CPU/2GiB. Devmeans predictive0,8814866909,
epistemic0,8798800591,none0,8730650594. Контроль победил; selectedscores
точно021 на всех4окнах, все5raw включаяфинал совпали с021, финальныебайты
одинаковы. ДубликатCSV не создаётся, рейтинг послеP42 прежний.
Новыеmodes не сталифиналистами; их2devокна сохранены какотрицательнаяпроверка.

Source_uncertainty_components содержитprelaunchcode/test/job. 41input/code
SHA сверены сactualrun. Remoteportfolio_combine.py оказалсяпрежнимreader:
единственнаяdiff — float_precision round_trip отсутствовал. Точнаяactualcode
сохранена дополнительновruntime_source; plannedcanonicalsource сохранён.
Все сохранённыеraw/half-up/метрики проверены независимымround_tripreader,
контрольточный и17protectedSHAunchanged. Дляследующихstudies remotehelper
синхронизирован. Исходнаяstudy не перезапускается, бюджет не увеличивается.
VersionsPython3.12.13,numpy1.26.4,pandas2.2.3,sklearn1.8.0,Optuna4.5.0.
Тестnoise-onlyfeatures показалepi≈0 иepi=nonefactor приpositive-noise;
defaultpredictive/P41 сохраняется. Все прежниеcutoffguardchecks прошли.
Воспроизведение:sbatch jobs/zhores_portfolio_uncertainty_components.sbatch;
дляточногоas-runповтора заменить только combine.py егоruntime_sourceверсией.
SQLite, trialraw, posteriorcomponents иindependent_verification сохранены.

Дополнительнаядиагностика actualfuturedayvolume для021 дала0,915285/
0,912571/0,926272/0,923876. Это недоступнаямоделиfuturetruth, используется
толькодляатрибуцииошибки, не кандидат/не строгийпотолок. Факты не подставлены
вниодинmodelCSV. diagnostic_p43_attribution.csv отдельно.

Поиск [официальной транспортной статистики](https://transport.mos.ru/mostrans/for_journs/data)
не дал числовогоmonthlyproxy2025:индексдоступен, прямойwebfetch/curl25с
завершилсяtimeout. ДоступноеJune2026yearonyear+15% не образует2025series.
ПодтверждённыйMay11detour26 имеетнеизвестнуюдатувосстановления; длительность
не выдумывалась иfeature не добавлен. Scoutingисточники/limitations сохранены
вexternal/macro_scouting_20260927.json. Никаких новыхмакротargets нет.

АудитпослеP43:965savedwindows/63internalfinal,7архивов/14640строк,
17protectedфайлов иChronoscontrol неизменны. Это повторноиспользуемые,
перекрывающиеся окна. CPU10432core-секунд=2,897778core-hour,
GPU1359с=0,377500GPU-hour. AccountingcheckpointP43,activejobsнет.
Goalactive:bestmean0210,888417/bestworst0220,880713;
триокна ниже0,90, closedрезультат не измерен.


### P44: нелинейная модель прошлых ошибок

Проверен новый механизм: GaussianProcessRegressor из установленного sklearn
учит гладкую нелинейную поправку вместо линейной BayesianRidge. Использованы
те же годовые и маршрутные признаки P34, расхождение прямого прогноза,
контекстные отношения спроса, календарь и разрешённая ретроспективная ERA5.
Источники входов — causal P41/inner/daily и P20 raw; часовая форма P29 фиксирована.

Обучение использует только полностью завершённые 61-дневные прогнозы и
последние224дня целевых дат. Внутри origin/route каждый блок из7дней имеет
среднее clippedlog((actual+100)/(base+100)) с весами1/count(route,date).
Признаки усредняются с теми же весами; будущий недельный блок получает
единую поправку. Часовые benchmark targets, метрика и четыре окна прежние.
Недельная агрегация уменьшает размер матрицы и шум; слабо моделирует изменение
поправки внутри недели. Альфа наблюдения=.0025/effective_days.

[Официальный GPR](https://scikit-learn.org/stable/modules/gaussian_process.html)
поддерживает выбор covariance, marginal-likelihood fit и прогноз mean/std.
Здесь priorlogcorrection=0, normalize_y=False. Kernel:
ConstantKernel(.04,bounds1e-4–1) × RBF(length5,bounds.5–20),
либо Maternν1,5/2,5, плюс WhiteKernel(.01,bounds1e-4–.25).
Все continuous kernel параметры оцениваются только по прошлому likelihood,
один запуск L-BFGS-B, без random restarts. Шесть вариантов OptunaGrid:
три kernel × attenuation0/1, seed42, objectivemeanW1/W2,
общий1800с familylimit без reset. Selected RBF/attenuation0.

Pilot8476991: COMPLETED3с/4CPU/4GiB, actual0,8775с, peak197280KiB,
558weeklysamples на finalcutoff31октября, без qualityselection.
Study8476992: COMPLETED11с/2CPU/1GiB, actual8,8424с, peak237268KiB≈0,226GiB.
Память/ресурсы уменьшены после измеренного pilot. Число weeklysamples
W1/W2/W3/W4/final:162/324/405/486/558. Кэш covariance каждого cutoff/kernel
общий для двух способов attenuation. Сохранены9studyfit и1pilotfit,
все без предупреждений likelihood optimizer. Стандартное начальное состояние
и отсутствие randomrestarts делают повтор второго seed ненужным.

023:0,87765270/0,87419258/0,88654053/0,89995288,
mean0,88458467/worst0,87419258. Все четыре окна слабее021;
mean ниже на0,00383264. Поэтому новое семейство не улучшило лидера и
не расширяется без новой гипотезы. Однако mean выше004/005/006/009/002,
finalrelativeL1 к0212,62597% и к0222,78059%: это другая близкая сильная
альтернатива, включена третьей вместо чистого002. 002 остаётся control/servicebundle.
Скрытая оценка023 неизвестна; выбор не доказывает лучшего ноябрьского качества.

Обучающие цели всех10fits независимо пересчитаны из frozen hourlytruth и
pastbase, с проверкой cutoff/weekly weights. Проверены31input/codeSHA,
все16новых modelwindows иfinal14640. JSON содержит fittedkernel/theta,
likelihood, scaler и время/RSS; NPZ содержит weeklyX/target/weights/futureX,
ключи каждой тренировочной строки сохранены отдельно. Условный GPposterior
не учитывает полностью зависимость повторных окон и неизвестную смену режима;
mean/std не считаются доверительным интервалом closedscore.

Архив023 SHA e0432a12a572a90fdccd21339cbdcfa3e868b0e941067dfaf1996458ed1fba91.
Воспроизведение из ml/: sbatch jobs/zhores_portfolio_gp_errors.sbatch pilot,
затем sbatch jobs/zhores_portfolio_gp_errors.sbatch. Source_gp_errors содержит
точный Python и отдельно pilot_job4CPU/4GiB, study_job2CPU/1GiB;
для точного pilotallocation использовать сохранённый pilot_job.
Модель/данные между фазами не менялись. GPerrors SQLite/trials,
independent_training_verification и run_started/completed сохранены.

Аудит послеP44:981savedwindows/64internalfinal,7 рекомендуемых архивов
022/021/023/004/005/006/009. Все сетки14640, integer/half-up/forcedzeros;
17protected файлов и осеннийChronoscontrol неизменны. 28finalistmetrics,
280routes, minimumrelativeL12,35963048% (021/022).
Это перекрывающиеся и повторно использованные окна, не981 независимое испытание.
CPU10466core-секунд=2,907222core-hour, GPU1359с=0,377500GPU-hour,
активных заданий нет. AccountingcheckpointP44. Ресурсный предел24GPUh,
срок27сентября20:40:40UTC и резерв15:40:40UTC прежние. Цель≥0,90 на каждом
окне остаётся активной; bestmean021=0,888417, bestworst022=0,880713.


### P45: часовая форма по изменению маршрутов

Три рецепта при фиксированных объёмах021: контроль,50/50 и новая форма.
Два признака — сокращённый/объединённый маршрут — учатся по pastnormal и
изменённым дням. Официальная отмена50 исключена из traininghourshares;
даты April17/August7/Julyverified/autumn закреплены прежними источниками.
Daily shares HGB absolute_error,150iterations,63leaves,minleaf40,l2=10,
seed42, календарь и дневнаяERA5. Нормализация сохраняет каждый дневной объём.

OptunaGrid3,meanW1/W2,1800с familylimit. Devmean0,88148669control,
0,88133005half,0,88022332full. Контроль победил: всё выбранное побайтно021.
НовогоCSV нет. Оба новыхdevshape проверены на сохранение сумм/forcedzeros,
13input/codeSHA иfuturepoison проверены. Полная проверка результатов:
991savedwindows/65internalfinal;7 рекомендуемых архивов прежние,
17protected файлов и Chronoscontrol неизменны. Независимого теста нет.

Slurm8476994 COMPLETED11с,4CPU/2GiB. cumulativeCPU10510core-секунд=
2,919444core-hour,GPU1359с=0,377500GPU-hour. AccountingcheckpointP45.
Source_operations_shape, SQLite/trials/raw, run_started/completed,
independent_verification и Slurmallocation/log сохранены.
Воспроизведение: sbatch jobs/zhores_portfolio_operations_shape.sbatch.
Порог≥0,90 на каждом окне остаётся недостигнутым; продолжениеP46 проверяет
почасовуюERA5 при фиксированном021dayvolume.


### P46: почасовая погода при фиксированном дневном объёме

Сохранён публичный rawJSON ERA5 Open-Meteo для Москвы55,7558/37,6173,
31.12.2024–01.01.2026, timezoneEurope/Moscow, UTC+3. Источник:
[Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api).
API осадки вh относятся к предыдущему часу: bucket[h,h+1) использует APIh+1;
температура/ветер — среднее мгновенийh,h+1. Пограничные дни обеспечивают
три previousbucket дляJan1 и precipitationDec31hour23. Нет weatherimputation.
Запрос/model/units/returnedgrid55,75/37,5/elevation140, retrieval2026 иSHA
76c7b02ca1bd503914ab19737101c0f184b7823d3a863607195affd843eb9df2 сохранены
вexternal/weather_hourly_source.json. Это разрешённый ретроспективный источник,
не прогноз погоды на61день, известный в2025.

Прежняя P29 normalhistory≤cutoff, calendar/dailyweather,150iterations,
63leaves,minleaf40,l2=10,seed42. Добавлены только4continuous features:
часовая температура минус дневнаяmean,log1p bucketprecipitation,
log1p суммы осадков предыдущих3bucket,wind/10. Operationsflag отключены:
P45 не помог. Дневные объёмы021 сохраняются нормализацией shares.
GridSampler3 control/half/full,seed42,meanW1/W2,1800с familylimit.

Slurm8476995 COMPLETED11с/2CPU/1GiB,actual9,0631с,peak233692KiB≈0,223GiB.
Devmean0,88148669control,0,87809413half,0,87208954full;
halfW1/W2=0,88324801/0,87294025,
fullW1/W2=0,88241243/0,86176664. Гипотеза отклонена:
победивший контроль имеет всё те же5raw+finalCSV021 побайтно.
Новый дублирующий архив не создаётся. W3/W4 не использованы для выбора.

Проверены15input/codeSHA,8760aligned rows независимо восстановлены изJSON,
суммы каждого route-day/forcedzeros двух новыхформ иcutoffpoison.
Source_hourly_weather_shape, SQLite/trials, raw, runtimeversions,
run_started/completed, independent_verification, Slurmallocation/log сохранены.
Воспроизведение изml/: sbatch jobs/zhores_portfolio_hourly_weather_shape.sbatch.


Аудит послеP46:1001savedwindows/66internalfinal;7 архивов14640строк,
17protected файлов иChronoscontrol неизменны. Все окна повторноиспользованы
и перекрываются;1001 не означает1001 независимое испытание.
CPU10532core-секунд=2,925556core-hour,GPU1359с=0,377500GPU-hour,
activejobsнет; accountingcheckpointP46. Goal≥0,90 накаждомокне active,
bestmean0210,888417/bestworst0220,880713. Закрытый score неизвестен.


### P47: байесовские ошибки часовых долей — новый024

Новый механизм обучается по прошлым часовым прогнозам P29 изP39inner_shape,
которые построены из каждой origin без будущихtarget. Используются только
полностью завершённые61-дневные периоды, последние224дняtargetdates.
Raw/source.json имеют frozenSHA иfit_latest=origin. Обучение исключает5,
hours1–4, официальные измененияApril17/Julyverified/August7/autumn идни
actualday<=max(500,0,35normaltypical(route,daytype)). Targetsbenchmark прежние.
Фильтрация нужна только для обучения обычныхshares; при оценке все строки
frozen сетки остаются. Daytotal/typical обучающего ряда известны≤cutoff.

Учится clippedlogshareerror:
log((actualhour+1)/(actualday+20))-log((pastforecast+1)/(pastforecastday+20)),
clip±log2. Weighted least squares — суррогат часовойWAPE, не её точныйminimizer.
Вес1/count(route,date,hour)×pastforecastimportance, meanimportance1;
это уменьшает повторную рольtargets, но не делает overlaps независимыми.
380features:route×hour180,weekdaynonbase6×hour120,annualsin/cos и
разностьphase target-origin×hour80. StandardScaler толькоpast.
[BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)
оценивает prior/noise по данным; используется установленныйsklearn1,8,0,
max_iter300,tol1e-5. Factor exp(mean) clip0,5..2, затем нормализациячасов
к прежнему дневномуrawобъёму021. No newdailyvolume model/weatherfeatures.
УсловнаяGaussianvariance по зависимым наблюдениям не являетсяCIhiddenquality;
из менее года не следует подтверждённая годовая сезонность.

Pilot8476997 COMPLETED6с/4CPU/2GiB:actual3,7956с,peak1324064KiB≈1,263GiB.
На finalcutoff8origins/71180строк/380features,8EMiterations.
Ноябрьские targetfacts отсутствуют; pilot измеряетcost иintegrity.
Study8476998 COMPLETED21с/2CPU/2GiB,лимит5мин, послеpilot ресурсыснижены.
GridSampler3control/half/full,seed42,meanW1/W2,1800сfamilylimit.
Devmean0,88148669control,0,88302727half,0,88202474full. Selectedhalf.
TrainrowsW1/W2/W3/W4/final21800/43600/53700/63300/71180,
EMiterations9/8/9/10/8, все меньше300.

024 score0,88474981/0,88130473/0,88815199/0,89954515,
mean0,888437919/worst0,881304733. Mean≈021 (+0,000020607),
W1/W2 лучше021,W3/W4 нижена0,002468/0,000531. Worst лучше021 на0,001629,
и022 на0,000592. По frozen rule при mean gap<0,001 предпочестьworst,
024 занимаетпервое место. Не достигнут0,90 накаждомокне.
Finaltotal12806186, rawdaytotals совпадают021, различие sumsпослецелочисленного
half-up естественно. RelativeL1final024/0212,67318% (10666 changedhours),
024/0223,74414% (10628),024/0233,71759% (10641). Это новый hourshape scenario,
не дубликат. В рекомендуемой7 заменил самыйслабыйmean009; архив009 сохранён.
Порядок024/022/021/023/004/005/006; основной пакет002 прежний.
SHA024d6c2cfc3ce7f6613ec2131f265d13b5bcc93f0bb73b702fbfdd130c9d1c7f505.
Скрытыйscore неизвестен; четыре повторноиспользованные окна неindependenttest.

32input/codeSHA обеих фаз совпали;6fits обучающиеtargets иweights независимо
восстановлены по frozenactual иoldrawforecasts, всеlatestdate<=cutoff.
Проверены20futureposteriorcells каждогоfit по сохранённымcoef/covariance/scaler,
весь fixedrawdayvolume/forcedzeros; predictorfeatures не используютtruth.
Pilot4CPU иstudy2CPU finalfullraw согласуются до1e-10relative/1e-8absolute.
EM/SVD deterministic, второйrandomseed не нужен; Gridseed42фиксирован.
Source_bayes_shape, отдельноpilot_job4CPU/study_job2CPU, SQLite/trials,
training.csv, posterior.npz, factors/fit.json, run_started/completed,
independent_training_verification иSlurmallocation/log сохранены.
Для точного воспроизведения изml/ сначала восстановить jobs/zhores_portfolio_bayes_shape.sbatch
из artifacts/portfolio_20260926/continuation/source_bayes_shape/pilot_job.sbatch,
затем sbatch jobs/zhores_portfolio_bayes_shape.sbatch pilot. Послепилота восстановить
этотжеjobsфайл изsource_bayes_shape/study_job.sbatch и запустить sbatch безpilot.
Так worker получает прежние4/2threadvars соответственно. Модель между фазами прежняя.

Итоговый аудитP47:1011savedwindows/67internalfinal,7архивов×14640строк,
integer/half-up/grid/zeros. 17protectedфайлов иChronoscontrol неизменны.
28finalistwindow rows/280route rows, minrelativeL12,35963048% у021/022.
CPU10598core-секунд=2,943889core-hour,GPU1359с=0,377500GPU-hour;
accountingcheckpointP47. Goalactive:bestmean/worst0240,888438/0,881305,
ниодноиз4новых оконне≥0,90. Closedscore не измерен.


### P48: дневной ансамбль с новой часовой формой — новый025

Объединяются rawdayvolumes fixed024 иfixed016TimesFMnormal_ratioq,3;
часовые доли024 сохраняются через существующийtransplant. Это комбинация
P42 complementaryvolumeerrors иP47hourcorrection, не новоеmodelсемейство.
Weights0240/,25/,5/,75/1, GridSampler5,seed42,meanW1/W2,1800сfamilylimit.
Weight1 exact024,weight0 имеетTFvolume и024shares (неoriginal016CSV).
Все входы frozen, модели не переобучались; W3/W4 не objective.
СтараяJulyграница вTFcomponent остаётся ограничением.

Slurm8477000 COMPLETED7с/1CPU/1GiB;actual4,776с,peak212968KiB≈0,203GiB.
Devmean0,87875821/0,88204298/0,88389448/0,88429591/0,88302727.
Выбран вес024,75: W1/W2/W3/W4
0,886150391/0,882441431/0,881955939/0,900888754,
mean0,887859129/worst0,881955939. Три окна лучше024,W3 слабее на0,006196.
Mean ниже024 на0,000578790 (<0,001), worst выше на0,000651207:
frozen tiebreak ставит025 первым. Mean best024 остаётся0,888437919.
По сравнению с022 mean выше на0,000083328 иworst на0,001242995;
025 заменяет022 в рекомендуемой7, архив022 прежний.
Порядок025/024/021/023/004/005/006. Никакой скрытыйscore не измерен;
0,90 накаждомокне не достигнут, цельactive.

Finaltotal12561064,SHAa4e606d5920155d713e9b3fd100219ae1d67961c3c9fb640f11c8534e7bbcea9.
RelativeL1025/0242,3595305% (10392hours),025/0222,6707455% (10660),
025/0213,3033655% (10700),025/0233,7411818% (10662).
14input/codeSHA совпали;15raw (10trialdev+5selected) независимо восстановлены
из weighteddaytotals и fixed024shares; structuralzeros иexactcontrolпроверены.
В selected всё5cutoff используют один общийвес; половинные integer округленыonce.
Source_bayes_shape_timesfm_blend, SQLite/trials/study_spec,rawpredictions,
run_started/completed, independent_verification, log/allocation сохранены.
Воспроизведение изml/: sbatch jobs/zhores_portfolio_bayes_shape_timesfm_blend.sbatch.
CPU10605core-секунд=2,945833core-hour,GPU1359с=0,377500GPU-hour;
accountingcheckpointP48. Budget24GPUh/deadline27сентября20:40:40UTC прежние.


Аудит послеP48:1025savedwindows/68internalfinal,7архивов×14640строк,
17protected файлов иChronoscontrol неизменны. 28finalistwindow/280route rows,
минимальныйrelativeL12,3595305% у025/024. Окна повторноиспользованы и
перекрываются,1025 не означает1025 независимых испытаний. Goalactive;
bestmean0240,888438/bestworst0250,881956. Скрытыйscore неизвестен.


### P49: TimesFM с подтверждённым календарём июля — новый026

Гипотеза исправляет calendar несоответствие компонента016: initialJul10–Aug6
оценка против подтверждённогоJuly10–Aug10/restoreAug11. Официальныеprimarysources
сохранены вexternal/july_restoration_sources.json; это разрешённыйретроспективный
режим, не событие заранееизвестноеJune2025. Никаких новыхtarget/sourceгипотез.
Normalinput factor_inputs получаетflagjuly_verified; остальные вызовы поdefaultFalse
и прежние результаты остаются. Будущиеoperationfactors используюттуже границу.
Representation normal_ratio иquantileindex3 изP36 фиксированы, TF2,5model200M,
config/weights/wheel/seed42 прежние. Часовые доли024фиксированы.

GridSampler2july_verifiedFalse/True,meanW1/W2,1800сfamilylimit;
controlFalse использует saved016dayvolumes+024shape, full4windows+final.
Где inputbytes/config/model/wheel/end/SHA совпали, исходный qcache переиспользован.
Apr30/Jun30/Jul31cache reused; Aug31/Oct31 inputs изменились,2новыхGPUforward.
NPZ содержит inputs/restoration/quantiles, metadata input/model/cacheSHA иflags.
Все5inputs/restoration независимо пересчитаны;42input/codeSHA совпали,
переиспользованные3qarrays exactoriginal. Scopefuturepoison и4days×2routes×20hours
проверены запускаемымsyntheticcheck. Первоначальный узкий syntheticJulyfixture
менял thresholdtypical sparseweekendgroup; расширенJune1–Aug12 для изоляцииmask,
не меняя model/benchmark. Измеренные160extraimputedcells прошли.

Slurm8477001 COMPLETED25с/2CPU/4GiB/1GPU2080Ti,лимит5мин.
Actual6,1259с послеmodelsetup, peakGPU983662592bytes≈0,916GiB;
Pythonpeakhost2425624KiB≈2,313GiB. ДваTrue/FalseGridtrials, nofinequantilesearch.
SelectedTrue development0,880726721 vsFalse0,878758215.
Matchedcontrol0,882844664/0,874671766/0,836535824/0,859636485,
mean0,863422184/worst0,836535824.
0260,882844664/0,878608779/0,840030284/0,862611173,
mean0,866023725/worst0,840030284. W1 неизменен, остальныеулучшены
на0,003937013/0,003494461/0,002974688. Mean+0,002601540.
Fullbenchmarktargets/windows не менялись, все4windows повторноиспользуются.

026 заметно слабее лидеров, но выше004/005/006 поmean ивыше006 поworst.
Finaltotal11740106,relativeL1к0257,68623% (10657hours),к0249,96962%(10693),
к0068,51628%(10890). Это отдельная pretrainedvolume alternative, пятая в7,
заменяет006; старыйCSV006 сохранён. Порядок025/024/021/023/026/004/005.
SHA0265b6716b78a5ebb979e2424776f28765064561bc274e44091cbfe9b55a8a07667.
Bestmean0240,888438/bestworst0250,881956 прежние; goal≥0,90everywindow active.
Новый026 разумно проверить какensemblecomponent вместостарого016;
текущий026 самнепредставляется улучшениемleading/closedscore.

Source_timesfm_verified_july, defaultregressiontests, SQLite/trials,
control/selected/raw/quantileNPZ/JSON, run_started/completed,
independent_verification, Slurmallocation/log сохранены.
Воспроизведение изml/: sbatch jobs/zhores_portfolio_timesfm_verified_july.sbatch.
Modernweightsavailability иусловнаяпредобученная модель прежние;
историческаядоступностьweights2025 неутверждается. Source42SHA проверен,
weightsSHA проверялся наZhores доmodelload, локальныйcheckpoint нескачивался.
CPU10655core-секунд=2,959722core-hour,GPU1384с=0,384444GPU-hour;
remainingGPU23,615556h,accountingcheckpointP49. Budget/deadline прежние.


Аудит послеP49:1037savedwindows/70internalfinal,7архивов×14640строк,
17protected файлов иChronoscontrol неизменны. 28window/280routefinalist rows,
minimumrelativeL12,3595305% у025/024. Это повторноиспользованные,
перекрывающиеся окна, не1037 независимых испытаний. Лучшие локальные
mean/worst не изменились; критерий≥0,90накаждомокне недостигнут.

### P50: обновлённый TimesFM в ансамбле — отрицательный результат

Fixed024/026dayvolume, fixed024shares,5OptunaGridweights; W1/W2 objective,
seed42/1800с/no retuning. Slurm8477002 COMPLETED8с/1CPU/1GiB.
Devmean weights0/,25/,5/,75/1:0,880726721/0,883399333/0,884673442/
0,884555771/0,883027269. Выбран,5; scores0,886213092/0,883133792/
0,871521533/0,895179415,mean0,884011958/worst0,871521533.
Слабее025 поmean иworst, новогоархива нет; рекомендуемыйнабор прежний.
14input/codeSHA/15raw independentlychecked, exact024/026 endpoints,
weightedroute-daytotals/fixedshares/zeros проверены. Source snapshot,
SQLite/trials/run_started/completed/raw/log/allocation/independent_verification сохранены.
Воспроизведение изml/: sbatch jobs/zhores_portfolio_bayes_shape_verified_timesfm_blend.sbatch.
CPU10663coreseconds/GPU1384seconds, accountingcheckpointP50.
Порог≥0,90everywindow недостигнут. Next: absolute-loss hourlyforecast error model.

Аудит послеP50:1051savedwindows/71internalfinal,7CSV×14640строк,
17protectedfiles/control002 unchanged. Перекрывающиеся reusedwindows,
не1051 независимых испытаний. Recommended7 прежние.

### P51: часовые ошибки с абсолютной потерей — альтернатива027

P47Gaussianlogshareloss не совпадает с hourlyL1. Новый HGB учится на ошибке
actualhour/actualday minus savedpastforecastshare. Весactualday/repeats,
mean1; соответствует L1при trueactualdayvolume доprojection, остаётся суррогатом
после нормирования/округления и при ошибке будущего dayvolume. Futureactualday
не используется: будущие features только исходный rawforecast/calendar/origin.
Дневныеrawобъёмы021фиксированы; clippingnegative shares и existingtransplant
renormalization/fallback. Это completedforecast error learning, в отличие от
rawhoursharesP29/P45/P46. Всеfrozenfacts/targets/windows прежние.

PastP39inner_shape sources ending≤cutoff/latestfit=origin, recent224days,
normalJulyverified/Aug7/Apr17/autumn и reliabledayfiltersP47переиспользованы.
12features route/hour/effectiveweekday, annualtarget+origin sin/cos,
horizon/basepredshare/logbasevolume/off/summer. HGB lossabsolute_error,
100iterations,learning0,05,minleaf120,l2=1,noearlystop,seed42;
leaves15/31 иhalf/fullmix плюс exact021control. GridSampler5recipes,
meanW1/W2,1800сwholefamily,no finegridafterselection.

Pilot8477003 COMPLETED5с/4CPU/2GiB, actual2,6287с,peak213928KiB≈0,204GiB;
largest31/fullfinal71krows. Request reduced2CPU/1GiB/5мин aftermeasurement.
Study8477004 COMPLETED19с,actual16,6622с,peak272288KiB≈0,260GiB.
Dev control/half15/full15/half31/full31:0,881486691/0,884175702/
0,884682592/0,884132469/0,884450848. Selectedfull15:
0,887640735/0,881724448/0,880425643/0,897080538,
mean0,886717841/worst0,880425643. Better021 W1/W2, weaker W3/W4;
weaker025 mean/worst, strongest local unchanged. Against023 mean+0,002133168,
worst+0,006233065, so useful alternative hourlyloss mechanism.

027 fixedfinal14640rows,rawdayvolumes021unchanged, integerfinaltotal12806165;
SHA b1414e047aa8134c4395629ca5aa997339dfe6b46197d4a9e25a0374f82a7d72.
RelativeL1к0242,20127%(10633hours),к0211,37941%(10608),к0253,05855%(10681).
These are close alternatives, not independent/testproof. New027 fourth;
replaces004 in7, retaining005Chronosvolume alternative and strongerRidge-derived
volumes024/025/021. Old004archive preserved. Order025/024/021/027/023/026/005.
No hidden-score measurement and ≥0,90everywindow not achieved.

8fits (pilot1+study7),33input/codeSHA perphase, fulltrainingkeys/actuals/teacher
rawbytes/weights/target independentlyreconstructed; 15trial/selectedraw corrected
shares+fixeddayvolumes/forcedzeros exactcontrol checked. Ownmodelpickles reload
exact inZhores sklearn1,8,0; local1,9,1 inference notused dueversiondifference.
All71krows belowbinning samplingthreshold; noearlystopping randomholdout,
seed42fixed, no newmulti-seedqualityclaim. Small runnable sharetarget/futureguard/
projection/closedroute/fallbackcheck added to existingtest and passed.
Source_absolute_shape freezes pilot/studyjobs separately. SQLite,trials,
training/delta/model.pkl/fitJSON/rawforecast/log/allocation/independent_verification
and verify_saved.py retained. Reproduce fromml with savedruntime sklearn1,8,0:
restore source_absolute_shape/pilot_job.sbatch to jobs/zhores_portfolio_absolute_shape.sbatch,
sbatch job pilot, then restore study_job.sbatch and sbatch job. These srun jobs
reference canonicalworkerfile; use separate freshoutput if repeating entire study.
Independent saved-output audit: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/absolute_shape/verify_saved.py.
CPU10721coreseconds=2,978056coreh/GPU1384s=0,384444GPUh,
remaining23,615556GPUh,accountingcheckpointP51,deadline/reserve unchanged.
Next useful unstarted hypothesis: causalTimesFM proposal/disagreement in Bayesian
volume-error correction, including missing4completedoriginteachers, not retuning weights.

Аудит послеP51:1065savedwindows/72internalfinal,7uniqueCSV×14640rows,
17protectedfiles иcontrol002 unchanged. 28window/280routefinalistrows,
minimumrelativeL11,37941% у027/021 (близкие варианты часовой формы).
Reusedoverlappingwindows не1065 независимых тестов;≥0,90everywindow unmet.

### P52: TimesFM disagreement в Bayesian дневной ошибке — контроль победил

P50globalweight не улучшилleader, поэтому новая гипотеза добавила TFproposal
как supervisederrorfeature кP41annualrouteBayesianRidge.10newcolumns
log1p(TFday)-log1p(base), общий и9routeinteractions; всего107features.
P41params strength1/predictiveuncertainty1/history224/300iter/tol1e-5
сохранены. Цель clippedlog((actualday+100)/(base+100)), вес1/repeat(route,date),
pastcompletedorigins only; futureactualboardings отсутствуют. Outputdayvolumes
операционногоP41base умножены наBayesianfactor, fixed024hourshares.
GridSampler2augmentedFalse/True, seed42, meanW1/W2,1800сwholefamily;
control exact024, no TFquantile/attenuation/history/ensemble tuning.

ДевятьTFteachers Jan31..Aug31monthly плюсOct31:61days,normal_ratio/qindex3,
verifiedJuly, pin/config/model/wheel прежниеP49.5P49q reused only exactinputbytes/
config/end/model/wheel/SHA;4missingorigins Jan/Feb/Mar/May newforwards.
Dailyteacher directly q3×restoration, nonnegative, cutoff operationfactors,
route5zero. Hourlyshapes не влияют наteacherdayvolume.56source/inputSHA phaseeach,
9NPZinputs/restoration/outputs проверены;5reusedq exactoldNPZ/dayvolumes exactP49
within1e-8 arithmetic. Softwareavailable2026/pretrained иretrospectiveoperations
ограничения прежние; это не доказательство доступности этих весов в 2025 году.

Пилот8477006 FAILED19с/2CPU/3GiB/1GPU beforeforecast: Jan31normalinput не
имел route1/Sunday/hour0 observedgroup. Fourmissingcells; samepast route1/hour0
pooledweekdaymedian7 существует. Общий impute_history получил causalroute/hour
fallback для отсутствующихweekdaygroups; отсутствиеиэтогопрофиля всёещёошибка.
Benchmarkhourly_clean/targets/metric/windows не менялись.5P49outerinput/restoration
arrays побайтно unchanged. Syntheticfallback/futurepoison/missingallroutehour
checkspassed. Failed source/metadata saved separately, fullbudgetnotreset.

Retrypilot8477007 COMPLETED20с/2CPU/3GiB/1GPU;oneforward on9×31shortcontext,
actual0,8578с aftersetup,peakGPU983662592/RSS2425732KiB≈2,313GiB.
Remainingteacher8477008 COMPLETED21с withsameCPU/RAM/GPU, timelimitreduced2мин
byactualcost;3newforwards,Janpilotcache reused +5P49cache,9origins total.
Actual2,4774с,peakGPU983662592/RSS2425720KiB. No GPUmemoryfallback/newpackage.

CPUmeta8477009 COMPLETED6с/2CPU/1GiB, actual3,1889с,peak214712KiB≈0,205GiB.
AugmentedW1/W2 0,884642269/0,880896286,mean0,882769278,
control0,884749806/0,881304733,mean0,883027269. Bothdev slightlyweaker;
control selected. Selected4scores0,884749806/0,881304733/0,888151989/0,899545149
andfinal exact024. NewvariantW3/W4 notmeasured, no claimall4negative.
No newduplicateCSV; ranking025/024/021/027/023/026/005 unchanged.
Goal≥0,90everywindow unmet, hiddenquality unknown.

Meta51input/codeSHA checked,2fits107features with1098/2196pastrows.
Fulltrainingcache origins/teachers/facts/horizon/weights+futureinputs reconstructed.
Posteriorcovariance fromsaved precisions+scaledweightedpast, predictiveSD/noise
andattenuation factors reconstructed independently: maxdiff1,22e-15.
9trial/selectedraw exactfixedvariant/control and024shares/forcedzeros checked.
Initialaudit mismatch was origin datetime64ns/s parsing, fixedverifierresolution;
no model/datafit changed. Source snapshots, own inputs/q/dailyTeachersJSON,
SQLite/trials/past/future/posterior/raw/log/allocation/completed/independentverification
and verify_saved.py preserved. Reproduce fromml/withfrozenruntime:
restore source_timesfm_teachers_retry canonicalpilotjob before sbatch pilot,
then stored study_job.sbatch before sbatch teachersjob; normal_timesfm teacherflag
makesseparateoutput. Finally sbatch jobs/zhores_portfolio_timesfm_errors.sbatch.
Existing caches validated/reused; newoutput required forfromscratch reproduction.
Savedmeta audit: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/timesfm_errors/verify_saved.py.

CPU10853coreseconds=3,014722coreh/GPU1444seconds=0,401111GPUh,
remaining23,598889GPUh,accountingcheckpointP52. Failedphase included.
Audit1073savedwindows/73internalfinal,7unique14640-rowCSV,
17protectedfiles/control002 unchanged; reusedoverlappingwindows notindependent.
Next plausibleunstartedfamily: nonlinearabsolute-loss daily error model using
these9teachers, instead of anotherTFquantile/fineweightgrid.

### P53: nonlinear absolute daily error — контроль победил

НовыйHGB учится на completedforecast dailyerror ratiosactual/base, вместоP52
Gaussianlinearlogcorrection иP25 rawdaylevel. Толькоorigin/end≤cutoff, recent224,
base>0/route≠5; weightbase/repeatedtargetcount normalizedmean1. WeightedratioL1
равнаdailyL1 с общимnormalization доfactorclip; не равнаfrozenhourL1.
23compactfeatures: route/effectiveweekday/daytype categorical, horizon/off/summer,
annualtarget+origin sin/cos/differences,logbase,TF/direct/ridge/regime gaps,
ratios7/14/28,temp/precip/daylight. FixedP52TFteachers/params/q3, no newGPU.
HGBabsolute_error100iter/.05learning/minleaf50/l2=1/noearlystop/seed42,
leaves7/15 andhalf/full correction, factorclip[0,5;2]. Modeloutput dayvolume
usesP41operationsbase×factor, fixed024hourshape; exact024control.
GridSampler5recipes, objectiveW1/W2,1800sfamily/no finegridafterresult.

Pilot8477010 COMPLETED3с/2CPU/1GiB;actual0,8266с,peak180456KiB≈0,172GiB,
largest15/fullOct31fit3738rows. Study requestreduced1CPU/512MiB/2мин.
Study8477011 COMPLETED9с,actual6,5254с,peak215068KiB≈0,205GiB.
Control/half7/full7/half15/full15devmean0,883027269/0,879784281/
0,871435822/0,880086202/0,872067962. All5complete; control wins,
selected4windows+final exact024, newduplicatearchive notcreated.
Half15 bestnoncontrol0,875434658/0,884737746: W1weaker,W2better;
not discardedafteronlyW1. NewvariantsW3/W4 notmeasured, no all4weakerclaim.
Recommended025/024/021/027/023/026/005 unchanged;≥0,90everywindow unmet.

50input/codeSHA eachphase,5savedfits pilot1+study4,15raw validated.
Fullpast/futureteacher/cachedforecast/actuals/originperiods reconstructed;
actual/base ratios,base/repeat weights, lossidentity,clip/fixedshares/zeros
andexactcontrol independentlychecked. Ownmodelpickle reloadexact inZhores1,8,0;
local1,9,1 inference notused. Smalltargetloss/featurepoison/futureguard checks added
and passed; no newrandomsubset or multi-seedqualityclaim fornegativevariant.
Sources/pilot-studyjobs/training/future/model.pkl/fitJSON/SQLite/trials/raw/log/
allocation/independent_verification and verify_saved.py preserved. Reproducefromml:
restore source_nonlinear_errors/pilot_job.sbatch to canonicaljob and sbatch pilot,
then restore study_job.sbatch and sbatch job (srun referencescanonicalworkerfile).
Saved audit: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/nonlinear_errors/verify_saved.py.

Descriptive monthly slices half15 vscontrol:May0,889873275 vs0,895090003,
June0,859833242 vs0,873576871;July0,881319353 vs0,875709503,
Aug0,888308413 vs0,887149205. June newbias+9,2524%. W1completedtraining
endsApril30, so no summers amongtargets; laterW2 hasJuneobservations.
This post-resultdiagnosis motivates a training-support gate byroute/season/weekday;
no futurefactswoulddefine gate, no manualouterdatewindow rule. Not yetstarted,
not proof of hidden improvement; slices are reusedoverlappingprotocoldata.

CPU10868coreseconds=3,018889coreh/GPU1444s=0,401111GPUh,
remaining23,598889GPUh,accountingcheckpointP53. Audit1087savedwindows/
74internalfinal,7unique14640-rowCSV,17protected/control002unchanged.
1087 are reusedwindows, not1087 independent experiments; finalgate not achieved.

### P54: внешний месячный citytramproxy — scouting без модели

Проверка другойинформации: independentlypublished citytram2025monthlyride totals
could anchorannualdemand, but no compatible absolute Jan–Dec2025series retrieved.
Bounded9officialqueries gaveprojectnarratives, March2024YOY andcurrentJune2026YOY
indexedsummary; detaileddata page continuedtimeout. Three finalstrictqueries empty.
Sources/coverage/querylist/checktime inexternal/macro_scouting_P54_20260927.json.
No newfeature, noSlurm/model, no12numbers inferredfromannualsum orsinglepercentage.
Это недостаток полученныхсвидетельств, не утверждение чтоpublicseries не существует.
Scoutingfamilyblocked pendingverifiedcoverage, globalgoalactive (coveragegate next).

### P55: автоматическая поддержка сезона — полезная защита, слабее лидеров

FixedP53half15 поправка применяется только при ≥2distinctcompletedtargetdates
дляroute×meteorologicalseason×effectiveweekday вpreparedpast224days.
December-Jan-Feb winter/Mar-Apr-May spring/Jun-Jul-Aug summer/Sep-Oct-Nov autumn,
existingseasonhelper reused. Duplicateforecastorigins don'tmultiplycoverage.
Otherwise exact024hourlyraw. Gate learnedfrompastcoverage, no futuretargets,
no manualoutermonthweights. HypothesisderivedafterP53monthlydiagnosis;
reusedoverlappingwindows are not independentvalidation.

PreregisteredGrid2control/gated,seed42,objectiveW1/W2,1800sfamily,
no retuningleaves/strength. Job8477014 COMPLETED10с/1CPU/512MiB,
actual7,0889с,peak220896KiB≈0,211GiB, noGPU. Gateddev0,883389111
vs0240,883027269 (+0,000361842). Fullscores0,882040476/0,884737746/
0,877428979/0,899545149,mean0,885938088/worst0,877428979.
Smalldevgain didn'ttransfertofullprotocol: W3worse, W4exactcontrol.
Recommended7 unchanged025/024/021/027/023/026/005; noarchive028.

Supportedroute/daygroups outof610:W1 279/W2 549/W3 279/W4 0/final259.
UnknownsummerW1June/unknownfallW3Sep/allW4/winterfinalDec fall back
withoutdate-specificexceptions. FinalinternalCSV saved, notrankedaboveleaders.
51input/codeSHA,5fitsfullpastteachers/truth/target/weights/future/modelSHA/reload
and9raw forecasts independentlychecked, exactunsupported024/fixedshares/zeros.
Localsklearn1,9,1 notusedfor1,8,0pickle inference. Smallcheckfourquarters,
duplicateorigins,route/weekdaycoverageandfutureguard passed.
Source/job/studySQLite/trials/gatecounts/training/future/models/raw/final/log/
allocation/independentverification saved; audit fromml:
PYTHONPATH=. python artifacts/portfolio_20260926/continuation/seasonal_gate/verify_saved.py.
Reproduce: sbatch jobs/zhores_portfolio_seasonal_gate.sbatch, sourcefrozen
insource_seasonal_gate; use freshoutputforcleanfit. Budget10878CPUcoreseconds
=3,021667coreh/1444GPUseconds=0,401111GPUh,23,598889GPUhremaining.
General audit1095savedwindows/75internalfinal,7unique14640-rowarchives,
17protectedfiles/control002 unchanged. No ownjobsactive; goalstillactive.
Threshold≥0,90eachwindow unmet, hiddennewscoreunmeasured.

### P56: TiRex recurrent architecture — отрицательный полный результат

TiRex35M xLSTM is a differentpretrainedarchitecture fromChronos/TimesFM.
Pinnedtirex-ts1.4.2 wheelSHAe470b2e1a4ad2fe6ab95012da6b92ffa20f69ff3a5d44b5b37ba3cd86ff2b2e0;
NX-AI/TiRex63c740922493f5fbe60b277609ec62babfba2762 model.ckpt141230262bytes,
SHAb8c3f5a036c63272ce4b91c00187e26922a394cb6cb49d4e16db070ad0422314.
OfficialREADME/API/PyPImetadata/license savedexternal/tirex; Built with technology from NXAI.
No fine-tuning, weightsnotavailableathistorical2025cutoffs, experimentalresearchonly.
Model usesnativeTorchbackend float32/compileFalse/batch9/noresample/horizon61/seed42;
2080Ti7.5 doesn'tsupportupstreamcustomCUDA8+, no customkernelsinstalled.
Separate--no-deps targetfolder leavesbaseenvunchanged. TiRex2 requiresTorch≥2.8/
NumPy≥2.1 sofirstcompatibleTiRex1 tested. P36series_inputs verifiedJuly,
P49teacher_daily knownoperations andfixed024hourshares reused, no newtargets.

Download8477016 COMPLETED8с/1CPU/1GiB, publictokenFalse/pinnedweightsSHAexact.
Pilot8477017 COMPLETED11с/2CPU/3GiB/1GPU:actual2,3055с forlargestOct31context
plusidenticalrepeat,deterministicexact,peakGPU290458112bytes≈277MiB,
RSS1014364KiB≈0,967GiB. Mainrequestreduced1CPU/1536MiB/1GPU/2мин.
Study8477018 COMPLETED23с,actual15,0834с,peakGPUunchanged/RSS≈0,982GiB.
Grid7recipes control/normal_daily q0.3,.5,.7/normal_ratio q0.3,.5,.7,
meanW1/W2,1800sfamily, all7completed. Devcontrol0,883027269 wins;
normal_daily means0,850156416/0,823992748/0,773791451,
normal_ratio0,876083729/0,865821109/0,831281774.
Selected4+finalexact024. Bestnoncontrol aspreregistered beforeGPU wasalso
checkedall4: normal_ratio q0.3 scores0,881261959/0,870905499/0,795185101/
0,844062497,mean0,847853764/worst0,795185101. Worseleaders/026 and005mean/worst;
noarchive028. Negativefullresult kept, noclaimxLSTMgenerallyinferior.

19input/codeSHAeachphase,8quantilecache inputs/restoration reconstructed,
futuretruthpoison hasexactlyzeroeffect,24rawselected/trials/architectureforecasts
reconstructed fromq/fixedoperations/024shares/zeros. Ownpilotdeterministicrepeat
exact, weightSHAcheckeddownload/bothGPUjobs, wheelSHAcheckedlocal/remote.
Smallfixedshare/daytotal/gridcheck passed. Generic audit1117savedwindows/
77internalfinal,7validunique14640-rowarchives,17protected/control002unchanged.
Modernweights/retrospectiveops documented; reusedwindows notindependent.
Source/job/version/modelmetadata/package/license/OptunaSQLite/trials/input/qNPZ/
raw/metrics/internalfinals/log/allocation/independent_verification preserved.
Auditfromml: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/tirex/verify_saved.py.
Reproduce: sbatch jobs/zhores_portfolio_tirex_download.sbatch, then restore
source_tirex/pilot_job.sbatch tocanonicaljob before sbatch pilot, finally restore
study_job.sbatch before sbatch study. Freshoutputforcleanreplay, cachedq guarded
bycontext/config/checkpoint/wheel/end/cacheSHAs. No serviceweightschanged.
CPU10931coreseconds=3,036389coreh/GPU1478seconds=0,410556GPUh,
remaining23,589444GPUh. No ownjobsactive. Recommended7unchanged;
≥0.90eachwindowunmet, hiddenqualityunmeasured, goalactive.

Descriptive024route/daydiagnostic in p56_baseline_route_diagnostic.csv and
p56_baseline_daily_diagnostic.csv: route26/28 overpredict W1/W2 while17underpredicts.
Next unstartedhypothesis: separate networkdayvolume fromconditionalroutefractions,
learn routeallocation errorfromcompletedpastforecasts, preserveglobalvolume.
AnalogousP47hourshares butrouteallocationnewtarget, notyetP57launch orproof.

### P57: Bayesian route allocation — небольшой близкий внутренний вариант

Newrelative targetactualroute_day×predictednetwork_day/actualnetwork_day
forcompletedorigin/date separates relativeallocation fromcommonnetworkvolume.
ExistingP41annual97features/BayesianRidge/P52validatedmonthly examples reader;
fixedstrength1/uncertainty1/predictive/recent224, baseP41operationsfuture×factor.
Normalize to024networkdayrawtotal, preserve024hourshares; mixcontrol/half/full.
Grid3seed42/objectiveW1/W2/1800sfamily, no futurefacts/windows/datachanges.
UnusedTFteachercolumn remains inreusedreader, notinP57features.

Pilot8477019 COMPLETED6с/1CPU/512MiB,actual0,5391с,peak180328KiB≈176MiB;
study8477020 COMPLETED8с,sameallocation,actual4,8692с,peak223588KiB≈0,213GiB.
Devcontrol0,883027269/half0,883040879/full0,883028068; halfselected.
Scores0,884263461/0,881818297/0,888409458/0,899212979,
mean0,888426049/worst0,881818297. Min+.000513565 vs024, mean-.000011871;
stillweaker025minimum. Finaldifference0240,438413%/9000hours,
finaltotal12806180 vs02412806186; publishrounding, rawnetworktotalsfixed.
Noarchive028forP57: smallclosevariant, internalforecastretained.
52input/codeSHAeachphase,6fits/11raw, fullpasttargets/networktotals/future/
Bayesianposteriorcovariance/attenuation/fixednetworkvolumes/024shares/zeros checked.
Smallrelative-target/scaleinvariance/networkvolume/structuralzeros/futureguard
checkpassed. Sources/job/SQLite/trials/past/future/posterior/raw/final/log/allocation/
difference/independentverification saved. Auditfromml:
PYTHONPATH=. python artifacts/portfolio_20260926/continuation/route_allocation/verify_saved.py.
Reproduce: sbatch jobs/zhores_portfolio_route_allocation.sbatch pilot, thenstudy;
freshoutputforcleanreplay. BudgetcheckpointP57CPU10945/GPU1478seconds.

### P58: nonlinear route fractions — новый лидер028

Differenttarget/loss fromP57Gaussianlogratio andP53absolutevolume ratio:
actualroutefraction−pastpredictedroutefraction, weightedactualnetworkvolume/
repeat(route,date), normalizedmean1. WeightedshareL1 isdailyrouteL1 surrogate
atactualnetworkvolume, notfrozenhourloss. Networktotalscomputedall10routes
beforefilter route5/base0/recent224; targetsallcompletedmonthly61dayforecasts.
P53compact23features +base_route_fraction=24, includesvalidatedTFdisagreement
andcalendar/weather/history/origin signals; no futureactuals. Futureestimated
routefractions=max(0,base_share+modelerror); structuralbase0 stay0; normalize
024networkdayrawvolume,024hoursharesfixed. HGBabsolute_error100iterations/.05/
15leaves/minleaf50/l2=1/noearlystop/seed42, no feature/leavessearch.
Grid3control/half/full,meanW1/W2,1800sfamily; seed73fixedselectedrecipe check.
Retrospectiveofficialops/ERA5/modernTFteacher context unchanged anddocumented.

Pilot8477021 COMPLETED4с/1CPU/512MiB,actual0,9590с,peak180136KiB≈176MiB;
study8477022 COMPLETED14с,sameallocation,actual11,4000с,peak224416KiB≈0,214GiB.
Devcontrol0,883027269/half0,883438055/full0,882687198, all3complete; halfselected.

| Вариант | W1 | W2 | W3 | W4 | Среднее | Худшее |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 028 nonlinear route fraction half | 0,884368483 | 0,882507627 | 0,887364626 | 0,900115549 | 0,888589071 | 0,882507627 |
| 024 parent | 0,884749806 | 0,881304733 | 0,888151989 | 0,899545149 | 0,888437919 | 0,881304733 |
| 025 previous recommended first | 0,886150391 | 0,882441431 | 0,881955939 | 0,900888754 | 0,887859129 | 0,881955939 |

028bestcurrentmean/min, smallimprovement onreusedoverlappingwindows,
notindependentvalidation orhidden-scoreproof. Vs024mean+.000151152/
minimum+.001202895; vs025mean+.000729942/min+.000551688.
W1weaker025/W2W3better/W4slightlyweaker. Seed73 all4metrics andall5raw/finalCSV
exactlysame, no seedselection. Rawnetworkdaytotals preserved, roundingmakes
publishedfinaltotal12806224 vs02412806186 (+38counts).
Finaldifference028/0241,24257%/9901hours,028/0252,62571%/10281hours,
028/0272,53992%/10660hours. Pairwiseall7tableinfinalist_diversity.csv;
closestpair028/0241,24257%. Archive028createdexclusively, unchangedoldarchives:
SHA1dac3b146ed518db2ec95133efc66d8d5befad559d43ca2b9575d2f51c121c4e.
Recommendedorder028/025/024/027/023/026/005; removes021ascoveredbase,
strong021retainedarchived (its dailyvolumes areparents024/025/028).

43input/codeSHAeachphase,11fits/16trial-selected-seedraw, completedteachers/facts/
networkfractions/targeterrors/weights/L1identity/futureinputs/fullraw transforms
independentlychecked. ModelpickleSHA/reloadexactonZhores1.8; local1.9.1 inference
notused. Smalltarget/weightL1identity/truthpoison/futureguardcheckpassed.
Networktotals/hourshares/zeros/024control/seed73exactness verified.
Source/job/runtime/SQLite/trials/parameters/training/future/model.pkl/fitJSON/
raw/seed73/metrics/final/difference/archive/log/allocation/independent_verification
preserved. Auditfromml:
PYTHONPATH=. python artifacts/portfolio_20260926/continuation/route_fraction/verify_saved.py.
Reproduce: sbatch jobs/zhores_portfolio_route_fraction.sbatch pilot, thenstudy;
freshoutputforcleanfit, frozen source_route_fraction. No new serviceversion/model.

General audit1141savedwindows/80internalfinal,7unique14640-rowarchives,17protected
files/control002unchanged. These countsinclude repeatedprotocolwindows/trials/
seeds, notindependentexperiments. CPU10963coreseconds=3,045278coreh,
GPU1478seconds=0,410556GPUh,remaining23,589444GPUh. No ownjobsactive.
≥0,90eachwindowstillunmet; hiddennewscoreunmeasured; goalactive.
Nextunstartedmeaningfulcombination:028routeallocation +025networkdaydynamics,
fixedlearnedrecipes insteadoffineweightsearch. No P59code/study yet.


## P59 — доли028 и дневная динамика025

Grid4: два исходных прогноза и две заранее заданные композиции. По W1/W2
выбрана relative: дневной объём025 × отношение долей028/024, затем
нормализация общего объёма дня к025. Новых весов или обучения нет.
Окна: **0,88567 / 0,88316 / 0,88019 / 0,90112**, среднее **0,88753**,
минимум **0,88019**. Первые два окна улучшились, третье стало хуже025/028.
В семёрку не добавлен, новый архив029 не создавался. Исходный028 остаётся первым.
Job8477023 COMPLETED7с/1CPU/512MiB, actual4,4169с, peak210716KiB.
21input/codeSHA и13raw математически проверены: controls точны, отношение
дневных объёмов, общий rawобъём025, часовые доли и структурные нули сохранены.
[Артефакты](fraction_dynamics), [аудит](fraction_dynamics/independent_verification.json).
Команда: sbatch jobs/zhores_portfolio_fraction_dynamics.sbatch; новые пути для
чистого повтора. Общая проверка1153окон/81внутренний final,17защищённых файлов
неизменны. Повторные окна не являются независимыми тестами.

## P60 — первичный школьный календарь

Найден календарь ЛИТ1533 2024/25 (подписан29.10.2024) и письмо ДОНМ
16.05.2025 Исх-12641/25 с вариантами каникул2025/26. Таблицы проверены
визуально, PDF/text/SHA/происхождение сохранены в
[external/school_calendar/sources.json](external/school_calendar/sources.json).
Модульный составной календарь — внешний ориентир. Он не является единым
расписанием всех московских школ; доля соответствующих пассажиров неизвестна.
Традиционные календари «Старой школы» отличаются, сохранены для контекста,
не смешаны с модульными признаками. Даты не выбирались по целевым валидациям.
На этом основании выполненP61, результаты ниже.


## P61 — школьный календарь в ошибках долей маршрутов: 029

Фиксированная модель028 получила6признаков: короткие каникулы/летний перерыв,
расстояния до предыдущего и следующего начала перерыва (ограничены61днём),
короткие/летние каникулы в точке origin. Первичные даты и known_at проверены;
признаки используют только информацию, опубликованную к соответствующему origin.
Это составной модульный ориентир, с неизвестной пассажирской долей и расписаниями
других школ. Календарь не задаёт коэффициенты вручную: эффект учится по прошлым
завершённым прогнозам. Эталон, окна, очистка, half-up и модельные параметры сохранены.
Grid3: control028/half/new. Новый=.5календарная доля+.5исходный024,
half=.5новый+.5старый028. ПоmeanW1/W2 выбранnew0,883516081 противcontrol
0,883438055/half0,883483205. Все3trials завершены, новых сеток и seedподбора нет.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884367615 | 0,882664547 | 0,887414039 | 0,900572921 | **0,888754781** | **0,882664547** |

Сравнение028: среднее+0,000165709/минимум+0,000156920. W1−0,000000867,
W2+0,000156920/W3+0,000049412/W4+0,000457372. Улучшение небольшое;
это повторно использованные перекрывающиеся окна, не свидетельство скрытогоscore.
Seed73:4метрики и5rawпрогнозов/финальныйCSV точно совпадают. Общий rawобъём
каждого дня024, его часовые доли и структурные нули сохранены. Publishedtotal
12806177 против02812806224; разница47 обусловлена округлением.
029/0280,34%/8905изменённых часов;029/0241,31%/9940часов;029/0252,71%.
029 заменяет028 как обновлённая версия той же семьи, прежнийCSV сохранён.
Текущая семёрка029/025/024/027/023/026/005, минимум любой пары1,31%.
SHA029:39b856b98c9e08544f1ccfaf6c2387361a2223d1f784afd85e99e82011c94dfd.
[Архив](../../../submissions/029_school_calendar_route_fractions.csv),
[происхождение](archive_029.json), [независимый аудит](school_fraction/independent_verification.json).

Pilot8477024 COMPLETED4с, study8477025 COMPLETED15с, оба1CPU/512MiB/2мин,
безGPU; actual1,0334с/12,1215с, peak181428/224008KiB. 54input/codeSHAeachphase,
11fits/16trial-selected-seedraw проверены; прошлые факты/teacherforecast/целевые
доли/повторные веса/L1тождество/future/calendar6features восстановлены независимо.
ModelpickleSHA/собственный reloadexact наZhores1.8; локально1.9модель не загружалась.
Сохранены первичныеPDF/text/SHA/dates/scope, код/Slurm/versions/preregistered,
OptunaSQLite/trials/training/future/model.pkl/fitJSON/raw/seed73/metrics/final/
archive/difference/log/allocation/accounting. Малый запускаемый тест проверяет
границы/known_at/coverage/truthpoison. Команда: sbatch
jobs/zhores_portfolio_school_fraction.sbatch pilot; затемstudy, свежийoutputдлячистогоповтора.
Аудит изml: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/school_fraction/verify_saved.py.

Общий аудит:1167сохранённых окон/83внутреннихfinal,7unique14640-rowarchives,
17защищённых файлов и контроль002 неизменны. Счётчики включают повторные
trials/seeds, не независимые испытания. CPU10989coreseconds=3,0525coreh,
GPU1478seconds=0,410556GPUh, осталось23,589444GPUh. Активных заданий нет.
Порог≥0,90каждое окно не достигнут, goalактивен, новыеCSVнеотправлялись.
Следующая новая гипотеза: отдельная модель общего дневного объёма всей сети
по завершённым прошлым ошибкам, с долями029. ПрежниеP21/P43/P53 учили
поправки отдельных маршрутов; P59 только смешивал готовые объёмы.
P62 пока не запускался; окно и эталон сохраняются.


## P62 — общий дневной объём сети: отрицательный результат

Grid5 завершён; выбран контроль029 (dev0,883516081). Лучший новый
bayes_half: W1–W4=0,880978547/0,882757733/0,883565558/0,901675155,
mean0,887244248/worst0,880978547. Улучшение W2/W4 не компенсирует W1/W3;
нового архива нет. Текущая семёрка сохранена. Отдельные сетевые Bayes/HGB
учились только на завершённых прошлых прогнозах, с исходным operations-base;
доли маршрутов и часов029 сохранены. Pilot8477026 COMPLETED4с,
study8477027 COMPLETED10с,1CPU512MiB/2мин каждый, безGPU. Actual0,9472/7,4115с,
peak175124/216108KiB. 48SHAeachphase,9fits/20raw, networkaggregation,
веса/цели, Bayesian precision/predictivevariance/shrink и часовые переносы
восстановлены независимо. HGB собственный reloadexact. Selected совпадает029.
[Артефакты](network_volume), [аудит](network_volume/independent_verification.json).
Общий аудит1185окон/85внутреннихfinals;17защищённых файлов неизменны.
CPU11003coreseconds=3,056389coreh/GPU1478seconds=0,410556GPUh.

## P63 — проверка TabPFN v2

Первичные исходники/PyPI/modelcard/лицензия и публичность весов проверены.
Минимальный вариантtabpfn2.0.9 поддерживает median и явный путь к44MB
v2-regressor checkpoint. Репозиторийgated:false, revision/SHA сохранены.
Требуется изолированныйsklearn1.6.1; базовая среда не изменяется.
Новые версии с аккаунтом/соглашением не используются. Веса пока не скачаны,
GPU/API не проверены; это готовность к пробе, не измеренный результат.
Нетsample_weight: будущая модель будет явно невзвешенным условным prior.
Лицензия Apache-derived с атрибуцией; исходники/снимок/копия лицензии:
[external/tabpfn/scouting_snapshot.json](external/tabpfn/scouting_snapshot.json).
Современный checkpoint не объявляется исторически доступным в2025.


## P64 — TabPFN v2 на прошлых ошибках долей: близкий вариант

Built with PriorLabs-TabPFN. tabpfn2.0.9/sklearn1.6.1 установлены изолированно
без обновления базовой среды. Публичный v2-regressor44,390,977bytes,
revision4972a65a1b30806315c6f92499959ffbfc69a673, SHA
2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736.
[Лицензия](external/tabpfn/model_LICENSE.txt),
[снимок источников](external/tabpfn/scouting_snapshot.json). Современный prior
не объявляется доступным к историческим origin2025. Нативный API не поддерживает
sample_weight: это невзвешенная условная модель, весаHGB сохранены для сравнения.
30признаковP61, последние224дня завершённых прошлых61-дневных прогнозов,
median,4estimators,seed42,low_memory/float32/cuda. Target=ошибка сетевой доли
маршрута. Learnedallocation нормируется к дневному объёму024, часовые доли024
сохранены; new=.5learned+.5old024. Grid3control029/half/new выбралhalf поW1/W2
0,883557252 противcontrol0,883516081/new0,883324490. Все3trials завершены.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884623450 | 0,882491055 | 0,887987485 | 0,900509220 | **0,888902802** | **0,882491055** |

Среднее выше029 на0,000148022, худшее окно ниже на0,000173493. По правилу
близкого среднего (<0,001) и лучшего минимума029 сохраняет первый приоритет.
Seed73 фиксированнойhalf:0,884730207/0,882750915/0,887964222/0,900182077,
mean0,888906855/min0,882750915. Он не подменяет выбранныйseed42 и не выбран
по лучшему результату. Финальная разница seed42/73raw=0,307%; P64/029published
relativeL1=0,4648%,9370изменённых часов, total12806187. Новый030 не архивирован:
слишком близкий вариант с более слабым выбранным минимумом. Исследовательские
raw/финальныйCSV/seed73 сохранены; текущий набор029/025/024/027/023/026/005.
Порог≥0,90 на каждом окне не достигнут; скрытый score неизвестен.

Setup8477029FAILED7с: отсутствовал каталог metadata; исправлено создание
корневого каталога, исходник до исправления сохранён. Setup8477030COMPLETED7с.
GPU pilot8477031COMPLETED22с,2CPU16GiB/5мин,actual18,3406с;
study8477032COMPLETED53с,2CPU2GiB/3мин,actual49,0522с. PeakGPUallocated0,399GiB,
reserved0,516GiB, peakhoststudy1338648KiB≈1,277GiB. По pilot запросRAM уменьшен.
Прогнозы pilot повторились точно; свежий fitseed42Oct31 вstudy точно совпал
pilot. Legacyloader с Torch2.6 допускает загрузку исходного полного checkpoint
только после проверенногоSHA и только в окружении задания; глобальная среда
не изменена. Не использованы аккаунт/соглашение/cloudclient/API других участников.

[Артефакты](tabular_fraction), [аудит](tabular_fraction/independent_verification.json).
51input/codeSHAeachphase,11fits/16raw, completedfacts/targets/weights,
обучающие и будущие матрицы, calendarknown_at, normalization/hourshares/zeros
проверены независимо. Локально нейросеть не запускалась. Код/Slurm/версии/wheel
метаданные/лицензия/источники/checkpointSHA/OptunaSQLite/trials/матрицы/targets/
raw/final/seed73/метрики/log/allocation/accounting сохранены. Малый тест прошёл.
Воспроизведение изml: sbatch jobs/zhores_portfolio_tabular_download.sbatch;
sbatch jobs/zhores_portfolio_tabular_fraction.sbatch pilot; затем
sbatch --mem=2G --time=00:03:00 jobs/zhores_portfolio_tabular_fraction.sbatch study.
Для чистого повтора использовать новый output; budget существующегоstudy не сбрасывать.
Аудит: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/tabular_fraction/verify_saved.py.

Общий аудит1199сохранённых окон/87внутреннихfinals;17защищённых файлов неизменны,
контроль002 и7готовых архивов проверены. Счётчики включают повторные trials/seeds
и перекрывающиеся окна, не независимые наблюдения. CPU11167coreseconds=3,101944coreh,
GPU1553seconds=0,431389GPUh; осталось23,568611GPUh. Активных заданий нет.


## P65 — общие моды часовых ошибок: кандидат030

Built with PriorLabs-TabPFN. Вместо отдельных часовых поправок обучаются общие
20-часовые векторы ошибок actualhour/actualday−teacherhour/teacherday.
Использованы только завершённые месячные61-дневные прогнозы, надёжные обычные
дниP47, последние224дня. Среднее/covariance поactualday/repeatedroute-date;
базис собственных векторов рассчитан только по прошлому каждой точки отсечения.
Правило90%variance с пределом6 дало6компонент во всех окнах, объяснило65,4–70,8%.
Это ограниченная аппроксимация: порог90%разброса не достигнут из-за cap6.

TabPFNv2:50признаков (календарь/моделиP61 +20исходныхчасовыхдолей),
median,4estimators,seed42,float32/low_memory/cuda. Нативная coordinate-регрессия
невзвешенная; базис взвешенный. Суточные rawобъёмы030 точно совпадают029,
меняется часовая форма. Learnedprofile неотрицателен, нормирован с fallback029;
ночные1–4 и маршрут5 нулевые. Grid3control/quarter/halfcorrection; имена
control/half/new соответствуют0/,25/,5learned. Выбранnew поmeanW1/W2
0,884087957 противcontrol0,883516081/quarter0,883897851. Все3trials завершены.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884592576 | 0,883583338 | 0,885548438 | 0,899591884 | **0,888329059** | **0,883583338** |

Сравнение029: mean−0,000425722/worst+0,000918791. W1/W2 лучше, W3/W4 хуже.
По правилу разницыmean<0,001/лучшего минимума030 первый;029 сохраняет
наибольшее среднее текущей семёрки. В сравнении027 у030 вышеmean иworst;
030 заменяет027 как новый способ коррекции часовых ошибок. Архив027 сохранён
с прежнимSHA. Новая семёрка **030/029/025/024/023/026/005**.
Final030/029relativeL1=1,46%/10556часов,030/024=1,95%,030/025≈2,96%
(точные пары вfinal_difference.csv с общим знаменателем). Минимальная разница
любых двух финалистов1,31% (029/024); это не гарантия независимых ошибок.
030 total12806176,02912806177: разница1 послеокругления; rawdailyvolumes unchanged.
SHA03031b646027a6c2a71810ea89d22303cc4969e352f15b39cae43350c9aeef7d26e.
[CSV](../../../submissions/030_tabular_hourly_profile_errors.csv),
[происхождение](archive_030.json), [аудит](tabular_shape/independent_verification.json).

Seed73 фиксированнойnew:0,884265802/0,883484091/0,885671313/0,899760600,
mean0,888295452/worst0,883484091. Основнойseed42 не подменён. Разница финального
rawseed42/73=0,4904%; оба минимума выше029, улучшение небольшое.
Pilotrepeat точно совпал по каждому из6medianпрогнозов; новый fitseed42Oct31
вstudy точно совпал сpilot поbasis/координатам/предсказаниям/дельтам.

Первыйpilot8477033FAILED10с до обучения: pivot сохранил подпись оси колонокhour,
строгая проверка ожидала unnamed; метаданные нормализованы, значения и ключи
не менялись. Исходникдоисправления сохранён. Повтор8477034COMPLETED103с,
2CPU2GiB/1GPU/4мин,actual99,591с. Study8477035COMPLETED329с,
2CPU2GiB/1GPU/10мин,actual324,514с. PeakGPUallocated0,6164GiB/reserved0,8262GiB;
hoststudy1351724KiB≈1,289GiB. Всего442GPUseconds на семействе, включая отказ.
Нет нового download/upgrade/authentication/cloudclient/rawtransfer.

73input/codeSHAeachphase,11fits/66coordinate-models/16rawпроверены независимо.
Из frozenфактов иcausalteacher восстановлены hourly/dailyпары, reliablemask,
weights/means/covariance/eigenvalues/rank/signs/coordinates,50признаков,
календарьknown_at иfutureматрицы, fixed029volume переносы, сетка/zeros/half-up/
контроль/архив/seedразличия. Локально нейросеть не запускалась. В аудите только
нормализованы integerwidth/datetimeresolution локальногоPandas иCSV; значения
сравниваются строго. [Артефакты](tabular_shape), [исходники](source_tabular_shape).
Версии, checkpointSHA/лицензия/источники,OptunaSQLite/trials,targets/matrices/basis/
raw/final/seed/metrics/log/allocation/accounting/audit сохранены. Малый тест прошёл.
Команды изml: sbatch jobs/zhores_portfolio_tabular_shape.sbatch pilot; затем
sbatch --time=00:10:00 jobs/zhores_portfolio_tabular_shape.sbatch study
(свежийoutput длячистого повтора; бюджет существующегоstudy не сбрасывать).
Аудит: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/tabular_shape/verify_saved.py.
Лицензия: external/tabpfn/model_LICENSE.txt; checkpoint современный prior,
не заявляется доступным в историческом2025.

Общий аудит1213сохранённых окон/89внутреннихfinals,7unique14640-rowarchives,
17protectedfiles unchanged/контроль002 exact. Счётчики включаютповторныеtrials/seeds
и перекрывающиеся окна, не1213независимых испытаний. Обновлены28finalistmetrics/
280routes/21diversitypairs. CPU12051coreseconds=3,3475coreh,
GPU1995seconds=0,554167GPUh, осталось23,445833GPUh. Активных заданий нет.
**Goal активен:** все4окна030<0,90; закрытыйscoreнеизмерен, CSVнеотправлены.


## P66 — суточные пассажирские ошибки с TabPFN: отрицательный результат

Две фиксированные цели: (actualday−teacherday)/10000 и
(hour-L1-optimalvolume−teacherday)/10000. Optimalvolume — взвешенная медиана
actualhour/share по завершённым61-дневным прогнозам P39. Использованы
надёжные обычные профилиP65/последние224дня;51признак (P61daily30 +20teacher
hourshares +logsourcevolume). Median4estimators/seed42/float32/low_memory/cuda,
невзвешенный nativeprior. Futurebaseline030; новая сумма ограничена0,5–2 от
исходной, его часовые доли/нулевые дни/route5/nightzeros сохранены.
Pastteacher — исходныйP29/P21, finalbaseline — обновлённый030. Это документированное
различие модельных семейств, не прежние прогнозы точного030. Optimalpoint —
суррогат будущей потери при другой форме, не доказанный оптимум будущего риска.

Grid5 control/actual_half/full/optimal_half/full завершён; победилcontrol030
dev0,884087957. Лучший новыйactual_half dev0,880584500 противoptimal_half
0,880350760/actual_full0,868788397/optimal_full0,867736017.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,873165104 | 0,888003896 | 0,852804563 | 0,900004642 | **0,878494551** | **0,852804563** |

УлучшениеW2/W4 не компенсируетW1/W3; новая посылка031 не добавлена.
Seed73 фиксированногоactual_half:0,873279752/0,887277822/0,858418421/0,897737260,
mean0,879178314/worst0,858418421, тоже хуже030. Seed42 не подменён.
Selectedraw/CSV точно совпал030; retainedalternative/seed73 — исследовательские.
Семёрка030/029/025/024/023/026/005 сохранена.

Pilot8477037COMPLETED46с,actual41,8425с;study8477038COMPLETED89с,actual84,2592с.
Оба2CPU2GiB/1GPU, лимиты5мин/4мин. PeakGPUallocated0,6239GiB/
reserved0,8398GiB, hoststudy1351488KiB≈1,289GiB. Всего135GPU/270CPUcoreseconds,
без отказов/новыхdownload/envupgrades/rawtransfer. Built with PriorLabs-TabPFN;
копия лицензии и происхождениеcheckpoint сохранены ранее вexternal/tabpfn.

[Артефакты](tabular_volume), [аудит](tabular_volume/independent_verification.json).
71input/codeSHAeachphase,14fits/25raw проверены. Hourly/dailyпары точно совпали
с независимо восстановленнымиP65. Hour-L1целевые объёмы проверены независимым
условием выпуклой оптимальности: левый subgradient≤0≤правый, loss≤teacher/actual
day loss. Passenger_targets,51признак/матрицы/future, фактор/clip/сохранениеhourshares/
zeros/control/rounding/seed проверены. Pilot повторился точно; selectednewOct31
свежийfit совпал сpilot. Локально нейросеть не запускалась. В пересчёте floatсумм
обнаружено отличие порядка суммированияPandas/NumPy≤2,2e−11; такие derived
величины сравниваются с atol1e−8/rtol1e−12, опубликованное half-up и integer
метрика остаются точными. Малый проверяемый тест прошёл.
Source/Slurm/params/OptunaSQLite/trials/targets/reference/hourly/daily/matrices/
future/raw/final/seed/versions/checkpointSHA/log/allocation/accounting/audit сохранены.
Команда изml: sbatch jobs/zhores_portfolio_tabular_volume.sbatch pilot; затем
sbatch --time=00:04:00 jobs/zhores_portfolio_tabular_volume.sbatch study
(новыйoutput длячистого повтора, существующийbudget не сбрасывать).
Аудит: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/tabular_volume/verify_saved.py.

### Диагностика различия прежних и текущих прогнозов

Для2053надёжных прошлых route-days четырёх originApr30/Jun30/Jul31/Aug31
сопоставлены исходныйteacher и сохранённый030. Объёмы отличаются на4,06%relativeL1;
суммарное завышениеteacher4,10%,0300,88%. Направление пассажирской ошибки
противоположно у17,29%пар. Это подтверждает различиеforecasters; само по себе
не доказывает причинность проигрышаP66 и не измеряет скрытыйscore.
[Пары](tabular_volume/teacher_current_volume_pairs.csv),
[сводка](tabular_volume/teacher_current_diagnostic.json). Таблица не выбирала
policy/weights и не является независимым тестом. Следующий содержательный шаг:
учить ошибки на доступных завершённых прогнозах точно того же030, а при
отсутствии таких прогнозов сохранятьконтроль. Параметры/бюджет требуют
отдельной регистрации доcompute; новоймодели пока нет.

Общий аудит1235сохранённых окон/92внутреннихfinal,7готовыхархивов и17protected
SHA неизменны/контроль002exact. Счётчики — повторныеtrials/перекрывающиеся окна,
не1235независимых проверок. CPU12321coreseconds=3,4225coreh,
GPU2130seconds=0,591667GPUh, осталось23,408333GPUh; активных заданий нет.
Goal≥0,90everywindow unmet; скрытыйscore новыхCSVнеизмерен, ничего не отправлено.

## P67 — ошибки того же030: перенос между периодами остаётся слабым

Изменён толькоhistoricalteacher: исходныеP29заменены точными сохранёнными030.
Завершённые61-дневныеorigins: W1none→exactcontrol;W2/W3Apr30;W4Apr30/Jun30;
finalApr30/Jun30/Jul31/Aug31. Надёжная обычнаяhourlymaskP65/224daytail сохранена;
dailyконтекстP52 и школьные признаки относятсякorigin. Teacher030 hourshares/
volume входятв51features; target(actualday−030day)/10000. Fixedactual_half0.5,
median4estimators/seed42/nativeunweightedprior/clip0.5–2 сохраненыP66.
Нет новогоMay31teacher, временные окна/эталон/метрика не менялись.

Grid2control/matched выбралmatched: dev0,885650906 против0300,884087957.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884592576 | 0,886709235 | 0,879092383 | 0,896454999 | **0,886712298** | **0,879092383** |

Второе окно улучшилось0,883583338→0,886709235, последние два ухудшились.
Mean−0,001616760/worst−0,004490955 против030. Исправлениеteacher mismatch
не решило переносошибок между периодами. Seed73:0,884592576/0,886842385/
0,879264496/0,897231346,mean0,886982701/worst0,879264496; seedне подменён.
Finalrawотличаетсяот030на1,63034%relativeL1, publishedtotal12 597 907
против12 806 176. Отдельная конкурснаяпосылка031 не добавлена; семёрка
030/029/025/024/023/026/005 сохранена. Закрытыйscore не измерен.

Pilot8477039COMPLETED21с/actual16,2008с,study8477040COMPLETED38с/actual34,4307с;
оба2CPU2GiB/1GPU, лимиты1мин/3мин. Всего59GPU/118CPUcoreseconds.
Pilot2053rows/4origins; studytrain549/549/1029/2053rows. Repeatedmedianpilotexact,
свежийfinalstudyseed42повторилpilotexact. PilotGPU0,4211GiB/reserved0,5508GiB;
studyhost1348668KiB≈1,2862GiB. P66/P67всего7trials/194GPUseconds,
исходныйfamilywallstarted_at1790489139.482144 и1800s сохранён вOptuna;
старые5trials/135GPUне сброшены. Современныйcheckpointresearchonly; Built with
PriorLabs-TabPFN, прежняя лицензия/хеш/isolatedenv сохранены. Rawне переносились.
Логи содержат предупреждение nativepreprocessor о новыхкатегориях, finiteoutputs
и teacher/futureматрицы проверены; это не подтверждение переноса на зимнийпериод.

[Артефакты](matched_volume),
[аудит](matched_volume/independent_verification.json).
72input/codeSHAeachphase/9fits/14rawchecked. Независимо сопоставлены teacher030,
actualфакты/каузальныеdailyконтекстыP65, завершённость всех61days, shares/volume/
passengertargets/X51/y/future/clip/scale/zeros/gatecontrol/mix0.5/seed.
W1без завершённыхteacherсохранёнточно030; historytargets≤cutoff, futureбезфактов.
GPUinferenceлокальноне запускалась. Точный half-up/метрика всехsavedforecasts
проверены общим оценщиком:1247сохранённых окон/94внутреннихfinal,7архивов,
17protectedSHA/контроль002unchanged. Это повторные и перекрывающиесяпроверки,
не1247независимыхтеста. Всеparams/code/teacherSHA/matrices/Optuna/log/allocation/
accounting/audit сохранены. CPU12439coreseconds=3,45528coreh,
GPU2189seconds=0,608056GPUh, осталось23,391944GPUh; активныхзаданий нет.

Воспроизведениеизml: sbatch --time=00:01:00 jobs/zhores_portfolio_matched_volume.sbatch pilot;
sbatch --time=00:03:00 jobs/zhores_portfolio_matched_volume.sbatch study.
После исходного familydeadline его нельзя сбрасывать: чистый повтор требует
отдельного обоснованного бюджета, существующиеcompletedcachedresults доступны.
Проверкаsaved: PYTHONPATH=. python artifacts/portfolio_20260926/continuation/matched_volume/verify_saved.py.
Малый completed-horizon/teacheralignment/featurepoison тестпрошёл.

P67 статус: проверено, отклонено дляархива. Goal≥0,90everywindow unmet, активен.

## P68 — байесовский риск текущих кандидатов: близкая смесь, лидер сохранён

Неизменённый P17 infer_weights применён к точным030/029/025/024 вместо прежних
ridge/Chronos/movement/regime. Старые названия — только внутренниеinterfacealiases,
явное соответствие сохранено. Dirichlet(2)165точексимплекса, Gibbs temperature0,15,
25%networkshrinkage/min7days/route×season×daygroup прежние. Loss — half-up hourlyL1,
агрегированнаявroute-dayblocks с усреднением повторныхorigin. Не165независимых
trials; Bayesian posteriorдетерминирован, MonteCarlo/подборseed отсутствуют.
Завершённые exactteachersApr30/Jun30/Jul31/Aug31, full61dayend≤cutoff.
W1none→exact030;W2/W3Apr30;W4Apr30/Jun30;final4origin. НетMay31teacher.
SeasonDec-Feb/Mar-May/Jun-Aug/Sep-Nov; fallbackпо типудня в другие сезоны при
<7same-season-route-group days, будущаяподдержкасезона записана отдельно.

Grid2posterior/control, meanW1/W2: mixture0,883791495 противcontrol0,884087957;
выбран030. Фиксированная Bayesianальтернатива проверена наall4+final:

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884592576 | 0,882990414 | 0,886288574 | 0,901056528 | **0,888732023** | **0,882990414** |

W3/W4 улучшились, W2 ухудшилось. Mean+0,000402964/worst−0,000592924 против030:
разница среднего<0,001, правило предпочитает030с лучшим минимумом.
Meanпочти029 (−0,000022758), worst+0,000325867; отличие finalот0290,92788%,
от0240,88092%, от0301,29485%. Близкая смесь не расширяет семёрку существенно,
новаяпосылка031 не добавлена. Finalpublishedtotal12 750 053, против03012 806 176.
Текущаясемёрка030/029/025/024/023/026/005, основныеCSV иservicebundle неизменны.

### Проверка прогнозируемого качества

Pastconditionalrisk/weight-SD рассчитаны безfutureactuals. Послефактум proxy
сопоставленс scoreокна; proxy агрегированс исходными030futurevolumeweights:

| Окно | Предсказанный proxy | Фактический score | Разница proxy−fact |
| --- | ---: | ---: | ---: |
| W1 | нет завершённыхrefs | 0,88459 | — |
| W2 | 0,87773 | 0,88299 | −0,00526 |
| W3 | 0,88527 | 0,88629 | −0,00102 |
| W4 | 0,88847 | 0,90106 | −0,01258 |

Proxy ошибается до1,26percentagepoints на этих reusedокнах; нет доказательства
надёжной калибровки закрытогоscore. SD характеризует только неопределённость
весов на конечномсимплексе, не futuredata/regimechange variance. Долябудущих
route-days с≥7same-season refs: W2 85,25%,W3 42,62%,W4 0%;final49,18%.
В finalноябрьподдержан прошлымосеннимриском, декабрь не имеет зимнихфактов
для этихтекущихforecasters. Это перенос из других сезонов, не зимняяпроверка.
Оценки качества полезныдля диагностики, уверенность из них не завышается.

Slurm8477041COMPLETED17с/actual13,8330с,ais-cpu1CPU1GiB/2мин, peak232236KiB≈0,2215GiB.
НовыхGPU/зависимостей/download/rawtransfer нет. СтарыеP17 23с/92coreseconds учтены:
семействоP17/P68всего40allocationseconds в1800secondsсогласованном computeceiling;
добавлены2trials к прежнейфиксированнойpolicy, старые артефакты нетронуты.
[Артефакты](current_bayes),
[аудит](current_bayes/independent_verification.json),
[калибровка](current_bayes/quality_calibration.csv).
31input/codeSHA/4calibrations×108posteriorgroups/14raw проверены. Независимо
восстановлены исходныеforecastpairs/frozenfacts, prior/grid165/Gibbs/algebra/
leave-route-outpool/repeatweighting/conditionalpastloss/weightSD/seasonfallback/
futureweights/mix/zeros/control. Источники всехtargets≤cutoff, calendarknownbeforeorigin.
Lightlocalaudit сверяет saveddata/finiteposterioralgebra; тяжёлых моделей локальнонет.
Исправления аудита касались только remoteabsolute path иdatetime[s]/[ns] сравнения;
значения/code/teacher/model/window неизменны. Smallriskpreference/normalization/
futurelossguard проверка прошла. Всеinputs/source/params/Optuna/weights/pastpanels/
quality/log/allocation/resource/audit сохранены.

Общийоценщик проверил1259сохранённых окон/96внутреннихfinal,7архивов/17protectedSHA,
контроль002exact. Повторные и перекрывающиеся окна не независимы.
CPU12456coreseconds=3,46coreh,GPU2189seconds=0,608056GPUh,
осталось23,391944GPUh; активныхзаданий нет. Goal≥0,90everywindow unmet, активен.
Командаизml: sbatch jobs/zhores_portfolio_current_bayes.sbatch.
Аудит: OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python artifacts/portfolio_20260926/continuation/current_bayes/verify_saved.py.
При повторе учитывать17secondsсуществующегоP68/40wholefamily; подбюджет не сбрасывать.
P68 статус: проверено, отклонено для расширенияархива; qualitycalibration сохранена.

## P69 — прямые суточные counts с TabPFN: отрицательный результат

Прямойnormalday target(actual/10000) и target(actual/reference), вместоsigned
ошибкиготовогоteacher. P25origincontext/reference56days/ratio7/14/28/ERA5 reused;
месячныеoriginsJan31..cutoff−1, фактическиеtargets1–61day≤outercutoff. Частичные
завершённыеtraininghorizons разрешеныдляdirectsupervisedtargets, outer61dayрежим
неизменен. Contextreference инедавние отношения вычисленытолькоизoriginpast.
Обучение включает февраль; январьиспользован как начальныйконтекст. Это не
отдельнаяпроверка будущегоноября–декабря. Normalhistory excludesknownJul/Apr17/
Aug7/autumn disrupted days; отсутствующиефакты/импутация не менялись.

21features=P25minusdiscretequarter15 +6schoolcalendarpublicationgatedP61;
annualsin/cos остаются, route/effectiveweekday categorical[0,1]. Nativeunweighted
median4estimators/seed42,float32/cuda/low_memory/memorysaving/n_jobs2.
Normalvolume nonnegative, безupperclip; existingoperationpriorsjuly7=.45/
july50=1.25/Aug7=.8 plusouterhistorycalibratedfactor, verifiedJulrestoreAug11.
030hourshares fixed, егоzero-days/route5/nightsневосстанавливаются.

Grid5control/count_half/full/ratio_half/full завершён;control030выбран:
dev0,884087957, bestnewcount_half0,883427444; count_full0,876232614,
ratio_half0,876426897, ratio_full0,860949406. Лучший новыйвариантall4:

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884290345 | 0,882564543 | 0,852961983 | 0,886213251 | **0,876507531** | **0,852961983** |

Всеокна слабее030, особенноAugust–September: bias−1 260 597 (≈−11,0%volume).
Mean−0,011821528/worst−0,030621355 против030. Seed73фиксированногоcount_half:
0,879437556/0,880858785/0,847157647/0,888557295,mean0,874002821/
worst0,847157647, безподменыseed42. Нативныйprior не устранил seasonaltransfer
припрямомпрогнозе; расширятьмелкуюсетку этогоdirectпредставлениянеобоснованно.
Finalrawdifferencevs0304,81143%, publishedtotal12 288 818 против12 806 176.
НовыйCSV031 не добавлен, семёрка030/029/025/024/023/026/005 неизменна.

Pilot8477042COMPLETED35с/actual30,2983с,study8477043COMPLETED60с/actual55,4750с;
оба2CPU2GiB/1GPU, лимиты2мин/3мин. Pilot4452rows/9monthlyorigins;
studyrows1352/2450/2920/3418/4452. Repeatedmedianprediction exact; свежий
selectednewfinalseed42 совпалсpilot. Всего95GPU/190CPUcoreseconds.
PilotGPU0,3400GiB/reserved0,4336GiB, host1297220KiB≈1,2371GiB;
studyhost1356928KiB≈1,2941GiB. СтарыеTabPFN13trials/711GPU учтены:
теперь18trials/806GPU, новыйbudgetне сбрасывался. ПрежниеP25/P37direct8+8trials
не повторялись. GPUdownload/envupgrade/rawtransfer отсутствуют. Built with
PriorLabs-TabPFN, прежнийcheckpointSHA/license сохранены. Nativepreprocessor
предупреждал о новыхкатегориях; конечныеpredictions/матрицы проверены.

[Артефакты](tabular_direct),
[аудит](tabular_direct/independent_verification.json).
27input/codeSHAeachphase/14fits/25raw проверены. Независимо восстановлены
normalhistorymasks/monthlyorigins/origin-onlyreference/ratios/weatherctx/calendar/
schoolpublicationgate/21X/targets/nativeoutput/operations/fixedshape/zeros/mix.
Всеtrainingtargets≤outercutoff, futureбезboardings; calendar/featurespoisoncheck прошёл.
Lightlocalaudit данных/арифметики; нейросетьлокально не запускалась.
Интерпретация derivedcountnormal/operatedfields: nativefloat32count arithmetic,
короткиеCSVdecimalпрочитаны обратно именнокакfloat32 иточно сверены сnpzoutput.
Ratiofieldsfloat64. Это формат внутреннихчисел, не изменениеfrozenrounding/
metric/rawforecasts; finalrawfloat64/half-up/intscoreпровереныexactобщимоценщиком.
Auditтакже нормализуетимена/orderколонок иdatetimewidth междуPandasверсиями;
ключи/facts/featureorder вматрицах проверены, prodcode/targets не менялись.
Source/hash/runversions/Slurm/Optuna/past/future/matrix/operation/raw/seed/resource/
audit сохранены. Воспроизведениеизml:
sbatch jobs/zhores_portfolio_tabular_direct.sbatch pilot;
sbatch --time=00:03:00 jobs/zhores_portfolio_tabular_direct.sbatch study.
Проверкаsaved: OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python artifacts/portfolio_20260926/continuation/tabular_direct/verify_saved.py.
Существующие95seconds/5trialsP69 приповтореучитывать, не сбрасыватьphasebudget.

Общийоценщик1281savedwindows/99internalfinal,7архивов/17protectedSHA unchanged,
control002exact. Повторныеперекрывающиесяокна не независимы. CPU12646coreseconds
=3,51278coreh,GPU2284seconds=0,634444GPUh,осталось23,365556GPUh; livejobsнет.
Goal≥0,90eachwindow unmet, активен. P69 статус: отклонено дляархива.


## P70: робастная годовая поправка не улучшила набор

Прошлые остатки Gaussian-модели имеют тяжёлые хвосты: около2% наблюдений
дают32–43% суммы квадратов ошибок. Проверка Student-t с прежними признаками,
данными и ограничениями завершена. Половина поправки дала0,88313/0,88424/
0,88436/0,90015, среднее0,88797, минимум0,88313. Optuna выбрала прежний030;
его среднее и минимум выше. НовыйCSV не добавлен, goal≥0,90eachwindow активен.

Это условный MAP с фиксированными prior/scale; полное байесовское распределение
и уверенность в закрытом score не рассчитывались. Дневной объём изменён,
распределение029 по маршрутам и030 по часам сохранено. Реальные наблюдения
не удалялись. [Артефакты](robust_annual), [независимая проверка](robust_annual/independent_verification.json).
44SHA для каждой фазы/7fits/22raw, causalinputs/MAPstationarity/attenuation/
networkratio/shape/zeros проверены. Общий оценщик1295windows/101internalfinals,
17protectedfiles/control002 неизменны. Повторные окна зависимы.

Затрачено11CPUseconds/0GPU. Один launcher остановился до расчётов из-за
старогоBash; исправлена передача пустого параметра. ВсегоCPU12657coreseconds
=3,515833coreh, GPU2284seconds=0,634444GPUh, осталось23,365556GPUh.
Следующая обоснованная проверка: в030 шесть компонент сохраняют65–71% прошлой
вариации часовых ошибок; для90% данные требуют12 на каждой точке.
Проверить дополнительные компоненты с сохранением объёмов и прежних окон.
Это ещё идея, не измеренное улучшение.


## P71: 031 заменил030 в рекомендуемой семёрке

Данные каждой точки потребовали12компонент для90% прошлой вариации часовых
ошибок. Первые6точно переиспользованы,6новых обучены. Retained91,15–91,92%,
50признаков/nativeTabPFNmedian4/seed42, raw029route-dayvolumes сохранены.
Выбрана полнаяноваяпоправка (.5learnedhourshape+.5raw029) изOptuna3trials.

| W1 | W2 | W3 | W4 | Среднее | Минимум |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0,884835229 | 0,883921611 | 0,885928782 | 0,899586386 | **0,888568002** | **0,883921611** |

К030mean+0,000238943/min+0,000338273 — небольшойприрост, всеокна<0,90.
Seed73mean0,888534752/min0,883870378, seedнеподменялся. 029 остаётся лучшимпо
mean0,888754781; gap<0,001, поэтому031первый поminimum. Архив030сохранён.
[CSV031](../../../submissions/031_adaptive_hourly_profile_errors.csv),
[артефакты](adaptive_shape), [аудит](adaptive_shape/independent_verification.json).
Finaltotal12 806 181, SHA28f6fb76ed137f4fda72e967ee7e3b35df6835eeb6fc5dc1e1088732c5a3c077;
с029плюс4roundingcounts. Rawdiffvs0300,48436%,031/029published1,48934%;
остальныепарыcurrent7 ≥1,30920%. 26SHAphase+50parents/11datasets/66newcoords/
16recipe raws checked, parentP65causal audit повторён. Pilot/freshstudy exact,
14 640unique keys/halfup/zeros/17protectedfiles/control002 verified. Общаяпроверка
1309savedwindows/103internalfinals; повторныеперекрытия не независимы.

Pilot8477052COMPLETED103GPU,study8477053COMPLETED320GPU, total423GPU/846CPU.
OldTabPFNcosts retained:21trials/1229GPU now. ProjectCPU13503coreseconds
=3,750833coreh,GPU2707seconds=0,751944GPUh,remaining23,248056GPUh,livejobsnone.

### Диагностика ограничения общего объёма

Толькоposthocизвестныеlocalfacts, неcandidate и неselection. Perfectactualroute-day
с031hourshares даёт0,91748/0,91655/0,91968/0,92162. Perfectobservedhourprofile
сcurrentroute-dayvolume —0,92191/0,92143/0,92040/0,93784. L1-optimalобщийобъём
дня (weightedmedian) сfixedroute/hourallocation plusworstcase halfuprounding
строго ограничивает score:0,90049/0,89618/0,89737/0,90684. Значит, network-only
поправка не может достичь90gate наW2/W3. [Числа](adaptive_shape/component_oracles_031.csv),
[метод и SHA](adaptive_shape/component_oracles_metadata_031.json).
Следующаяобоснованная гипотеза — автоматическое байесовское ослабление разных
route/calendar/annualfeatures вместооднойсилы регуляризации длявсех97признаков.
ARDnativeAPI проверен; модель ещё не запускалась. Текущийgoalактивен.


## P72 — автоматическая релевантность

ARD выбрала половину новой поправки: 0,88238/0,88758/0,88251/0,89965, mean0,88803/min0,88238.
Общая оценка и минимум ниже031: набор сохранён. 3trials/12CPU seconds, GPU0.
Проверены причинность, posterior normal equations, fixed attenuation и точный повтор.
Исправлена синхронизация таблиц ошибок финалистов: теперь их создаёт общий оценщик.
[Журнал](../../../EXPERIMENTS.md), [аудит](ard_annual/independent_verification.json).


## Пауза после P72

Пользователь сообщил030=0,88543/031=0,88537 на платформе, независимо не проверено.
Локальный gate0,90eachwindow не достигнут; P73 не запущен. Результаты сохранены.
Активных собственных заданий нет; новые эксперименты остановлены.
