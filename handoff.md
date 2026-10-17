# Передача XAUUSD / 222 инструмента / learned reward teacher

**Исследовательский снимок: 16 октября 2026, UTC. Передача опубликована 17 октября. Язык общения с пользователем — русский.**

Этот документ передаёт фактическую работу следующему агенту. Он отделяет завершённые проверки от намерений и неисправленного кода. [previous.md](previous.md) — хронологическая история старых направлений; здесь — подробный технический контракт и точка продолжения. Настройка авторизации/доступа к Colab намеренно опущена: следующий агент уже сделал её. Никакие credentials в комплект не входят.

## 0. Прочитать прежде любого запуска

1. **Старый GPU-run v3 остановлен по прямому указанию пользователя. Не возобновлять его, watchdog или старые supervisor автоматически.** В момент остановки подтверждены 0 assignments и расход 0. Последний исторический баланс 64.44 compute units; это не обещание актуального баланса другого агента.
2. Последняя рабочая ветка исследования — `experiments/sequential_teacher_v4/`. Это **прототип исправленной механики и дешёвые контрольные исследования**, не готовая прибыльная система и не завершённое обучение на 222 рынках.
3. На момент передачи **16 тестов проходят**, ученик с денежной наградой проходит 4 синтетические задачи × 3 seed. Но outer REINFORCE teacher не проходит перенос; отдельный exact meta-gradient проходит direction/wait, проваливает cross_market. Оба допуска к реальному pilot закрыты. Это ожидаемый результат проверки, а не ошибка установки.
4. v2 дал +$231.67, но все сделки LONG; always-LONG +$255.63. Аудит не подтвердил полезность тайминга. **Нельзя говорить пользователю, что прибыль доказала хорошего учителя.** v3 обе стороны использует, но убыточен.
5. Не путать три разных сущности: hindsight-teacher/reward model; causal student/actor; его value head. Teacher видит будущее и **никогда не является доступным live-input**.
6. Старые `RUNNING.md`, `live_status.json`, PID, VM names и supervisor/status-файлы — исторические снимки. Текущая остановка и этот документ имеют приоритет. Никакого фонового мониторинга или обучения, обещанного от имени прошлого агента, сейчас нет.
7. Исходники и результаты в этом коммите сохранены без изменения исследовательского алгоритма. SHA256 привязывают численные результаты к коду. Новые правки требуют повторить затронутые проверки, а не просто обновить manifest вручную.
8. Следующая задача — улучшить/проверить обучение **полезного teacher** на контролируемой межрыночной задаче, затем малый реальный pilot. Не скачивать десятки GB и не выделять GPU только ради чтения этого handoff.

## 1. Что хочет пользователь и что уже согласовано

### Конечная задача

Использовать совместную прошлую информацию **всех 222 инструментов** для решений по XAUUSD, первоначально хорошие входы LONG/SHORT, затем отдельно научить закрытию. Важно найти информацию сверх собственного прошлого золота: взаимодействия нескольких рынков, а не только попарную корреляцию. Желательна внутридневная торговля, умеренная просадка, достаточно регулярный результат хотя бы по неделям.

Teacher должен оценивать каждую точку входа двумя числами **в (0, 1)**: LONG и SHORT. Изначальная идея — около 120 минут прошлого и 120 минут будущего. Student видит только причинное прошлое. Teacher нужен не как произвольная вручную выведенная метка, а как обученная функция награды, которая помогает новым ученикам принимать полезные решения.

Teacher обучается по качеству обучения student на других данных. Это двухуровневая оптимизация: внутреннее обучение student по teacher reward; внешняя оценка результата student по реальному торговому objective. Требование «без меток» означает отсутствие списка единственно правильных входов, а не отсутствие внешнего критерия прибыли/риска.

### Важные предпочтения

- Transformer и для teacher, и для student. Малые контролируемые v4 Transformer — сознательный диагностический этап; исходная цель больших глубоких моделей не отменена, но не заменяет исправность метода.
- Не подавать абсолютный уровень цены в student. Относительная/векторная геометрия, причинные преобразования и availability masks допустимы.
- Не выдавать произвольные ATR/forward-score коэффициенты за математически универсальное качество входа. Прибыль, риск, время и scale должны быть явно определены; окончательные risk preferences пока не выбраны.
- Не навязывать искусственный баланс LONG/SHORT ради красивой картинки. Нужно отделить настоящий direction edge от bias и частоты входов.
- Пользователь хотел поощрять много полезных сделок, поскольку видел диверсификацию риска. **Количество само по себе не доказывает независимость рисков** и не должно автоматически вознаграждаться за спам/дробление.
- Проводить этапы с понятным отчётом, не проскочить все ступени curriculum без оценки. Не объявлять любой неуспех «в данных 99% нет информации».
- GPU разрешена, если реально быстрее. Пользователь платит за время выделения, не за процент VRAM. Не надо занимать всю память ради показателя: измерять throughput/end-to-end runtime, оставлять место под activations/optimizer, учитывать OOM. 96 GB, 56 GB и названия GPU в разговоре не являются проверкой объёма текущей карты.
- Нужны checkpoints и сохранение лучших кандидатов с кодом, schema, transforms и provenance. Сохранять failure тоже — не перетирать хороший архив новым неудачным результатом.

### Последние решения пользователя

Последовательность: сомнения в пользе v2 → аудит → обсуждение остановки v3 → «В пизду. Оффай. Перерабатываем» → подробный разбор → план исправления → «Реализуй план». Затем текущая задача: **передать работу, всё нужное/актуальное в GitHub без больших файлов, previous.md и максимально подробный handoff.md**. Передача не является новым разрешением возобновить старый дорогой run.

## 2. Где что лежит и чему доверять

### Приоритет чтения

1. Этот `handoff.md` и [план v4](docs/sequential_teacher_v4_plan.md).
2. [Аудит v2/v3](reports/teacher_training_audit_20261016/REPORT_RU.md).
3. [Текущий отчёт v4](reports/sequential_teacher_v4_20261016/REPORT_RU.md), его `gate_manifest.json`, `launch_decision.json`, `meta_gradient_suite.json`.
4. [README v4](experiments/sequential_teacher_v4/README_RU.md) и исходники.
5. При работе со старым teacher — [инструкция полного архива v2](exports/xauusd_teacher_v2_cycle80/README_RU.md) и поздний аудит. Старая инструкция описывает интерфейс, а не сертификат полезности.
6. Общая история и ссылки на ранние отчёты — [previous.md](previous.md).

### Каталоги

| Путь | Назначение / статус |
|---|---|
| `experiments/sequential_teacher_v4/` | Текущий исправленный исследовательский harness |
| `reports/sequential_teacher_v4_20261016/` | Текущие CPU результаты, тесты, gates, маленькие checkpoints |
| `docs/sequential_teacher_v4_plan.md` | 8 пунктов плана с реальным статусом выполнения |
| `experiments/meta_reward_transformer_v2_foreground/` | Код завершённого v2, от которого идут реальные цифры v2 |
| `reports/meta_reward_transformer_v2_foreground_20261015/remote/results/` | Первоисточники v2: config, curves, fresh students, тестовые решения |
| `experiments/meta_reward_transformer_v3/` | Старый остановленный v3; features/model/schema используются для восстановления |
| `reports/meta_reward_transformer_v3_20261016/` | Последний аудит/снимок/остановка v3 |
| `reports/teacher_training_audit_20261016/` | Кодовые, математические и execution-дефекты старого метода |
| `reports/v2_entry_timing_audit_20261016/` | Точное воспроизведение сделок v2, 8000 рандомизаций |
| `exports/xauusd_teacher_v2_cycle80/` | Текстовая часть полного Drive-пакета; большие веса и fixture брать из архива |
| `experiments/intermarket_lightgbm/` | Старое межрыночное дерево, causal features и metadata |
| `experiments/spectral_year/` | Воспроизведение causal STFT/wavelet года |
| `reports/causal_spectral_year_2021_20261012/` | Schema/provenance/нормы/малый context; HDF5 исключены |
| `docs/handoff_20261016/` | Manifest перенесённых файлов и реестр исключённых локальных артефактов |

Изначальный remote `main` до передачи: commit `4128bac87029d1be5b73492e1aaa0aeff24ade82`, снимок исследований 9 октября. Он содержал ранние angular/MBO исследования и M1 parquet. Новая передача добавляет последующие исследования, а не удаляет ту историю. Branch передачи: `handoff/2026-10-16-teacher-v4`. Актуальный commit смотреть в Git, не в динамическом runtime status старых файлов.

## 3. Данные: источники, размеры, время, причинность

### 3.1 Исходные 222 инструмента

Папка пользователя: <https://drive.google.com/drive/folders/1BHeZ-romQ4oIZ970lxZCEYDKpyZ8QFMd>.

Файлы и Drive IDs: `experiments/meta_reward_transformer_v3/drive_manifest.json`. Там **222 файла**, порядок модели в `remote/prepared.json` / `exports/.../prepared.json`; не строить порядок по произвольной выдаче API. Проверенный ранним аудитом объём: 384 509 621 M1-свеча, ~5.42 GB Parquet. Пользователь называл примерно 10 GB — в отчётах использовать измеренный размер и не путать с распакованным/обогащённым объёмом.

Источник — Spotware demo/cTrader OpenAPI. `timestamp` — UTC начало M1; свеча с timestamp t полностью доступна после t+1 минуты. Volume — tick volume, не достоверный биржевой объём. Истории/сессии инструментов сильно различаются. XAUUSD: `XAUUSD_41.parquet`, 3 505 124 строки, первая 2016-09-04 22:00 UTC, последняя 2026-09-04 13:08 UTC (правая граница запроса 13:09).

Во временной среде прошлого агента были только небольшие локальные исходные выборки:

- `reports/intermarket_lightgbm_20261015/samples/XAUUSD_41.parquet` (~55 MB);
- `EURUSD_1.parquet` (~56 MB);
- `APPLE_21508.parquet` (~8 MB).

**Они не включены в новую передачу.** Не считать три файла заменой 222. Скачивать нужную исходную поставку по manifest/папке; старый `data/M1/xauusd_m1.parquet` в remote относится к ранним исследованиям, идентичность с XAUUSD_41 не заявляется.

### 3.2 Признаки старого teacher v2/v3

`experiments/meta_reward_transformer_v3/features.py` задаёт **48 причинных признаков на инструмент**. Группы: returns 1/5/15/30/60/240/1440; vols 15/60/240; ranges/positions 15/60/240; body/wicks; tick volume; acceleration; lagged returns/volatility/position; quote age/availability/gaps; angular OHLC; Bollinger z60; tick-weighted deviation60; efficiency120; return120; vol ratio60/1440. **Точные формулы и порядок — `NAMES` в коде**, не этот пересказ.

Хранение: clip ±50, float16; на входе asinh и finite masks → **96 каналов** на инструмент. Нет отдельного обученного глобального StandardScaler. Воспроизводимость требует features.py, data.py, порядка symbols/names, временной сетки, warmup и масок.

Actor input в v2/v3: `past[B, 222, 120, 96]`, только завершённые минуты `[t−120min, t)`; `clock[B, 8]` — sin/cos 1/2/3 гармоник времени суток и дня недели. Teacher дополнительно `future[B, 120, 8]`: XAUUSD t..t+119, OHLC increments, cumulative H/L/C относительно entry Open, availability; нормировка из прошлого. Прямого включения будущего в student input по чтению кода не нашли; полного независимого сертификата отсутствия утечек нет.

Медианная доля свежих наблюдений по инструментам относительно gold anchors ~24.5%. Это не автоматически баг: разные биржевые часы/даты начала истории. Но требует проверки stale features и абляций.

Полный prepared cache имел ~74.7 GB; рабочий training cache занимал ~62–64 GB VRAM. Он жил в удалённой GPU VM и после остановки **не сохранён здесь целиком**. Его наличие в памяти не означало полный проход моделей по 10 годам.

### 3.3 Не смешивать две временные конвенции

В **ранних спектральных экспериментах** по просьбе пользователя свечи склеены: период 8192 = 8192 реально наблюдаемых свечей, выходные удалены. `available_timestamp=timestamp+1min` при интерпретации timestamp как начала свечи. Односторонние causal Morlet не равны центрированной CWT первого визуального исследования.

В **межрыночной панели и v2/v3/v4 execution** нужна синхронная календарная UTC-сетка с availability/age. Нельзя склеить каждому рынку выходные независимо и потом сопоставлять строки как одно время. Горизонт 120 календарных минут не равен 120 склеенным барам. Раннее пользовательское пожелание склеить FFT-паузы не переносить автоматически на торговый симулятор.

### 3.4 Спектральный год и воспроизведение

2021: 354 340 свечей, 2356 компонент, STFT128/512/8192 и 75+75 причинных wavelet scales. HDF5 22.865 GB не в Git. Включены маленькие `normalization.npz` и `input_context.npz` (502 499 строк Aug2020–Dec2021), schema, dataset/manifest, provenance, исходники; этого достаточно для повторной генерации представления по протоколу. Исходный SHA256 в `input_provenance.json` — `74534bc17ccda0a0b849997ca5bee5d1f4bb35b5a31f6075c24517f751636410`; context SHA — `5be7e88da185bcc758dcfd31948718c6a884eaed13af05dc554c553c6b7d661c`.

Не генерировать заново десятки GB без необходимости: это предыдущая ветка исследования, не вход текущих v2/v3 учителей. Они используют 48 causal features, а не все спектральные коэффициенты.

## 4. Что реально делали v2 и v3

### 4.1 V2 foreground: архитектура и обучение

Основной источник: `experiments/meta_reward_transformer_v2_foreground/`, `.../remote/results/run_config.json`, `dataset_protocol.json`, `evaluation_windows.json`.

- MarketTransformer: temporal stack6 + cross-market stack6, width192, heads6, dropout0.25, patch10. Teacher 5 588 740 параметров.
- Teacher выдаёт два Beta distribution; параметры `1+softplus`. Scores в (0, 1), но **не вероятности успеха**. Два outputs LONG/SHORT не обязаны суммироваться к 1.
- Три постоянных student, обучавшихся на последовательных версиях teacher.
- Один outer cycle: 32 inner updates × batch256 = 8192 отдельных случайных точек. Student lr3e-4, teacher lr1e-4; всего 96 outer updates.
- Inner student был **contextual bandit**, не последовательный RL: WAIT/LONG/SHORT на отдельной точке. Награда WAIT=0, OPEN=2*score−1. Value head предсказывал эту немедленную награду. Не было TD/GAE и корректной стоимости занятости позиции.
- Policy advantage нормировался с floor0.1; adaptive entropy начиналась с alpha0.05, target0.5*log3, множитель exp(0.01*error), clip alpha0.01..2. Не трактовать рост/падение entropy как самостоятельное доказательство обучения.
- Outer teacher REINFORCE: один scalar `U_after−U_before` за 32 inner updates, умноженный на log-prob обеих teacher scores всех точек, включая неиспользованные.
- `U=(sum(net_trade_pnl)−sqrt(sum(net_trade_pnl²)))/calendar_days`. Позиция одна унция, постоянные условные roundtrip издержки $0.25. Это **не процент доходности счёта**, не Sharpe и не MTM drawdown.
- Во внутреннем meta-loop teacher.sample + dropout; у fresh student teacher.mean + eval. Этот mismatch не был изолированным экспериментом.

Разбиение: train2016–2022, meta2023, dev28 calendar days c 2024-03-04, test84 calendar days с 2025-01-06 до 2025-03-31. Фактически 786432 combined training examples за 96cycles (~35.482% TRAIN), по 262144 на каждого из 3student (~11.83%). Свежий student seed888 —768updates×256=196608 (~8.87% TRAIN). Это не 96 полных эпох по 10 годам.

Selected teacher **cycle80**, но процедура отбора flawed: сравнивали dev накопившего знания student, сохраняли текущего teacher, включая teacher step, ещё не повлиявший на проверяемого student. Наименование `best_teacher` историческое, не утверждение о лучшем независимо проверенном teacher.

### 4.2 Кто закрывал сделки

У student НЕ было обучаемого действия CLOSE. В evaluation отдельный executor после OPEN блокировал новые входы и закрывал автоматически: план `min(t+120min, 23:59 UTC)`; затем `searchsorted` первой доступной котировки. Student выбирал WAIT/сторону/момент, **выход задавал симулятор**. Поэтому объяснение «ученик сам научился входить и выходить» было бы неверным.

При отсутствии плановой котировки `searchsorted` переносил выход на следующий доступный бар, иногда через выходные. Это существенный баг; подробнее ниже. По 23:59 возникали сделки нулевой длительности с одними издержками.

### 4.3 V2 результаты и сохранность

| Frozen reward для свежего student | Test net $/унцию | Сделки | Комментарий |
|---|---:|---:|---|
| learned v2 teacher | +231.67 | 720 | 720 LONG, 0 SHORT |
| direct_money | −37.67 | 750 | Не доказывает превосходство teacher без сильных controls |
| untrained_teacher | −37.10 | 777 | Один seed |
| always-LONG, posthoc control | +255.63 | 778 | Лучше learned по сумме на этом же executor |

Старый отчёт содержит drawdown98.56 по дневной агрегации. Эта агрегация относила PnL к **дню входа**, а не выхода; это не внутрисделочная MTM просадка. Выводы о риске ограничены. Bootstrap learned-minus-money также включал 0. Все эти Jan–Mar2025 данные уже изучены.

Полный backup:

- Folder: <https://drive.google.com/drive/folders/1J5z1MDoS08ueTiGcg06rPeqWNrCtWio5>
- ZIP: <https://drive.google.com/file/d/17V1lXJW0JItNDE01TGH8VsC0fCC9ucTu/view>
- Имя: `XAUUSD_teacher_v2_best_cycle80_complete.zip`
- Размер **67 744 620 bytes**.
- ZIP SHA256 **`1c45e2a05fb6684594139407f3255a0b9533dc124a152c005266171d2b1d9e82`**.
- `best_teacher.pt` SHA256 **`5371038e6b8a4dc8613b1a3ce0898d45df4d6254c4ad249f83b5d3ede22b9ed4`**.

Drive metadata size проверен после upload; локальный archive/manifest/self-test проверены. Не утверждать, что Drive вернул SHA256: этот hash вычислен локально. Архив содержит weights+optimizer/config/cycle, loader, оригинальные model/features/data/prepare/train/objective, schema/order, 222manifest, config/protocol, метрики/графики, syntheticfixture и MANIFEST. Нет 75GBcache/сырых 222 файлов.

Подключение **после распаковки полного ZIP**, не только текстовой папки Git:

```bash
python load_teacher.py --self-test
```

```python
from load_teacher import TeacherScorer
scorer = TeacherScorer('cuda')  # либо cpu
scores = scorer.score(past, clock, future)  # [B, 2] LONG, SHORT, Beta.mean
```

Пакет в Git частичный: `MANIFEST.json` внутри export описывает **полный Drive ZIP**, поэтому его self-test без .pt и fixture из архива намеренно не пройдёт. Не «чинить» manifest удалением записей. Использовать полный ZIP. Weight file ~67MB исключён из Git, текстовая часть сохранена для чтения.

### 4.4 V3: что изменили

Warmstart v2cycle80, reset optimizer. Teacher lr3e-5, studentlr1e-4, accumulation4outercycles, advantageclip±3. 4096 TRAIN anchors seed1701 для per-direction calibration. Награда открытия `2*(score−b_direction)`, WAIT0; b записывается в checkpoint, recalibrates после teacher step. Энтропия cosine0.10→0.03 за 1500studentupdates вместо adaptivefeedback.

Meta2023: 24 окна×14 дней, 12up12down. Блок 4cycles проверял up/down; objective использовал прирост минимального excess utility относительно max(alwaysLONG, alwaysSHORT, 0). Development4×28days2024 starts03-04, 06-03, 09-02, 12-02. Fresh finaltest84days2025-04-07..2025-06-30. Это **тоже уже просмотренный** период.

96cycles, 24teacher optimizersteps, выбран cycle72; best persistentstudent devnet−246.08 (367LONG, 619SHORT). По 24steps нельзя утверждать глубокую сходимость teacher. Текущие цифры fresh брать из **`fresh_results_at_stop.json` в корне отчёта**, не из более раннего `remote/results/fresh_results.json`.

| Arm | Seed | Test net | Сделки | LONG/SHORT |
|---|---:|---:|---:|---:|
| learned_teacher | 888 | −268.29 | 738 | 430/308 |
| previous_teacher | 888 | −235.79 | 741 | 414/327 |
| direct_money | 888 | −191.53 | 700 | 424/276 |
| learned_teacher | 889 | −403.85 | 739 | 405/334 |

Оставшиеся два arms seed889 не завершены. `previous_teacher` здесь применялся с **новым centered reward adapter**, это не точное повторение исходного v2 learner. Не сравнивать названия как одинаковый протокол.

Остановка подтверждена `user_stop.json`. Последний проверенный remote snapshot был распакован локально; дублирующий tar486MB затем удалён для места. Сохранены локально большие best_teacher/best_student/initial_teacher/resume_state и checkpoints трёх fresh seed888; они **не публикуются в Git и не подтверждены как Drive-backup v3**. Freshseed889 результат есть, сохранность его checkpoint не гарантирована. Для продолжения v4 эти веса не обязательны; при желании повторить полную v3-ablation нельзя предполагать, что они доступны другому агенту.

## 5. Подтверждённый аудит: почему старый результат нельзя принимать

Первичные расчёты: `reports/teacher_training_audit_20261016/{REPORT_RU.md, evidence.json, execution_edge_cases.json}`; точный timingaudit отдельно.

### 5.1 Teacher почти постоянен, а entropy объясняет политику

На 4096 фиксированных TRAIN points v2cycle80:

- mean scores LONG0.48288324475, SHORT0.4212822020;
- std **между условными средними разных точек**, не Beta sampling std: 0.00243521575 и 0.00193745876;
- mean reward `[WAIT0, LONG−0.03423351, SHORT−0.15743560]`.

При предположении активного advantage floor0.1 и финальном alpha0.2461458561 optimum constant reward+entropy:

`pi(a)=softmax(r(a)/(0.1*alpha)) ≈ [0.79964886, 0.19901719, 0.00133395]`.

Реальная mean testpolicy `[0.80385558, 0.19566736, 0.00047706]`. Это сильное объяснение без содержательного timing, **аналитическая диагностика с допущениями**, не доказательство полного отсутствия любого сигнала. Alpha/advantage scale менялись; все фактические batchscale не записаны.

V3cycle72 means примерно[0.4988835752, 0.4233492017], std[0.0023839136, 0.0016982129]. Вычитание отдельного среднего стороны превращает почти constant reward в почти 0, после чего entropy поддерживает почти равномерность. Centering по действиям меняет objective относительно WAIT; это **не безвредная общая baseline** и может удалить настоящий direction advantage вместе с bias.

### 5.2 Аудит момента входа

8000 CPU randomizations =4controls×2000, seed20261016. Исходные 720trades/PNL воспроизведены точно. Контроли: constant LONG probability той же средней частоты, глобальная перестановка, перестановка внутри дня, циклический сдвиг внутри дня. Mean net около 208.73/208.51/205.65/203.92 против 231.67 ученика. Доли случайных controls не хуже actual≈23.1%/22.8%/20.2%/19.0%; недельные block-bootstrap CI разницы включают 0.

Это posthoc диагностика одного уже просмотренного 84daytest и одного studentseed, не независимые открытия. Сохранялись sampled actions, **не полные per-minute probabilities**. Поэтому нельзя восстановить распределение политики и сделать более сильное causal утверждение о teacher.

Для повторения в Git включены компактные `test_execution_vectors.npz`, `randomization_runs.npz` и исходные v2test arrays. `analyze.py` использует cachedexecutionvectors; scratchanchors/goldarrays для этого больше не нужны. Нельзя заменить cachedvector данными другого источника и утверждать exactreproduction.

### 5.3 Несогласованная train/eval задача

Bandit rewards отдельной точке не выражают цену занятости позиции и возможности подождать. В eval OPEN блокирует 120min, WAIT только 1min. Например +1 сейчас может заблокировать +20 через минуту. Valuehead немедленной pointreward не является последовательным V(s). Исправление — единый SMDP/transition semantics и temporalTD, а не только замена MSE/Adam/PCGrad.

### 5.4 Выходы через пропуски и неправильная equity

На 79 913candidatepoints v2test:6493 получали exit>120min, 60 нулевую длительность. Среди 720 исполненных:47>120min, 9 нулевых; max3119min=51h59m. Teacher будущее 120min не содержит результат такой длинной сделки. DailyPnL на entryday, не exitday; нельзя выдавать за closeequity/MTM.

Нельзя исправить это удалением перед входом всех точек, у которых **в будущем** окажется пропуск:это новый lookahead. Нужно явное executionpolicy для непредвиденного пропуска и известный заранее календарь. V4failclosed пока останавливает неоценимый эпизод.

### 5.5 Внешний credit, отбор и reward objective

На 16 384 teacher sampled scores за цикл приходился один общий scalar. Log-prob невыбранных scores/WAIT добавляли шум. Отбор teacher по persistent student, накопившему знания от прежних teacher, и сохранение после ещё не проверенного step не оценивают фактическую полезность кандидата. Fresh-student тест полезен, но не исправляет ошибочный выбор задним числом.

`sumPnL−L2PnL` не зависит от порядка сделок. [10, −10, 10, −10] и [10, 10, −10, −10] одинаковы для objective, а closed drawdown равен 10 против 20. Много мелких коррелированных сделок не обязательно диверсифицируют риск. Функция не выражала заявленную временную стабильность.

### 5.6 Чего аудит не доказал

Не доказаны ни отсутствие сигнала в M1/222 инструментах, ни недостаточность Transformer вообще, ни невозможность meta-RL. Также не доказано, что более высокий LR или меньшая entropy решат проблему. Сглаживание MSE, отрицательный pilot и overfit tiny sample не исключают все причины ошибки. Полный lookahead/data-alignment audit ещё нужен при реальном pilot.

## 6. V4: реализация и границы готовности

Исходники: experiments/sequential_teacher_v4/. Все текущие проверки выполнены на CPU, без выделения новой GPU. Это отдельная реализация; v2/v3 не исправлялись задним числом.

### 6.1 Среда и исполнение

environment.py задаёт Episode: явную поминутную сетку, Open quotes, причинные observations, заранее объявленный scheduled_close. Недостающие котировки обозначаются NaN. Execution задаёт horizon (по умолчанию 120 календарных минут), roundtrip cost ($0.25 на одну унцию) и gamma_per_minute (по умолчанию 1).

Действия: WAIT=0, LONG=1, SHORT=2. WAIT переходит на следующую минуту. OPEN приводит к фиксированному выходу min(entry+horizon, scheduled_close). Пока позиция занята, новых решений нет: переход сразу ведёт к следующему доступному решению. Управляемого CLOSE пока нет. Это корректное сокращённое представление задачи входа с фиксированным выходом, а не обученная exit policy.

В момент scheduled_close действий нет; нулевые сделки запрещены. Entry mask зависит только от текущей доступности котировок и объявленного расписания. Наличие будущей exit quote в mask не читается. Если на запланированном выходе котировки нет, возникает UnpriceableExit. Нет скрытого переноса через выходные и удаления неудачных сделок из среднего. Real pilot пишет unscorable.json и прерывает оценку.

Промежуточные пропуски mark-to-market помечают просадку как неполную. Equity строится по доступным Open с оценкой полной стоимости ликвидации; денежный результат реализуется на выходе. Для live нужны валидированный календарь, bid/ask и slippage. Предопределённые UTC trading windows не являются автоматически календарём брокера.

Для перехода длительностью d continuation discount = gamma^d; награда на выходе дисконтируется gamma^(d−1); в GAE используется lambda^d. При gamma=1 временная ценность денег одинакова, но стоимость занятости позиции и будущих возможностей остаётся. Варианты gamma≠1 покрыты тестами. backward_oracle — динамическое программирование по известному будущему для проверки механики, не supervised action labels для actor.

### 6.2 Модели и последовательный student

model.py: Actor — Transformer Encoder, стандартно width32, depth2, heads4, context8, dropout0; policy head на три действия и value head. Teacher объединяет past/future Transformer encoders и выдаёт два Beta distribution. Малый v4 teacher получает future с двумя каналами; это не интерфейс v2/v3 future[B,120,8]. Совпадение названия Transformer не делает checkpoints совместимыми.

learning.py реализует batched sequential rollout, PPO с ratio clip 0.8..1.2, три epochs на update, minibatch256, AdamW lr3e-4, weight_decay0.001, gradient clip1. GAE lambda_per_minute0.97. Advantage нормируется с minstd1e-6. В коде vl=0.5*(value−target)^2, затем loss=policy+0.5*vl−0.01*entropy: итоговый коэффициент квадратной value-ошибки равен 0.25.

Денежная reward делится на фиксированные $10 только для масштаба оптимизации. Отчётный net остаётся в исходных долларах. Teacher reward: OPEN=2*score−1, WAIT=0. Это ещё не окончательная reward specification. Процентная доходность без капитала/позиционной модели не заявляется.

Actor action RNG и teacher sampling RNG разделены: подключение teacher не должно само менять поток случайных действий actor. В outer REINFORCE credit получает только sampled score реально выбранной стороны при OPEN. WAIT и невыбранная сторона не получают такого credit. Sampling одинаково используется при meta training и fresh-student проверке; dropout в контрольных моделях выключен.

Rollout paths содержат полные вероятности, logp, value, reward, money, duration, discount и terminal flags. pilot.py сохраняет доступные решения в JSONL. Это устраняет прежнюю невозможность восстановить политику из одних sampled actions.

### 6.3 Контролируемые задачи и тесты

synthetic.py создаёт новые procedural episodes с известной причинной структурой:

1. direction: будущее направление задаётся заранее доступным знаком.
2. wait: ранний вход даёт небольшой результат, ожидание — существенно лучший; есть зеркальные направления. Проверяется стоимость упущенной возможности.
3. cross_market: совместный сигнал двух внешних знаков, аналог XOR. Один отдельный внешний признак не заменяет их условное взаимодействие.
4. null: будущее независимо от causal inputs. Проверяется отсутствие подтверждённого положительного edge; нельзя требовать, чтобы каждая случайная реализация была строго отрицательной.

Обычный harness использует 6 features, context8, length48 (для wait — 3). Student seeds11/22/33; test — 128 новых episodes на seed. Gate ratio к DP oracle ≥0.65; для wait ≥0.7 и first-WAIT ≥0.8. Null gate: нижняя 95% оценка mean net не выше нуля. Это диагностический критерий, не доказательство нулевой истинной предсказуемости.

Тесты: test_environment.py — 10; test_learning.py — 4; test_meta_gradient.py — 2. Всего 16. Проверяются position blocking, waiting opportunity, издержки и симметрия знаков, отсутствие future-gap entry mask, конец сессии и нулевая длительность, MTM, discount/GAE, реальные обновления весов, независимость RNG, causal real-panel slice t−8..t−1, exact DP и outer derivative методом конечных разностей. Это проверки конкретных механизмов, не исчерпывающий сертификат всех данных и гиперпараметров.

### 6.4 Прямой PPO student: завершённые результаты

run_gates.py: 250 updates × 16 episodes для каждой задачи и каждого из трёх seed. Первоисточник — synthetic_gates.json; кривые — synthetic_<task>_<seed>_curves.json. Маленькие checkpoints включены в Git.

| Task | Mean net по seed11/22/33 | Дополнительный результат | Gate |
|---|---|---|---|
| direction | 16.512 / 16.476 / 16.450 | oracle ratios ≈0.9985 / 0.99525 / 0.99910 | 3/3 |
| wait | 9.871875 / 9.707422 / 9.652344 | first-WAIT 1 / 0.9921875 / 0.984375; ratios ≈0.99215 / 0.97562 / 0.97008 | 3/3 |
| cross_market | 16.510 / 16.470 / 16.432 | ratios ≈0.998409 / 0.994917 / 0.998013 | 3/3 |
| null | −0.05218 / −0.55608 / −0.39057 | edge не подтверждён | 3/3 |

Точные числа смотреть в JSON. Null policy не обязательно научилась оптимально всегда ждать: entropy может оставлять убыточную активность. Greedy метрики тоже сохранены; нельзя молча подменять ими stochastic результаты.

Первый run использовал 100 updates и не делил денежный reward на $10; wait gate не прошёл. Результаты сохранены в initial_unscaled_gates/. Потом изменились оба фактора: scale и бюджет 100→250. Нельзя целиком приписать улучшение только scale. Финальные результаты привязаны к финальному коду хешами.

### 6.5 Outer REINFORCE teacher: реализован, но перенос не прошёл

meta.py: 24 outer cycles, 16 inner updates на trial, 8 episodes/update, length16, horizon4, cost0.05. Actor каждый trial создаётся заново. Train student seeds11/12/13; selection101/102 каждые четыре цикла после обновления teacher; финальная проверка на seeds201/202/203. На новых seed другие эпизоды. Кандидат действительно обучает reset-учеников перед отбором.

Outer gain = afterNet−beforeNet; EMA baseline0.9/0.1; advantage clip±10; выбранные teacher score log-prob; teacher AdamW3e-4, gradient clip1. Это проверяемое score-function обучение, но эффективность внешнего градиента не доказана. Нормирование credit и зависимость числа действий от политики — место для дополнительного математического аудита.

Controls: learned, initial, ConstantTeacher, ShuffledTeacher, money. ConstantTeacher выдаёт одинаковые Beta(2,2) для обеих сторон: это постоянное распределение со случайными scores, а не детерминированное число 0.5. ShuffledTeacher циклически переставляет distributions между текущими active samples; когда sample один, перестановка ничего не меняет. Это ограниченный контроль, не идеальное уничтожение всего контекста.

| Reward | Seed201 | Seed202 | Seed203 | Mean net |
|---|---:|---:|---:|---:|
| learned | 0.0119005 | −0.00143346 | −0.0539381 | −0.01449 |
| initial | 0.132685 | −0.447883 | 0.130711 | −0.06150 |
| constant | 0.242861 | −0.173332 | 0.242429 | 0.10399 |
| shuffled | 0.00580149 | −0.0628208 | −0.0539381 | −0.03699 |
| money | 1.7442565 | 1.1139223 | 1.2704707 | 1.37622 |

meta_gate.json: complete=true, passed=false. Критерий: mean learned выше каждого control и минимум 2/3 seed выигрывают против каждого control. Ограниченный бюджет не доказывает невозможность meta-RL, но допуск к рынку не получен. candidate_synthetic_teacher.pt — исследовательский отрицательный результат, не рекомендуемые live weights.

### 6.6 Отдельный exact meta-gradient: другой алгоритм

После неуспеха score-function варианта добавлен meta_gradient.py. Смена метода явно отражена в отчёте: это не PPO и не прежнее outer REINFORCE.

На маленьком известном finite SMDP динамическое программирование точно суммирует expected return по WAIT/OPEN trajectories. Actor получает causal observations; teacher видит future и выдаёт Beta.mean scores. Inner loss — отрицательный expected return по teacher reward. Outer loss — отрицательная денежная expected profit causal student на других эпизодах после трёх inner SGD steps. Градиент teacher проходит через эти обновления через functional_call/create_graph. Нет списка правильных действий; есть известный simulator/reward и перечисление действий. Для произвольной большой среды такая точная сумма может быть непрактична.

Параметры: width16, depth1; inner SGD step0.25 × 3; teacher Adam0.001 × 80 cycles; gradient clip1. H4/N12 для direction и cross_market; H2/N3 для wait; cost0.05. Train seeds11/12/13; selection101/102 каждые восемь циклов; test201/202/203. Проверка expected profit на 32 episodes/seed. Каждая task обучает отдельного teacher. Math SDPA и отключённый MHA fastpath нужны для second derivatives; FlashAttention нельзя включать без проверки double backward.

Controls: learned, initial, constant rewards [WAIT0,LONG0.5,SHORT0.5], shuffled rewards по batch, money/10, money_scaled=money. Constant здесь отличается от stochastic Beta(2,2) control в meta.py. Money_scaled проверяет чувствительность к scale при том же большом inner step; это не гарантированно хорошо настроенный денежный baseline.

| Task | Learned по трём seed | Money/10 по трём seed | Mean learned / money | Gate |
|---|---|---|---|---|
| direction | 2.604882, 3.643682, 3.428260 | 0.996955, 1.950308, 2.865199 | 3.2256 / 1.9375 | PASS |
| wait | 0.221873, 1.526320, 1.996669 | 0.502991, 1.119982, 1.477237 | 1.2483 / 1.0334 | PASS |
| cross_market | 0.015568, 0.095213, −0.698025 | 0.206875, 0.222052, 0.172456 | −0.19575 / 0.20046 | FAIL |

Gate означает только положительную mean paired difference против каждого control на трёх seed, без проверки статистической значимости. В wait первый seed проиграл money; после трёх steps результат далёк от oracle9.95. Это не доказательство стабильной сходимости и не выигрыш на каждой инициализации. Seed201/202/203 уже исследованы; дальнейший подбор должен проверяться и на новых seed.

Авторитетные результаты лежат в подкаталогах meta_gradient_direction/, meta_gradient_wait/, meta_gradient_cross_market/: в каждом history, gate.json и маленький teacher.pt. Агрегат — meta_gradient_suite.json. Одноимённые meta_gradient_gate.json/history.json/teacher.pt в корне отчёта относятся к раннему одиночному direction run; они сохранены как история, но не заменяют suite.

Нельзя автоматически применять этот synthetic teacher в PPO pilot: другой learner protocol, width и входы.

### 6.7 Диагностика и реальная панель

diagnostics.py предоставляет per-point means/variance, past/future shuffle, zero future, optional relation к net. Это готовый инструмент, а не уже проведённая полная реальная абляция.

diagnose_legacy.py принимает настоящие v2/v3 inputs: past[N,222,120,96], clock[N,8], future[N,120,8], optional net[N,2], N≥32. Загружает оригинальную модель, считает actual/future_shuffle/past_shuffle/zero_future, сохраняет all_scores.npz и diagnostic.json, CPU batch4. Полный real input cache не восстановлен; реальная абляция ещё не выполнена. Synthetic fixture из Drive проверяет подключение, а не качество teacher на рынке.

reward_audit.py считает net, MTM DD, weekly results и чувствительность к penalty0/0.1/0.5/1. Это инструмент явного выбора предпочтений; окончательная risk utility не выбрана. Нельзя однозначно вывести её из частот таргета или размера датасета.

real_data.py требует все 222 файла manifest. Отсутствующий инструмент не подменяется нулями молча. Используется v3 features.py, затем asinh/finite masks; past context — восемь завершённых баров. Flat feature count =222×96=21312 на минуту. Gold-only явно обнуляет остальные каналы и masks, оставляя совместимый вход.

Конфиг содержит заранее объявленные UTC windows и chronological train/dev/test; между splits нужен purge1560 минут. Panel хранит объединение небольших выбранных окон с восьмиминутным контекстом. Однако features.encode внутри строит рабочую календарную сетку между earliest/latest window плюс warmup. Если разнести окна на десять лет, подготовка снова станет дорогой. Начинать с короткого периода. Для точного восстановления старого teacher нужен его исходный preparation protocol, а не произвольная усечённая история.

pilot.py — gated PPO student runner с режимами money/constant/learned/shuffled, all222 и --gold-only, подробным JSONL решений. Learned mode требует совместимый реально обученный checkpoint features21312/future_steps120 и default architecture. Такого v4 teacher сейчас нет. Synthetic weights не подходят.

Важное ограничение: v4 learning.py/pilot.py пока ориентированы на CPU tensors и NumPy conversion. Готового большого CUDA runner нет. Перенос на GPU требует согласовать device, RNG, data ownership и second derivatives, затем повторить проверки. Старый v3 GPU cache не оправдывает использование его неправильного inner loop.

Реальный outer meta-training runner не завершён. meta.py и meta_gradient.py создают synthetic episodes. Нужен адаптер real episode factory, causal/future views, расписаний, train/meta/dev separation, fresh-student selection и обработки неполных котировок. Наличие panel builder и student pilot не равно готовому real teacher pipeline.

check_market_execution.py проверил только исполнение на девяти заранее объявленных окнах UTC08:00–18:00, Feb28–Mar11,2022: train Feb28–Mar4, dev Mar7–8, test Mar10–11. Все окна без пропусков; 90 always-LONG/SHORT smoke trades (по пять каждой стороны в день) закрылись через 120 минут внутри окна. Обученной стратегии там нет. Это диагностика исполнения, не прибыльности. Результат market_execution_smoke.json, конфиг pilot_windows.json.

## 7. Доказательства и блокировки запуска

provenance.py снимает SHA256 исходников при старте каждого эксперимента. COMMON: environment.py, learning.py, model.py, synthetic.py; дополнительно соответствующий runner. gates.py проверяет хеш evidence, code_hashes и общий source_hashes. finalize_report.py повторяет 16 тестов, читает готовые научные результаты, проверяет provenance и формирует report/figures/manifest. Перегенерация отчёта не делает старые результаты подходящими новому коду.

| Stage | Обязательные проверки | Текущее состояние |
|---|---|---|
| real_pilot | mechanics + synthetic + meta REINFORCE | allowed:false, meta failed |
| exact_pilot | mechanics + synthetic + meta_gradient suite | allowed:false, cross_market failed |
| scale_up | mechanics + synthetic + meta + real_pilot + untouched_test_protocol | Не допущено |

exact_pilot пока является gate, а не завершённым реальным exact learner runner. Его возможный PASS также не означает автоматической готовности PPO pilot с этим teacher. Scale-up contract сейчас связан с PPO/meta; альтернативному алгоритму нужен отдельный согласованный протокол.

Synthetic PASS проверяет части механики и обучаемость. Он не доказывает market edge. Честный текущий статус: основа стала проверяемой, teacher transfer ещё не решён, большое обучение не начато.

## 8. Восстановление и проверка без GPU

### 8.1 Checkout, зависимости, пути

Использовать branch handoff/2026-10-16-teacher-v4 репозитория Flyingzaptop/xauusd; после публикации этот же снимок доступен в main, если main обновился без конфликта. Старый data/M1 уже находился в repository; новая передача не добавляет большие сырые данные.

Многие скрипты используют абсолютные /workspace/xauusd и /content/meta_reward. Самое простое воспроизведение без изменения scientific hashes — checkout/worktree по /workspace/xauusd. Если проект находится иначе, можно создать ссылку на этот путь только когда он свободен; не стирать существующую работу. Другой вариант — явно портировать paths, сохранить исторические results и повторить затронутые gates.

Старая команда /workspace/colab-tools/bin/python — особенность прежней машины. Новый агент может создать свою venv и установить experiments/sequential_teacher_v4/requirements.txt. Проверенная среда: Python3.12.14, torch2.14.1+cpu, numpy2.5.3, pandas3.0.6, pyarrow26.0.0, matplotlib3.11.2. Requirements — lower bounds, не полный lockfile. Другая версия Torch/attention backend требует проверки second derivatives/fixture; битовое совпадение чисел не обещается.

### 8.2 Быстрая проверка

Из корня repository:

~~~bash
python -m unittest discover -s experiments/sequential_teacher_v4 -p 'test_*.py' -v
python experiments/sequential_teacher_v4/gates.py --stage real_pilot --reports reports/sequential_teacher_v4_20261016
python experiments/sequential_teacher_v4/gates.py --stage exact_pilot --reports reports/sequential_teacher_v4_20261016
~~~

Ожидается 16 PASS. Обе gate-команды сейчас возвращают exit code2: полезность teacher не подтверждена. Не снижать критерии ради зелёного статуса.

Полный повтор перезаписывает одноимённые артефакты: сначала сохранить baseline commit или копию results. Запускать последовательно. Прежняя workspace имела два CPU и ~9.7GB RAM; параллельные Torch jobs только создавали oversubscription.

~~~bash
python experiments/sequential_teacher_v4/run_gates.py
python experiments/sequential_teacher_v4/meta.py
python experiments/sequential_teacher_v4/meta_gradient.py --task direction
python experiments/sequential_teacher_v4/meta_gradient.py --task wait
python experiments/sequential_teacher_v4/meta_gradient.py --task cross_market
python experiments/sequential_teacher_v4/finalize_report.py
~~~

run_validation.py объединяет unit tests, эти исследования, market execution smoke и финализацию; timeout600s на stage. Для полного run_validation нужен XAUUSD_41.parquet по пути, ожидаемому check_market_execution.py. Без него synthetic suite работает, а market smoke — нет. Это недостающая raw dependency, не доказательство сбоя алгоритма.

CLI budgets: run_gates.py --updates 250 --tasks direction wait cross_market null --seeds 11 22 33; meta.py --cycles 24 --inner-updates 16; meta_gradient.py --cycles 80 --task direction|wait|cross_market. Частичный запуск tasks/seeds перезапишет общий JSON: не выдавать его за полный suite.

### 8.3 Старый аудит и full-context teacher diagnostics

~~~bash
python reports/v2_entry_timing_audit_20261016/analyze.py
python reports/v2_entry_timing_audit_20261016/make_report.py
~~~

Нужен корректный путь /workspace/xauusd. Компактные исходные v2 test arrays, execution vectors и randomization results включены. analyze.py использует cached vectors, поэтому удалённые scratch anchors/gold arrays не обязательны. 8000 рандомизаций не нужно повторять просто для чтения отчёта.

Пример полной реальной диагностики:

~~~bash
python experiments/sequential_teacher_v4/diagnose_legacy.py --inputs /path/to/actual_222_contexts.npz --checkpoint /path/to/unpacked_v2/best_teacher.pt --out /path/to/new_diagnostic
~~~

Сначала восстановить настоящие tensors по исходному feature code, порядку инструментов, временному выравниванию и warmup. Полный 75GB cache не обязателен, если точно восстановлены выбранные окна и предшествующая rolling history. Synthetic fixture не заменяет actual_222_contexts.

### 8.4 Команды будущего real pilot

После успешных допусков и подготовки внешнего протокола:

~~~bash
python experiments/sequential_teacher_v4/real_data.py --raw-dir /path/to/all_222_parquets --config reports/sequential_teacher_v4_20261016/pilot_windows.json --out /path/to/small_panel
python experiments/sequential_teacher_v4/pilot.py --panel /path/to/small_panel --out /path/to/pilot_money --mode money --seed 501
~~~

Сейчас второй команде положено отказать по meta gate. Первая требует все 222 файла. --gold-only означает явную абляцию, а не замену основного набора. Learned/shuffled modes требуют real teacher checkpoint, которого ещё нет. pilot_windows.json — формат и диагностические сессии, не новый слепой финальный тест.

## 9. Следующие действия в конкретном порядке

### A. Зафиксировать исходную точку

Прочитать аудит, проверить payload manifest, 16 tests и оба отрицательных gate. Сохранить commit как baseline. Не возобновлять v3. Не менять одновременно model depth, features, execution, reward, entropy и optimizer: иначе причин изменения результата не установить.

### B. Решить дешёвый вопрос: почему teacher не помогает на cross_market?

Прямой PPO student умеет решать synthetic cross_market почти идеально. Это не неразрешимая задача для всей малой архитектуры. Но там 250 PPO updates, а exact meta оценивает три SGD steps другого learner; нужен matched-budget контроль прежде вывода о teacher.

Приоритетные эксперименты:

1. Измерить teacher mean, межточечную variance и динамику. Проверить past/future dependence: нулевое будущее, перестановка контекста, постоянная награда. Не объявлять PnL единственной автоматически правильной teacher меткой.
2. Проверить outer gradient norm/SNR и finite differences на конкретном cross_market batch. Разделить влияние inner step0.25, количества шагов, насыщения teacher и student initialization.
3. Сравнить teacher с money при равном inner compute и заранее записанном scale/LR protocol. После подбора на 201/202/203 нужны новые проверочные seeds/data.
4. Разделить stochastic reward и deterministic mean. Score-function и exact варианты отличаются больше чем одним outer optimizer.
5. Сохранить direction/wait positive controls и null. Нельзя исправлять cross_market утечкой будущего в actor или скрытой заменой RL на action labels.

Это направления поиска, не уже доказанные fixes. Если teacher не помогает при сопоставимом бюджете, честный отрицательный результат полезнее ещё одного дорогого раунда.

### C. Завершить реальный протокол до большого запуска

- Зафиксировать position units, календарь, spread/fees/slippage и unexpected-missing execution. Одна унция/$0.25 — сценарий, не полноценный контракт исполнения.
- Определить внешний objective через net money, MTM equity и недельный риск с явными trade-offs. Risk aversion нельзя однозначно вывести из данных. Сравнение коэффициентов показывать как sensitivity.
- Сделать real episode factory и outer reset-student validation. Train/eval должны использовать одну среду; teacher future не должен попадать в actor features или selection mask.
- Сравнить all222/gold-only, learned/constant/shuffled/initial/money, always-LONG/SHORT/flat, а также timing controls с похожей частотой/экспозицией. Использовать несколько student initializations.
- Отбирать обновлённый teacher по реальному обучению fresh students с одинаковыми initial weights, optimizer state и budget, а не по накопленному persistent student.
- Сохранять per-point probabilities, executed trades, MTM equity, actual durations, reward components, policy/value/entropy gradient norms, teacher mean/variance, seeds, compute и фактический объём увиденных данных.
- Только после малого real pilot решать вопрос десяти лет, глубины Transformer и большого GPU cache.

### D. Затем производительность и масштабирование

75GB cache плюс model/optimizer/gradients/activations могут не поместиться в 96GB. Название GPU не заменяет проверку доступной памяти. Измерить несколько batches, H2D, throughput и peak allocation. Cache в VRAM использовать, если он ускоряет полный цикл; memmap/pinned streaming — если выгоднее. Обещать полчаса вместо двух без измерения нельзя.

Отчёт должен разделять число подготовленных строк, seen examples, уникальные окна, полные passes, inner updates, outer teacher steps, fresh students и оплаченное wall time. В v3 именно это различие было потеряно.

## 10. Какие данные уже просмотрены

- Январь/февраль2024 многократно использованы в capacity/classification/hypernet/curriculum.
- Весь 2021 год просмотрен спектральным анализом; Oct–Dec2021 не стали новым blind test при Transformer.
- Старый teacher: train2016–2022, meta2023, dev2024 (v2 — март; v3 — март/июнь/сентябрь/декабрь).
- V2 test: Jan6–Mar31,2025; v3 test: Apr7–Jun30,2025. После аудита это исследовательские периоды.
- LightGBM имел свой protocol и широкий датасет. До выбора final test прочитать его PROTOCOL/metadata. Нельзя объявлять Jul2025–Sep2026 непросмотренными лишь потому, что их не было в v3 test.
- V4 execution smoke Feb–Mar2022 — техническая диагностика.
- Synthetic seeds201/202/203 и11/22/33 уже участвовали в интерпретации метода. Для дальнейшего подбора нужны новые проверочные реализации.

Утверждённого ещё не просмотренного final test protocol на момент передачи нет. Это отдельная блокировка scale_up. Purge, overlapping labels, пропуски, множественный поиск и posthoc selection должны быть явно описаны.

## 11. Состав передачи и восстановление артефактов

### В Git

Перенесены доступные текстовые исходники, протоколы и отчёты последующего workspace: loss/PCGrad/capacity/classification/hypernet/curriculum/Transformer/FFT/STFT/wavelets/spectral analysis/LightGBM/entry evaluator/meta-RL. Также компактные JSON/CSV, графики/HTML, исторические логи, v2/v3 configuration/schema/source hashes/curves/fresh evaluations, аудиты, v4 tests/results/gates и маленькие synthetic checkpoints.

Маленькие binary exceptions включены намеренно: spectral year context/normalization для воспроизведения; v2 execution vectors/randomizations/test decisions; v4 контрольные .pt обычно33–164KB. Большие v2/v3 weights и многогигабайтные arrays не входят.

docs/handoff_20261016/payload_manifest.json содержит sizes/SHA256 перенесённых research files. Его область покрытия явно описана в manifest; он не хеширует сам себя. excluded_local_files.json перечисляет исключённые локальные артефакты с размерами без дорогого хеширования десятков GB. Наличие имени в списке не означает доступность файла на новой машине.

### В Drive

Полный ZIP v2 cycle80 с weights/optimizer/fixture/code/schema: ссылка и SHA256 в разделе4.3. Исходные 222 Parquet — в папке пользователя. Credentials и права доступа в Git не переносятся.

### Не гарантированы на новой машине

Большие v3 weights, исторические checkpoints других экспериментов, 22.865GB HDF5, feature/prediction arrays и архивы прежней workspace. Полный удалённый 75GB teacher feature cache утрачен с VM. Не выдавать его за существующий backup. Код и raw данные дают путь повторного вычисления, но старые случайные состояния весов без backup восстановить нельзя.

Старые MANIFEST.json и ссылки /workspace/... в исторических отчётах могут относиться к исключённым weights/arrays. Это сохранённый record прежнего эксперимента, а не обещание присутствия каждого файла в Git. Для восстановления сверяться с inventory и текущим handoff.

Repository публичный. В комплект не включаются OAuth codes, access/refresh tokens, ADC credentials и секретные ключи. Старые одноразовые Google коды из разговора не переносить в документы. Windows Qwen/OpenCode adapter — отдельный проект.

## 12. Ответы, которые не нужно заново выяснять у пользователя

**Прибыльный полезный teacher уже найден?** Нет. Сохранён v2 candidate; его all-LONG profit объясним дрейфом и не подтверждает timing edge.

**Кто закрывал сделки?** Executor по фиксированному правилу. В v4 выход тоже фиксирован, теперь согласован с обучением. Обучаемое закрытие — будущий этап.

**Scores — вероятности профита?** Нет. Это reward scores; диапазон (0,1) не даёт calibration.

**Есть отдельный нормализатор?** У v2/v3 нет fitted scaler: нужны точные features/data/order/warmup. У spectral pipeline есть normalization.npz.

**Нужно продолжить v3?** Не автоматически: пользователь остановил его для переработки. Продолжать v4/audit.

**Что готово в v4?** Environment, PPO, synthetic controls, два teacher experiments, diagnostics, panel builder и gated student pilot. Не готовы полезный межрыночный teacher, full real ablation, real outer pipeline, окончательный risk objective/calendar, independent final test и большой CUDA runner.

**Ближайший полезный результат?** Matched-budget эксперимент, где teacher помогает новым student seeds на cross_market и сохраняет direction/wait/null controls. Либо обоснованный отрицательный результат и пересмотр метода без подгонки старого test.

**Надо повторять настройку Colab?** Нет: следующий агент уже её сделал. Это не означает необходимости держать GPU выделенной во время анализа документов.

---

Передача сохраняет результаты вместе с причинами отказа от прежней интерпретации. Цель следующей итерации — проверить пользу learned reward для causal student на новых данных при корректном исполнении.
