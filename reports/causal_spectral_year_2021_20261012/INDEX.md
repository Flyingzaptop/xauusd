# Данные готовы к анализу: 2021 год

**354,340 реальных M1 свечей**, шаг 1. Паузы склеены без вставок.
Первый timestamp: 2021-01-03T23:00:00; последний: 2021-12-31T21:59:00.
Полный календарный период: 2021-01-01…2022-01-01, все доступные записи.

STFT: окна 128 / 512 / **8192**, 31 / 127 / 2048 компонент.
Односторонний Morlet: **75 масштабов 4…8192**, прежние 48 сохранены.
Комплексные коэффициенты complex128, энергия и фаза float64.
Нормы зафиксированы только по декабрю 2020; warmup с августа 2020.
Каждая строка использует только историю и текущую уже закрытую свечу.

Подготовка и технические проверки завершены. **Изображений, анализа рыночных
паттернов, прогнозов, моделей или торговых выводов нет.**

Общий размер HDF5: 22.865 ГБ (21.295 GiB).
Данные хранятся помесячно: можно читать день/блок, не загружая весь год.

## Файлы данных

| Месяц | Реальных свечей | Размер, ГБ | Файл |
|---|---:|---:|---|
| 2021-01 | 27,421 | 1.770 | [2021-01.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-01.h5) |
| 2021-02 | 27,359 | 1.766 | [2021-02.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-02.h5) |
| 2021-03 | 31,737 | 2.048 | [2021-03.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-03.h5) |
| 2021-04 | 28,852 | 1.862 | [2021-04.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-04.h5) |
| 2021-05 | 28,860 | 1.862 | [2021-05.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-05.h5) |
| 2021-06 | 30,352 | 1.959 | [2021-06.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-06.h5) |
| 2021-07 | 29,992 | 1.936 | [2021-07.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-07.h5) |
| 2021-08 | 30,460 | 1.965 | [2021-08.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-08.h5) |
| 2021-09 | 30,108 | 1.942 | [2021-09.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-09.h5) |
| 2021-10 | 28,951 | 1.868 | [2021-10.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-10.h5) |
| 2021-11 | 29,955 | 1.933 | [2021-11.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-11.h5) |
| 2021-12 | 30,293 | 1.955 | [2021-12.h5](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/months/2021-12.h5) |

## Документация и чтение

- [SCHEMA.md](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/SCHEMA.md) — поля, формулы, единицы, временная семантика.
- [days.csv](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/days.csv) — индекс всех 365 дат, диапазоны строк внутри месяцев.
- [dataset.json](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/dataset.json) — размеры, настройки, версии и SHA256 HDF5.
- [verification.json](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/verification.json) — технические проверки и причинность.
- [MANIFEST.json](/workspace/xauusd/reports/causal_spectral_year_2021_20261012/MANIFEST.json) — все файлы и SHA256.
- `experiments/spectral_year/dataset.py` — загрузчик дня/блоков и выбранных периодов.
- `experiments/spectral_year/README.md` — воспроизведение и пример чтения.

Воспроизводимый исходный снимок `input_context.npz` включает только timestamp,
Close и номера строк оригинала. Он позволяет заново вычислить весь год без
загрузки исходных десяти лет. Производные коэффициенты сохранены отдельно
в 12 HDF5. Большие HDF5 уже сжаты без потерь и не продублированы в архиве
кода; маленький архив содержит исходный снимок, код, нормы, индексы и проверки.
