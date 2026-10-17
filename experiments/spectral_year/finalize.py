"""Publish local file index / hashes / reproduction bundle, without analysis."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    a=p.parse_args();meta=json.loads((a.out/'dataset.json').read_text());qa=json.loads((a.out/'verification.json').read_text())
    assert qa['passed'] and qa['year_rows']==meta['year_candles']
    assert not list(a.out.rglob('*.partial.h5'))
    forbidden=['*.png','*.jpg','*.svg','*.html','*.pdf']
    assert not any(list(a.out.rglob(pattern)) for pattern in forbidden)
    text=f'''# Данные готовы к анализу: 2021 год

**{meta['year_candles']:,} реальных M1 свечей**, шаг 1. Паузы склеены без вставок.
Первый timestamp: {meta['first_timestamp']}; последний: {meta['last_timestamp']}.
Полный календарный период: 2021-01-01…2022-01-01, все доступные записи.

STFT: окна 128 / 512 / **8192**, 31 / 127 / 2048 компонент.
Односторонний Morlet: **75 масштабов 4…8192**, прежние 48 сохранены.
Комплексные коэффициенты complex128, энергия и фаза float64.
Нормы зафиксированы только по декабрю 2020; warmup с августа 2020.
Каждая строка использует только историю и текущую уже закрытую свечу.

Подготовка и технические проверки завершены. **Изображений, анализа рыночных
паттернов, прогнозов, моделей или торговых выводов нет.**

Общий размер HDF5: {meta['total_hdf5_bytes']/1e9:.3f} ГБ ({meta['total_hdf5_bytes']/1024**3:.3f} GiB).
Данные хранятся помесячно: можно читать день/блок, не загружая весь год.

## Файлы данных

| Месяц | Реальных свечей | Размер, ГБ | Файл |
|---|---:|---:|---|
'''
    for m in meta['months']:
        text+=f"| {m['month']} | {m['rows']:,} | {m['bytes']/1e9:.3f} | [{m['month']}.h5]({a.out/m['file']}) |\n"
    text+='''
## Документация и чтение

- [SCHEMA.md](SCHEMA.md) — поля, формулы, единицы, временная семантика.
- [days.csv](days.csv) — индекс всех 365 дат, диапазоны строк внутри месяцев.
- [dataset.json](dataset.json) — размеры, настройки, версии и SHA256 HDF5.
- [verification.json](verification.json) — технические проверки и причинность.
- [MANIFEST.json](MANIFEST.json) — все файлы и SHA256.
- `experiments/spectral_year/dataset.py` — загрузчик дня/блоков и выбранных периодов.
- `experiments/spectral_year/README.md` — воспроизведение и пример чтения.

Воспроизводимый исходный снимок `input_context.npz` включает только timestamp,
Close и номера строк оригинала. Он позволяет заново вычислить весь год без
загрузки исходных десяти лет. Производные коэффициенты сохранены отдельно
в 12 HDF5. Большие HDF5 уже сжаты без потерь и не продублированы в архиве
кода; маленький архив содержит исходный снимок, код, нормы, индексы и проверки.
'''
    for name in ['SCHEMA.md','days.csv','dataset.json','verification.json','MANIFEST.json']:
        text=text.replace(f']({name})',f']({a.out/name})')
    (a.out/'INDEX.md').write_text(text)
    ready={'ready':True,'year':2021,'rows':meta['year_candles'],'step':1,'max_period':8192,
           'hdf5_bytes':meta['total_hdf5_bytes'],'technical_verification_passed':True,
           'images_created':False,'scientific_analysis_performed':False,'gpu_used':False}
    (a.out/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
    code=a.root/'experiments/spectral_year'
    files=sorted(p for p in code.iterdir() if p.is_file())
    files+=sorted(p for p in a.out.iterdir() if p.is_file() and p.name!='MANIFEST.json')
    entries={str(p.relative_to(a.root)):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in files}
    # HDF5 hashes were computed by a full-file read after each file was closed.
    for m in meta['months']:
        path=a.out/m['file'];assert path.stat().st_size==m['bytes']
        entries[str(path.relative_to(a.root))]={'bytes':m['bytes'],'sha256':m['sha256']}
    manifest=a.out/'MANIFEST.json';manifest.write_text(json.dumps({'files':entries,'hdf5_hashes_full_read_after_write':True},indent=2)+'\n');files.append(manifest)
    with tarfile.open(a.bundle,'w:gz',compresslevel=1) as t:
        for path in files:t.add(path,arcname=str(path.relative_to(a.root)))
    with tarfile.open(a.bundle,'r:gz') as t:
        for path in files:
            if path==manifest:continue
            key=str(path.relative_to(a.root));assert hashlib.sha256(t.extractfile(key).read()).hexdigest()==entries[key]['sha256']
    print(json.dumps({**ready,'bundle':str(a.bundle),'bundle_bytes':a.bundle.stat().st_size,'hdf5_files':12}))


if __name__=='__main__':main()
