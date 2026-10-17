#!/bin/bash
# Full run B: convert remaining raw zips (2 workers) -> final-test daily files -> A100 session -> chunked upload -> background training.
# Idempotent: safe to re-run after a local container restart (the Colab VM keeps running independently).
set -u
cd "$(dirname "$0")";A=/home/user/crypto/aggtrades;O=/home/user/crypto/seq;D=/home/user/crypto/daily;P=/home/user/venv/bin/python
C=~/.local/bin/colab;S=seqB;B=https://data.binance.vision/data/futures/um
conv(){ for f in $(ls $A/$1-aggTrades-*.zip 2>/dev/null | sort); do b=$(basename $f .zip);mo=${b##*aggTrades-}
  $P -W ignore prep.py $1 $f $O/$1-$mo.npz | tail -1 | cut -c1-90 && rm $f; done; }
conv BTCUSDT & conv ETHUSDT & wait
mkdir -p $D
for s in BTCUSDT ETHUSDT; do
  [ -f $O/$s-final.npz ] && continue
  for d in $(seq -w 1 16); do fn=$D/$s-aggTrades-2026-10-$d.zip;[ -f $fn ] || curl -sS -f --retry 5 -o $fn $B/daily/aggTrades/$s/$s-aggTrades-2026-10-$d.zip || echo missing $s $d; done
  $P -W ignore prep.py $s $(ls $D/$s-*.zip | sort) $O/$s-final.npz | tail -1 | cut -c1-90
done
echo CONVERTED
for k in 1 2 3 4 5; do $C status -s $S >/dev/null 2>&1 && break; timeout 900 $C new -s $S --gpu A100 2>&1 | tail -1; sleep 30; done
echo 'import os;[os.makedirs(d,exist_ok=True) for d in ("/content/seq/.parts","/content/data/.parts","/content/outB")];print("dirs")' | timeout 120 $C exec -s $S 2>&1 | tail -1
./colab_put.sh $S /content/seq model.py data.py train.py
# skip data files already uploaded with the right size
have=$(echo 'import os;print(" ".join(f"{f}:{os.path.getsize("/content/data/"+f)}" for f in os.listdir("/content/data") if f.endswith(".npz")))' | timeout 120 $C exec -s $S 2>&1 | tail -1)
for f in $O/BTCUSDT-20*.npz $O/ETHUSDT-20*.npz $O/*-final.npz; do
  echo " $have " | grep -q " $(basename $f):$(stat -c %s $f) " || ./colab_put.sh $S /content/data $f
done
TR=$(for s in BTCUSDT ETHUSDT; do for m in 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-03; do echo -n "/content/data/$s-$m.npz "; done; done)
VA=$(for s in BTCUSDT ETHUSDT; do for m in 2026-04 2026-05 2026-06; do echo -n "/content/data/$s-$m.npz "; done; done)
TE=$(for s in BTCUSDT ETHUSDT; do for m in 2026-07 2026-08 2026-09; do echo -n "/content/data/$s-$m.npz "; done; done)
CMD="cd /content/seq && nohup python -W ignore train.py --train $TR --val $VA --test $TE --final /content/data/BTCUSDT-final.npz /content/data/ETHUSDT-final.npz --d 384 --depth 12 --heads 8 --batch 8 --pre_steps 15000 --ft_steps 15000 --eval_every 2500 --val_stride 600 --eval_stride 120 --eval_batch 64 --out /content/outB > /content/outB/train.log 2>&1 &"
printf 'import subprocess,os\nif os.path.exists("/content/outB/train.log"):print("already running")\nelse:subprocess.Popen(%s,shell=True);print("launched")\n' "$(printf '%s' "$CMD" | $P -c 'import sys,json;print(json.dumps(sys.stdin.read()))')" | timeout 120 $C exec -s $S 2>&1 | tail -1
