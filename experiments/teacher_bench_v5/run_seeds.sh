#!/bin/bash
cd "$(dirname "$0")/../.."; P=${PY:-/home/user/venv/bin/python}; R=reports/teacher_bench_v5
for s in 1 2; do n=noisy_direction_n4_K10_c600_tseed$s; mkdir -p $R/$n
  $P -W ignore experiments/teacher_bench_v5/bench.py --task noisy_direction --n-inner 4 --K 10 --cycles 600 --n-test 48 --seed $s --out $R/$n > $R/$n/console.log 2>&1; echo done $n; done
