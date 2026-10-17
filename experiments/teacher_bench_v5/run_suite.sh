#!/bin/bash
# Sequential suite (CPU). Each run writes reports/teacher_bench_v5/<name>/result.json
cd "$(dirname "$0")/../.."; P=${PY:-/home/user/venv/bin/python}; R=reports/teacher_bench_v5
run(){ name=$1; shift; mkdir -p $R/$name; $P -W ignore experiments/teacher_bench_v5/bench.py --out $R/$name "$@" > $R/$name/console.log 2>&1; echo "done $name $(date -u +%T)"; }
run direction_K10          --task direction       --K 10 --n-inner 32
run cross_market_K10       --task cross_market    --K 10 --n-inner 32
run noisy_direction_n4_K10 --task noisy_direction --K 10 --n-inner 4
run noisy_cross_n8_K10     --task noisy_cross_market --K 10 --n-inner 8
run null_K10               --task null            --K 10 --n-inner 32
