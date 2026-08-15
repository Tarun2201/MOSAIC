#!/bin/bash
# Train v12 (from scratch), then test (test_map_v4.py works — base U-Net) + breakdown.
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
CSV="${CSV:-$ROOT/data/all_data_cleaned_with_splits_60_20_20_full.csv}"
MD="${MD:-$ROOT/results/seg_map_refine/exp_v12_scratch/4_3_2_1_/map_refine_v12}"
cd "$ROOT/federated_map_refine"
mkdir -p logs
bash run_map_v12.sh > logs/train_v12.log 2>&1
echo "=== v12 test ===" > logs/test_v12.log
$PY test_map_v4.py --clients "1,2,3,4" --csv_path "$CSV" \
    --model_dir "$MD/" --base_filters 32 --img_size 224 --K 3 --tta --gate_tau 0.5 --gpu 0 --eval_3d \
    >> logs/test_v12.log 2>&1
echo "=== breakdown ===" >> logs/test_v12.log
$PY analyze_errors.py "$MD/test_results" >> logs/test_v12.log 2>&1
echo "=== ALL DONE ===" >> logs/test_v12.log
