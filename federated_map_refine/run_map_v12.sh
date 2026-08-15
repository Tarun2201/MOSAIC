set -e
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${PY:-python}"
CSV_PATH="${CSV_PATH:-$ROOT/data/all_data_cleaned_with_splits_60_20_20_full.csv}"
MANIFEST="${MANIFEST:-$ROOT/pseudo_labels_binary/manifest.csv}"
SAVE_DIR="${SAVE_DIR:-$ROOT/results/seg_map_refine/exp_v12_scratch/}"
cd "$ROOT/federated_map_refine"
$PY train_map_v12.py \
    --clients "1,2,3,4" \
    --csv_path "$CSV_PATH" --manifest_path "$MANIFEST" --save_dir "$SAVE_DIR" \
    --K 3 --tta \
    --num_rounds 100 --epochs 1 --batch_size 16 --img_size 224 --num_workers 6 \
    --learning_rate 1e-3 --patience 25 --base_filters 32 \
    --warmup_rounds 5 --ramp_rounds 20 --alpha_max 0.7 \
    --tv_alpha 0.3 --tv_beta 0.7 --cls_weight 0.5 --crf_weight 0.1 --gate_tau 0.5 \
    --crf_kernel 5 --crf_dilations "1" \
    --spa_weight 5.0 --rcsa_weight 5.0 --spa_bands 8 --poverty_scale 0.5 \
    --clean_targets 1 --r_in 2 --r_out 4 --speckle_min 10 \
    --fnfd_weight 0 --fnfd_poverty 1.0 \
    --save_rounds 1 --round_min_val 0.60 --grad_clip 10.0 \
    --device cuda:0
