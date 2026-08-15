#!/bin/bash
# Stage 2: turn the trained binary classifier + aggregation net into CAM
# pseudo-labels for Stage 3.
#
# Binary-only: no multiclass checkpoints are needed.
#
# Override any variable from the environment, e.g.
#   CSV_PATH=/data/splits.csv BASE_PATH=/data/checkpoints bash run_save_pseudo_labels.sh
set -e

CSV_PATH="${CSV_PATH:-./data/all_data_cleaned_with_splits_60_20_20_full.csv}"
OUT_DIR="${OUT_DIR:-./pseudo_labels_binary}"
CLIENTS="${CLIENTS:-1,2,3,4}"
SPLITS="${SPLITS:-train,val}"
IMG_SIZE="${IMG_SIZE:-[224]}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU="${GPU:-0}"
FREQ_BANDS="${FREQ_BANDS:-8}"
PY="${PY:-python}"

# Must match the Stage-1 --save_dir values (see run_all_exp.sh).
BASE_PATH="${BASE_PATH:-./checkpoints}"
CNET_BASE="${CNET_BASE:-${BASE_PATH}/cnet_binary_personalization/exp_4/cnet_stat_proto_exp4_spectral}"
AGG_BASE="${AGG_BASE:-${BASE_PATH}/agg_net_binary_personalization/exp_4/spectral}"

# Both trainers nest their checkpoints under a directory named by REVERSING the
# client list and joining with underscores, e.g. "1,2,3,4" -> "4_3_2_1_".
# The agg net adds a further "num_of_bands_<N>" level.
CLIENT_SUFFIX=""
IFS=',' read -ra _clients <<< "$CLIENTS"
for c in "${_clients[@]}"; do
    CLIENT_SUFFIX="${c}_${CLIENT_SUFFIX}"
done

CNET_DIR="${CNET_DIR:-${CNET_BASE}/${CLIENT_SUFFIX}}"
AGG_DIR="${AGG_DIR:-${AGG_BASE}/${CLIENT_SUFFIX}/num_of_bands_${FREQ_BANDS}}"

# argparse wants a Python literal list.
CLIENTS_LIT="[${CLIENTS}]"

echo "=========================================="
echo "Stage 2: save binary CAM pseudo-labels"
echo "  clients   : ${CLIENTS_LIT}"
echo "  splits    : ${SPLITS}"
echo "  csv       : ${CSV_PATH}"
echo "  cnet dir  : ${CNET_DIR}"
echo "  agg  dir  : ${AGG_DIR}"
echo "  out dir   : ${OUT_DIR}"
echo "=========================================="

# Fail early with a readable message rather than deep inside the model loader.
missing=0
for c in "${_clients[@]}"; do
    for f in "${CNET_DIR}/_personalized_unet_client_${c}.pth" \
             "${CNET_DIR}/_personalized_modality_client_${c}.pth" \
             "${AGG_DIR}/_personalized_scoring_client_${c}_best.pth"; do
        if [ ! -f "$f" ]; then
            echo "MISSING: $f"
            missing=1
        fi
    done
done
if [ "$missing" -ne 0 ]; then
    echo
    echo "Required Stage-1 checkpoints are missing. Run run_all_exp.sh first, or"
    echo "point CNET_DIR / AGG_DIR at the directories that hold them."
    exit 1
fi

$PY save_pseudo_labels_binary.py \
    --clients "$CLIENTS_LIT" \
    --csv_path "$CSV_PATH" \
    --img_size "$IMG_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --gpu "$GPU" \
    --bin_pretrained_dir "$CNET_DIR" \
    --bin_modality_pretrained_dir "$CNET_DIR" \
    --bin_score_pretrained_dir "$AGG_DIR" \
    --bin_score_modality_pretrained_dir "$AGG_DIR" \
    --splits "$SPLITS" \
    --out_dir "$OUT_DIR"

echo
echo "Done. Manifest written to ${OUT_DIR}/manifest.csv"
echo "Feed it to Stage 3 with --manifest_path ${OUT_DIR}/manifest.csv"
