#!/bin/bash
# Stage 1: federated binary classifier (cnet) -> federated aggregation network.
# Override any variable from the environment, e.g.
#   CSV_PATH=/data/splits.csv BASE_PATH=/data/weights bash run_all_exp.sh

CSV_PATH="${CSV_PATH:-./data/all_data_cleaned_with_splits_60_20_20_full.csv}"
NUM_ROUNDS="${NUM_ROUNDS:-100}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
IMG_SIZE="${IMG_SIZE:-224}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
CLIENTS="${CLIENTS:-1,2,3,4}"
DEVICE_GPU0="${DEVICE_GPU0:-0}"
DEVICE_GPU1="${DEVICE_GPU1:-cuda:0}"
NUM_CLASSES="${NUM_CLASSES:-2}"

# Base path for pretrained models
BASE_PATH="${BASE_PATH:-./checkpoints}"
echo "Base Path: $BASE_PATH"

# Create logs directory if it doesn't exist
LOG_DIR="${LOG_DIR:-./nohups}"
mkdir -p ${LOG_DIR}

nohup bash -c '
    python train_federated_with_dual_alignment_prox_moon.py \
        --clients '"$CLIENTS"' \
        --num_rounds '"$NUM_ROUNDS"' \
        --batch_size '"$BATCH_SIZE"' \
        --img_size '"$IMG_SIZE"' \
        --num_workers '"$NUM_WORKERS"' \
        --csv_path '"$CSV_PATH"' \
        --starting_lr '"$LEARNING_RATE"' \
        --save_dir '"${BASE_PATH}"'/cnet_binary_personalization/exp_4/cnet_stat_proto_exp4_spectral/ \
        --gpu '"$DEVICE_GPU0"' \
        --alignment_type spectral \
        --alignment_weight 0.1 \
        --cosine_start_round 1000 \
        --fed_method fedprox \
        --amp \
        --amp_dtype float16
    echo "=========================================="
    echo "Stage 3/4: binary score (agg net binary)"
    echo "Time: $(date)"
    echo "=========================================="
    python train_federated_agg_net_with_statistical_prototypes_client_specific_dual.py \
        --clients '"$CLIENTS"' \
        --num_rounds '"$NUM_ROUNDS"' \
        --batch_size '"$BATCH_SIZE"' \
        --img_size '"$IMG_SIZE"' \
        --num_workers '"$NUM_WORKERS"' \
        --csv_path '"$CSV_PATH"' \
        --learning_rate '"$LEARNING_RATE"' \
        --save_dir '"${BASE_PATH}"'/agg_net_binary_personalization/exp_4/spectral/ \
        --device '"$DEVICE_GPU1"' \
        --alignment_type spectral \
        --alignment_weight 0.1 \
        --cosine_start_round 1000 \
        --fed_method fedprox \
        --task binary \
        --bin_pretrained_dir '"${BASE_PATH}"'/cnet_binary_personalization/exp_4/cnet_stat_proto_exp4_spectral/ \
        --bin_modality_pretrained_dir '"${BASE_PATH}"'/cnet_binary_personalization/exp_4/cnet_stat_proto_exp4_spectral/ \
        --amp \
        --amp_dtype float16
' >> ${LOG_DIR}/agg_net_ep_new.log 2>&1 &
