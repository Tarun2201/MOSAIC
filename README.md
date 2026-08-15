# MOSAIC: Modality-agnostic Spectral Alignment for federated Image-level weakly supervised tumour segmentation under Client-specific missing modalities

Whole-tumor segmentation on multi-modal MRI, trained **federated** (clients never
share data) and **weakly supervised** (only a per-slice "tumor / no tumor" label).
Each client holds a *different subset* of modalities — that is the hard part.
Ground-truth masks are used only for validation and the final test score.

## Pipeline

| Stage | What it does | Driver |
|---|---|---|
| 1 | Federated binary classifier, then an aggregation net that fuses its multi-scale CAMs into one map | `run_all_exp.sh` |
| 2 | Thresholds those CAMs into binary pseudo-masks (one PNG per slice) + a manifest | `run_save_pseudo_labels.sh` |
| 3 | Federated segmentation on the pseudo-masks, with prototype consensus + EMA teacher| `federated_map_refine/run_map_v12.sh` |


### Data

Not in this repo — `data/*.sample.csv` document the schema. You need MRI slices
as PNGs, one sibling directory per modality:

```
<DATA_ROOT>/<case_id>/
  ├── flair/<case_id>_flair_0.png   ← the only path stored in the CSV
  ├── t1ce/ <case_id>_t1ce_0.png    ← derived by replacing "flair"
  ├── t2/   <case_id>_t2_0.png
  └── seg/  <case_id>_seg_0.png
```

The CSV stores **only the FLAIR path** — the other modalities and the mask are
derived by string substitution, so `flair` must appear in the path only where it
should be replaced. Filenames must end `_<slice_index>.png`: the 2.5D dataset
finds neighbours by incrementing that integer and 3D eval groups volumes by
stripping it. Slices are single-channel PNGs; the seg PNG is RGB and whole-tumor
is any non-zero pixel.

**Split CSV** — columns 0, 1, 2 are read *positionally*, so order matters:

| # | column | meaning |
|--:|---|---|
| 0 | `image_path` | FLAIR slice PNG |
| 1 | `mask_path` | GT seg PNG — read for val/test only, never a training target |
| 2 | `label` | 0/1 weak image-level label; the **only** training supervision |
| 3–5 | `necrosis`, `enhancing`, `edema` | subclass flags, only 1 when `label == 1` |
| 6 | `client_id` | must appear in the `clients` config map |
| 7 | `split` | `train` / `val` / `test`, by patient not by slice |

**`manifest.csv`** (Stage-2 output, read by name): `image_path`,
`pseudo_mask_path` (the 0/255 training target), `gt_mask_path` (reference only),
`label`, `split` (`train`/`val` only), `client_id`, `corrected_by_label` (forced
empty because `label == 0`), `forced_blank` (forced empty by the blank gate).

### Clients and modalities

A client is any group of cases sharing a `client_id`, with its own modality
subset. The mapping from `client_id` to modalities lives in the `config` dict at
the top of each script — edit it to match your data, then set `CLIENTS` to the
IDs you want to federate. Clients with a single modality, or missing the
modality the others share, are the cases the method is meant to help.

## Running

Paths come from the environment:

```bash
export CSV_PATH=/data/splits.csv         # split CSV
export BASE_PATH=/data/checkpoints       # Stage 1 weights
export CLIENTS=1,2,3,4
export PY=python
```

```bash
bash run_all_exp.sh                       # Stage 1 — backgrounded under nohup
tail -f nohups/agg_net_ep_new.log

bash run_save_pseudo_labels.sh            # Stage 2 — foreground
                                          # -> pseudo_labels_binary/ + manifest.csv

CSV_PATH=... MANIFEST=.../manifest.csv \
  bash federated_map_refine/run_map_v12.sh   # Stage 3 (run_and_test_v12.sh = train+test)
```


## Evaluation

```bash
$PY federated_map_refine/test_map_v4.py \
    --clients "$CLIENTS" --csv_path "$CSV_PATH" \
    --model_dir results/seg_map_refine/exp_v12_scratch/<clients>/map_refine_v12/ \
    --K 3 --tta --gate_tau 0.5

$PY federated_map_refine/analyze_errors.py <model_dir>/test_results
```


## Configuration

| Variable | Used by | Default |
|---|---|---|
| `CSV_PATH` | all | `./data/all_data_cleaned_with_splits_60_20_20_full.csv` |
| `MANIFEST` | Stage 3 | `./pseudo_labels_binary/manifest.csv` |
| `BASE_PATH` | Stages 1–2 | `./checkpoints` |
| `OUT_DIR` | Stage 2 | `./pseudo_labels_binary` |
| `SAVE_DIR` | Stage 3 | `./results/seg_map_refine/exp_v12_scratch/` |
| `CLIENTS` | all | `1,2,3,4` |
| `PY` | Stages 2–3 | `python` |
| `GPU` / `DEVICE_GPU0` | all | `0` |


