import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.environ.get("ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV = os.environ.get("CSV", f"{ROOT}/data/all_data_cleaned_with_splits_60_20_20_full.csv")
RESDIR = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/seg_map_refine/exp1/4_3_2_1_/map_refine/test_results"

df = pd.read_csv(CSV)
df['img_name'] = df['image_path'].apply(lambda p: str(p).split('/')[-1])
lab = dict(zip(df['img_name'], df['label']))

clients = [1, 2, 3, 4]
overall = []
for c in clients:
    f = os.path.join(RESDIR, f"test_result_client_{c}.csv")
    if not os.path.exists(f):
        print(f"missing {f}"); continue
    t = pd.read_csv(f)
    t.columns = [x.strip() for x in t.columns]
    t['name'] = t['Img Name'].astype(str).str.strip()
    t['label'] = t['name'].map(lab)
    t = t.dropna(subset=['label'])
    pos = t[t['label'] == 1]; neg = t[t['label'] == 0]
    # negative slices: Dice==1 means correctly predicted empty
    spec = (neg['Dice'] >= 0.999).mean() if len(neg) else float('nan')
    fp_slices = (neg['Dice'] < 0.999).sum()
    pos_dice = pos['Dice'].mean() if len(pos) else float('nan')
    pos_miss = (pos['Dice'] < 0.1).mean() if len(pos) else float('nan')
    print(f"client {c}: overall={t['Dice'].mean():.4f}  | n_pos={len(pos)} pos_Dice={pos_dice:.4f} "
          f"pos_total_miss={pos_miss:.3f} | n_neg={len(neg)} neg_specificity={spec:.3f} fp_slices={fp_slices}")
    overall.append(t['Dice'].mean())
print(f"\nMEAN overall test Dice across clients: {np.mean(overall):.4f}")
