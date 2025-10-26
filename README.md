GA-SEG (gaseg)

This repository is a lightweight experimental skeleton for exploring Genetic Algorithm (GA) driven segmentation and localization for industrial defect detection. It isn't a full paper release yet — it's a starting point with safe defaults and scaffolding for the GA-SEG concept.

What I implemented (summary)
- Safer evaluation helpers: `src/eval/metrics.py` now contains `safe_image_auc()` and `evaluate_auroc()` to avoid crashes on degenerate label sets.
- PatchCore-style baseline: `src/baselines/patchcore.py` (SimplePatchCore) extracts ResNet feature maps, builds a patch memory bank and computes patch-wise anomaly scores via NearestNeighbors. I added optional memory subsampling (`max_patches`) to avoid OOM, and the anomaly maps are upsampled to 256x256 by default to match dataset transforms.
- Simple classical baseline: `src/baselines/otsu_baseline.py` provides a fast Otsu-threshold segmentation baseline for ablation / sanity checks.
- GA scaffold: `src/ga/ga_seg.py` includes a tiny `ThresholdGA` that evolves a single continuous threshold in [0,1] to maximize IoU on a validation set. This is a scaffold for building multi-gene chromosomes.
- Demo & orchestration: `src/run_experiments.py` was cleaned to run a PatchCore demo (`run_patchcore_demo`) with CLI usage. `evaluate_all.py` runs the demo across MVTec categories and writes `results_mvtec.csv` (present; currently empty placeholders).
- Visualization helper: `src/viz/overlay.py` overlays masks on images for qualitative checks.

Quick start (local)
1) Activate the repository virtualenv (provided `env`):

```powershell
. .\env\Scripts\Activate.ps1
```

2) (Optional) Install/update requirements if you are not using the provided `env`:

```powershell
python -m pip install --upgrade pip; python -m pip install -r requirements.txt
```

3) Prepare MVTec AD: place the MVTec dataset in `mvtec_anomaly_detection/`. Each category should follow the MVTec layout (`train/good`, `test/...`, `ground_truth/...`).

4) Run the demo for one category (example 'bottle'):

```powershell
python .\src\run_experiments.py --data_root mvtec_anomaly_detection --category bottle
```

5) Or run the full evaluation across categories (this script calls the demo per-category and writes `results_mvtec.csv`):

```powershell
python .\evaluate_all.py
```

Important notes about running
- Activate the `env` virtual environment to avoid unresolved-import warnings in the editor. The `env` folder shipped with the repo contains the packages used during development.
- `SimplePatchCore.build_memory()` may still need more robust coreset sampling for very large datasets — I added a `max_patches` cap (set to 20000 in the demo) to keep memory small for laptops.
- The `ga_seg.py` is a scaffold. It demonstrates how to plug a GA into a segmentation pipeline; it currently evolves a single threshold only.

Where to look in the code
- `src/baselines/patchcore.py` — SimplePatchCore implementation (feature extraction, memory bank, scoring).
- `src/datasets/mvtec_loader.py` — dataset loader; returns (img_tensor, mask, path).
- `src/eval/metrics.py` — metrics and helpers (safe wrappers).
- `src/ga/ga_seg.py` — minimal GA scaffold (ThresholdGA).
- `src/baselines/otsu_baseline.py` — simple Otsu baseline.
- `src/run_experiments.py` — demo CLI and main example.


