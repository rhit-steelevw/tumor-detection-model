# Tumor Detection Model

Predicts whether a whole-body PET/CT scan contains a tumor (lesion present / absent). Scans from the autoPET dataset (FDG and PSMA tracers) are passed through a pretrained 3D encoder, each scan is reduced to a single 640-dimensional feature vector, and a small fully connected classifier is trained on those vectors.

The encoder comes from the autoPET III winning solution by Rokuss et al. (Team LesionTracer, DKFZ). We use it for transfer learning: the network is fine-tuned on our training split, then only its encoder is kept for feature extraction.

## Pipeline

```
autoPET PET/CT scans (CT = _0000, PET = _0001)
        │  prepare_datasets.py   patient-level 60/20/20 split, test set held out
        │  nnU-Net preprocessing + organ-label merge + copy_plans.py
        ▼
autoPET III ResEncL network (MultiTalent pretrained init → fine-tuned, fold 0)
        │  extract_features.py   encoder only, tiled, mean + max pooled
        │  merge_features.py     combine 20 SLURM chunks into one file
        ▼
features/*.npz   (1,611 scans × 640 features)
        │  classifier.ipynb
        ▼
tumor present / absent
```

## Repository layout

| Path | Contents |
|---|---|
| `classifier.ipynb` | Baselines and the fully connected classifier trained on encoder features, with validation metrics |
| `encoder/` | Data preparation, preprocessing, training, and feature-extraction scripts plus SLURM job files (`jobs/`). See `encoder/README.md` |
| `autopet/` | Copy of the scripts as run on the cluster, with the split files |
| `features/` | Extracted encoder features (`encoder_features.npz`, `encoder_features_ep760.npz`, `encoder_features_best.npz`) |
| `data/` | Split files (`split_60_20_20.json`, `splits_final.json`), FDG/PSMA metadata CSVs, `dataset.json`, dataset fingerprint |
| `training-guides/` | Plan for a tumor present/absent classifier |
| `FINE_TUNING_GUIDE.md` | Fine-tuning the autoPET III ResEncL model for lesion segmentation |
| `autopet-3-submission/` | Git submodule: [MIC-DKFZ/autopet-3-submission](https://github.com/MIC-DKFZ/autopet-3-submission) |

## Data

- **Source:** autoPET PET/CT dataset (FDG and PSMA), in nnU-Net format as `Dataset200_autoPET_lesions`.
- **Cases:** 1,611 scans that have an organ mask from the autoPET III repo.
- **Split:** patient-level 60/20/20, done separately within each tracer (seed 42): **980 train, 315 val, 316 test**. Test scans are moved out of the training folder before preprocessing, so the encoder never trains on them. The split is in `data/split_60_20_20.json`.
- **Label:** a scan is "tumor" if its lesion mask has any foreground voxels.

The raw data, preprocessed arrays, and model weights (~1.4 TB) stay on the cluster and are not tracked in git.

## Feature files

Each `.npz` in `features/` has these keys:

| Key | Shape | Meaning |
|---|---|---|
| `features` | (1611, 640) | Mean- and max-pooled deepest encoder features |
| `case_ids` | (1611,) | Scan ID |
| `split` | (1611,) | `train` / `val` / `test` |
| `tracer` | (1611,) | `fdg` / `psma` |
| `lesion` | (1611,) | 1 = tumor, 0 = no tumor |
| `metadata` | — | Checkpoint and code hashes (newer files only) |

```python
import numpy as np
d = np.load("features/encoder_features_ep760.npz")
X, y, split = d["features"], d["lesion"], d["split"]
X_train, y_train = X[split == "train"], y[split == "train"]
```

## Reproducing on the cluster

Scripts assume these environment variables: `AP` (working folder), `AP3` (autopet-3-submission clone), `nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`.

1. **Environment.** Clone `autopet-3-submission` and `pip install -e .` in a Python 3.10 conda env. For newer PyTorch, remove the trailing `, False` in `super().__init__` in `nnunetv2/training/lr_scheduler/polylr.py` and set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the sbatch files already do this).
2. **Data.** Place the autoPET data in `$nnUNet_raw/Dataset200_autoPET_lesions/`, then run `python encoder/prepare_datasets.py` to create the split and the organ dataset (`Dataset201_autoPET_organs`) from the organ masks shipped with autoPET III.
3. **Preprocess.** `sbatch encoder/jobs/preprocess.sbatch` runs nnU-Net preprocessing on both datasets, merges the organ labels, and runs `copy_plans.py` to adapt the MultiTalent ResEncL plans to our dataset.
4. **Fine-tune.** Download the pretrained init (Dataset619_nativemultistem, [Zenodo 13753413](https://zenodo.org/records/13753413)) and run `sbatch encoder/jobs/train.sbatch`. It resumes from the last checkpoint when resubmitted.
5. **Extract features.** Freeze a checkpoint and run the 20-task array job, then merge. Full commands are in `encoder/README.md`:
   ```bash
   sbatch --export=ALL encoder/jobs/encode.sbatch
   python merge_features.py "$FEATURE_OUT" 20 --split split_60_20_20.json
   ```
6. **Classify.** Open `classifier.ipynb` and point it at the new feature file.

## Results (validation set, 315 scans)

| Model | Accuracy | Balanced accuracy |
|---|---:|---:|
| Classifier on earlier features | 72.06% | 71.51% |
| Classifier on epoch-760 features (16 hidden units, threshold 0.60) | 85.40% | 84.91% |

The test set has not been used for model selection. PSMA scans have been harder than FDG, so check per-tracer results alongside the overall numbers.

## References

Rokuss, M., Kovacs, B., Kirchhoff, Y., Xiao, S., Ulrich, C., Maier-Hein, K. H., & Isensee, F. (2024). *From FDG to PSMA: A Hitchhiker's Guide to Multitracer, Multicenter Lesion Segmentation in PET/CT Imaging.* arXiv:2409.09478. Code: [MIC-DKFZ/autopet-3-submission](https://github.com/MIC-DKFZ/autopet-3-submission).

Isensee, F., et al. nnU-Net. [github.com/MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet).

## AI usage

We used an AI assistant (Claude, by Anthropic) to help get the PET/CT data into the encoder, following the autoPET III method from Rokuss et al. It read through the `autopet-3-submission` code and README and turned their steps into instructions and scripts for our SLURM cluster. This covered:

- Downloading the autoPET data and arranging it in the nnU-Net folder layout.
- Writing `prepare_datasets.py` to make the patient-level 60/20/20 split by tracer and hold out the test set before preprocessing.
- Using the organ masks shipped with autoPET III instead of rerunning TotalSegmentator, and merging them with the lesion dataset.
- Writing `copy_plans.py` to adapt the MultiTalent ResEncL plans from Rokuss et al. file to our dataset.
- Writing the preprocessing and training scripts for submitting SLURM jobs, including starting from the pretrained MultiTalent weights and resuming after time limits, plus the PyTorch compatibility fixes.
- Writing the chunked feature extraction (`extract_features.py`, `merge_features.py`, `feature_checks.py`, `encode.sbatch`). It splits the 1,611 scans into 20 jobs so none hits the time limit, keeps only the encoder, and checks that every chunk used the same checkpoint before merging.

The team ran all jobs, checked the outputs, and made the modeling decisions.
