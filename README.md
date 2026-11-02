# Sparse wind turbine wake reconstruction with gradient supervision

Code and numerical outputs for **Gradient Supervision and Representation Limits in Sparse Wind Turbine Wake Reconstruction**.

This repository records a limited reconstruction study on a public 60 s LES excerpt. It includes the original experiments and the revision diagnostics. It does not establish a general performance advantage for gradient supervision or demonstrate edge-device deployment.

## Contents

- `original/src/`: reconstruction, numerical tests, data audit, and mechanism checks. Checkpoints are selected by normalized validation velocity-plus-gradient MSE.
- `revision/src/`: the same reconstruction trajectories with velocity-only checkpoint selection, plus training-only POD rank and chronological-field-change diagnostics.
- `original/results/` and `revision/results/`: configuration, run-level CSV, summaries, logs, and nine selected parameter checkpoints per selection rule.
- `sources/`: public source metadata and the pinned data download manifest. No third-party paper full text or raw LES files is redistributed.
- `SHA256.json`: checksums of the archived files. It does not contain its own checksum.

## Reproduce

Use Python 3.12. Recorded versions are pinned in `requirements.txt`. From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_archive.py
python download_data.py
python -m unittest discover -s original/src -p 'test_numerics.py' -v
python -m unittest discover -s revision/src -p 'test_numerics.py' -v
python original/src/pilot.py
python original/src/diagnose_pilot.py
python original/src/mechanism_checks.py
python revision/src/pilot.py
python revision/src/reviewer_diagnostics.py
```

`download_data.py` downloads about 131 MB, checks the two recorded file sizes and SHA256 hashes, and installs copies into `original/data/` and `revision/data/`. To work offline, place the files from `sources/data_download_manifest.json` in those directories after independently checking their hashes. Rerunning experiments overwrites generated result files; keep a fresh checkout for comparison with the archived outputs. Exact timings depend on hardware; floating-point differences may occur across NumPy/BLAS environments.

The checkpoints contain the small MLP parameter arrays. The fixed POD decoder and normalization are rebuilt from the recorded training frames by `pilot.py`; they are not additional learned checkpoint arrays.

## Data provenance

The original DTU dataset is registered at [DOI 10.11583/DTU.10259936.v1](https://doi.org/10.11583/DTU.10259936.v1). The two MAT excerpt files are fetched from [Feng Guo's PINN-Lidar-WFR repository](https://github.com/fengguoFUAS/PINN-Lidar-WFR/tree/dda521625a96dce48f138f53b512fdf82db0a266), pinned to commit `dda521625a96dce48f138f53b512fdf82db0a266`. Data provenance, authorship, and applicable source terms remain with the original providers. No author-provided PINN implementation was executed in this study.

## Recorded scope and findings

The crop spans x/D = 2–6 and y/D = −1.5–1.5 on a 121 by 91 grid. Sixteen fixed sensors are used. The chronological train, validation, and test frame ranges are 0–119, 140–179, and 200–249 (zero based), with intervening guard frames. These short intervals are not demonstrated to be statistically independent.

- IDW test velocity RMSE: 1.40334 m/s. POD-MLP means for gradient weights 0, 0.1, and 1 are 2.92103, 2.89729, and 2.91455 m/s across seeds 11, 22, and 33.
- Changing the validation selection criterion selected identical parameter arrays for all nine recorded runs. This is an observed sensitivity result for these trajectories, not a general equivalence of the criteria.
- Training-only rank-16 and rank-119 POD test projections have RMSE 1.77868 and 1.69778 m/s. Projection uses test truth and is a representation diagnostic, not a deployable reconstruction baseline or a test-tuned rank choice.
- For the rank-16 projection residual, unresolved interval-mean change and fluctuations contribute 2.83395 and 0.32976 (m/s)^2. The mean-change component accounts for 89.58 percent. This is descriptive evidence from one short record, not an aerodynamic causal or stationarity test.
- Persistence uses the previous full field as extra information and is not a fair same-input baseline.

Four numerical unit tests cover derivative axes and units, backpropagation by finite differences, MAT array shape and ordering, and quadratic-risk invariance. Reported experimental results are retained even when they favor the simple interpolation baseline.

## 中文说明

本仓库用于复现稀疏观测风机尾流重建及梯度监督分析，包含原实验与返修新增诊断。先校验归档，再下载固定版本数据并运行以上命令。结论仅适用于本次短时间记录和指定模型；未验证实测风场、多工况泛化或边缘设备性能。
