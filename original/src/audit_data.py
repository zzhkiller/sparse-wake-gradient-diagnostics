"""Audit the author's actual LES subset; do not assume its coordinate origin."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from mat_v5 import loadmat

ROOT = Path(__file__).resolve().parents[1]


def load_data():
    first = loadmat(ROOT / "data/DTU_LES_data_wake_60s_part1.mat")
    second = loadmat(ROOT / "data/DTU_LES_data_wake_60s_part2.mat")
    x, y = first["x_LES"].ravel(), first["y_LES"].ravel()
    t = np.concatenate([first["t1"].ravel(), second["t2"].ravel()])
    u = np.concatenate([first["U_LES1"], second["U_LES2"]], axis=2).transpose(2, 0, 1)
    v = np.concatenate([first["V_LES1"], second["V_LES2"]], axis=2).transpose(2, 0, 1)
    assert u.shape == v.shape == (len(t), len(x), len(y)), "Unexpected MAT axes"
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert np.all(np.diff(t) > 0)
    return x, y, t, u, v, first, second


def main():
    x, y, t, u, v, first, second = load_data()
    info = {
        "source_kind": "author-provided DTU LES subset, not newly simulated data",
        "author_statement": "60 s wake condition; ambient TI=12.8%; non-yawed demonstration",
        "array_order": "time,x,y",
        "shape": list(u.shape), "time_range_s": [float(t[0]), float(t[-1])],
        "time_step_s": np.unique(np.round(np.diff(t), 9)).tolist(),
        "x_range_m": [float(x.min()), float(x.max())],
        "y_range_m": [float(y.min()), float(y.max())],
        "dx_m": np.unique(np.round(np.diff(x), 9)).tolist(),
        "dy_m": np.unique(np.round(np.diff(y), 9)).tolist(),
        "u_range_m_s": [float(u.min()), float(u.max())],
        "v_range_m_s": [float(v.min()), float(v.max())],
        "u_mean_m_s": float(u.mean()), "v_mean_m_s": float(v.mean()),
        "mat_variables": {name: list(value.shape) for name, value in first.items() if not name.startswith("__")},
        "mat_variables_part2": {name: list(value.shape) for name, value in second.items() if not name.startswith("__")},
        "coordinate_warning": "Inspect variables and the mean-field figure before assuming a turbine origin or changing wind direction.",
    }
    path = ROOT / "results/data_audit.json"
    path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    from plotting import heatmaps
    heatmaps(ROOT / "results/data_audit", [u.mean(0),u.std(0),np.gradient(u.mean(0),y,axis=1)],
             x/126,y/126,["Mean streamwise velocity","Temporal standard deviation","Mean transverse gradient"],
             ["m/s","m/s","1/s"],"Author LES subset: descriptive audit, not model performance")


if __name__ == "__main__":
    main()
