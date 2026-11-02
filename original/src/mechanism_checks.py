"""Numerical checks of estimator/derivative claims. Synthetic examples, never LES results."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def central_difference(n, spacing):
    d = np.zeros((n - 2, n))
    j = np.arange(n - 2)
    d[j, j] = -1 / (2 * spacing)
    d[j, j + 2] = 1 / (2 * spacing)
    return d


def main():
    y = np.linspace(-1.5, 1.5, 121)
    h = y[1] - y[0]
    d = central_difference(len(y), h)
    sigma1, sigma2, amplitude1, sensor_y = .4, .6, .6, 1.
    deficit_at_sensor = amplitude1 * np.exp(-sensor_y**2 / (2 * sigma1**2))
    amplitude2 = deficit_at_sensor / np.exp(-sensor_y**2 / (2 * sigma2**2))
    u1 = 1 - amplitude1 * np.exp(-y**2 / (2 * sigma1**2))
    u2 = 1 - amplitude2 * np.exp(-y**2 / (2 * sigma2**2))
    targets = np.stack([u1, u2])
    mean = targets.mean(0)
    obs = np.array([np.argmin(abs(y - a)) for a in [-sensor_y, sensor_y]])
    assert np.allclose(u1[obs], u2[obs], atol=1e-12)
    rows = []
    predictions = []
    # A low-degree decoder intentionally restricts capacity; it is not a wind solver.
    basis = np.polynomial.legendre.legvander(y / 1.5, 4)
    for weight in [0., .001, .01, .1, 1.]:
        w = np.eye(len(y)) / len(y) + weight * (d.T @ d) / len(d)
        optimum = np.linalg.solve(w, w @ mean)
        limited = basis @ np.linalg.solve(basis.T @ w @ basis, basis.T @ w @ mean)
        predictions.append(limited)
        rows.append({
            "lambda": weight,
            "unrestricted_max_difference_from_conditional_mean": float(abs(optimum - mean).max()),
            "unrestricted_expected_velocity_rmse": float(np.sqrt(np.mean((targets - optimum)**2))),
            "limited_expected_velocity_rmse": float(np.sqrt(np.mean((targets - limited)**2))),
            "limited_expected_gradient_rmse": float(np.sqrt(np.mean(((targets - limited) @ d.T)**2))),
        })
    checkerboard = (-1.) ** np.arange(len(y))
    rng = np.random.default_rng(20261031)
    noise_sigma = .05
    noise = rng.normal(0, noise_sigma, size=(10_000, len(y)))
    noise_variance = float(np.var(noise @ d.T))
    info = {
        "data_kind": "analytical Gaussian-profile counterexample; not CFD, LES, or field measurements",
        "sensor_coordinates": y[obs].tolist(),
        "sensor_values_profile1": u1[obs].tolist(), "sensor_values_profile2": u2[obs].tolist(),
        "center_velocity_profile1": float(u1[len(y)//2]),
        "center_velocity_profile2": float(u2[len(y)//2]),
        "physical_units": "dimensionless synthetic coordinates and velocities",
        "gaussian_widths": [sigma1, sigma2], "amplitudes": [amplitude1, amplitude2],
        "weighted_loss_checks": rows,
        "central_difference_checkerboard_max_abs": float(abs(d @ checkerboard).max()),
        "forward_difference_checkerboard_rms": float(np.sqrt(np.mean((np.diff(checkerboard)/h)**2))),
        "white_noise_derivative_variance_observed": noise_variance,
        "white_noise_derivative_variance_theory": noise_sigma**2 / (2*h*h),
        "conclusion_scope": "Checks algebra and discrete operators only; no claim about LES reconstruction performance.",
    }
    assert max(row["unrestricted_max_difference_from_conditional_mean"] for row in rows) < 1e-8
    assert info["central_difference_checkerboard_max_abs"] < 1e-12
    assert abs(noise_variance / info["white_noise_derivative_variance_theory"] - 1) < .02
    (ROOT / "results/mechanism_checks.json").write_text(json.dumps(info, indent=2))
    np.savez_compressed(ROOT / "results/mechanism_arrays.npz", y=y, u1=u1, u2=u2,
                        mean=mean, limited_predictions=np.asarray(predictions), obs=obs)
    print(json.dumps(info, indent=2))

    from plotting import lines
    lines(ROOT / "results/mechanism_checks",
          [("Possible profile A",y,u1),("Possible profile B",y,u2),("Conditional mean",y,mean)],
          "Identical sparse observations do not identify the full profile", "y*", "u*",
          "Analytical counterexample, not LES: both profiles have u*=0.973638 at y*=+/-1")


if __name__ == "__main__":
    main()
