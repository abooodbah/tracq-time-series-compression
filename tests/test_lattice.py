"""Tests for the TRACQ-2 lattice codec (tracq/lattice.py)."""

import numpy as np
import pytest

from tracq import lattice


def _mixed_data(n_vars=6, n_time=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_time)
    rows = [
        1000 + np.cumsum(rng.normal(0, 1.0, n_time)),          # random walk
        50 * np.sin(2 * np.pi * t / 200) + rng.normal(0, 0.5, n_time),  # periodic
        np.full(n_time, 42.0),                                  # constant
        rng.laplace(0, 2.0, n_time),                            # zero-crossing
        1e-3 * np.cumsum(rng.normal(0, 1, n_time)),             # tiny scale
        1e4 + 1e3 * np.sin(2 * np.pi * t / 500) + rng.normal(0, 20, n_time),  # large scale
    ]
    return np.array(rows[:n_vars])


@pytest.mark.parametrize("predictors", ["p1", "bank"])
@pytest.mark.parametrize("eps", [1e-2, 1e-3, 1e-4])
def test_abs_mode_error_bound(predictors, eps):
    data = _mixed_data()
    blob, grid, header = lattice.encode(data, eps=eps, mode="abs", predictors=predictors)
    recon, _ = lattice.decode(blob)
    rng = data.max(axis=1) - data.min(axis=1)
    rng = np.where(rng <= 0, np.maximum(np.abs(data[:, 0]), 1.0), rng)
    err = np.abs(recon - data)
    assert recon.shape == data.shape
    # guaranteed pointwise bound: eps * per-variable range (+ float slack)
    assert (err <= eps * rng[:, None] * (1 + 1e-9) + 1e-12).all()


def test_rel_mode_error_bound():
    data = _mixed_data()
    eps = 1e-3
    blob, grid, header = lattice.encode(data, eps=eps, mode="rel")
    recon, _ = lattice.decode(blob)
    s = np.asarray(header["s"], dtype=float)[np.argsort(header["order"])]
    # mixed bound: eps relative for |x| >> s, ~eps*s absolute near zero
    denom = np.sqrt(data ** 2 + s[:, None] ** 2)
    rel_err = np.abs(recon - data) / denom
    assert rel_err.max() <= eps * 1.05


def test_no_drift_long_horizon():
    rng = np.random.default_rng(1)
    n_time = 200_000
    data = (1000 + np.cumsum(rng.normal(0, 1, n_time)))[None, :]
    eps = 1e-4
    blob, _, _ = lattice.encode(data, eps=eps, mode="abs", predictors="p1")
    recon, _ = lattice.decode(blob)
    err = np.abs(recon - data)
    bound = eps * (data.max() - data.min())
    # error at the END must satisfy the same bound as the start: no drift
    assert err[:, -100:].max() <= bound
    assert err.max() <= bound


def test_escapes_exact():
    rng = np.random.default_rng(2)
    data = np.cumsum(rng.normal(0, 1, 2000))[None, :]
    data[0, 500] += 1e6  # violent spike -> residual far outside int8
    data[0, 1500] -= 5e5
    blob, grid, header = lattice.encode(data, eps=1e-4, mode="abs", predictors="p1")
    recon, _ = lattice.decode(blob)
    rngv = data.max() - data.min()
    assert np.abs(recon - data).max() <= 1e-4 * rngv * (1 + 1e-9)
    assert (grid == lattice.ESCAPE).sum() >= 2


def test_zero_crossing_and_constant_channels():
    rng = np.random.default_rng(3)
    data = np.array([
        rng.normal(0, 1, 1000),            # crosses zero constantly
        np.zeros(1000),                    # all zeros
        np.full(1000, -7.5),               # negative constant
    ])
    for mode in ("abs", "rel"):
        blob, _, _ = lattice.encode(data, eps=1e-3, mode=mode)
        recon, _ = lattice.decode(blob)
        assert np.isfinite(recon).all()
        assert np.abs(recon[1]).max() <= 1e-6
        assert np.abs(recon[2] + 7.5).max() <= 1e-2


def test_seasonal_predictor_selected_and_exact():
    t = np.arange(4000)
    row = 100 + 10 * np.sin(2 * np.pi * t / 167)  # strong period 167
    data = np.vstack([row, np.random.default_rng(4).normal(0, 1, 4000)])
    blob, _, header = lattice.encode(data, eps=1e-3, mode="abs", predictors="bank")
    recon, _ = lattice.decode(blob)
    rngv = (data.max(axis=1) - data.min(axis=1))[:, None]
    assert (np.abs(recon - data) <= 1e-3 * rngv * (1 + 1e-9)).all()


def test_deadzone_bound_and_ratio():
    rng = np.random.default_rng(5)
    data = (100 + np.cumsum(rng.normal(0, 0.01, 5000)))[None, :]
    eps = 1e-3
    b0, _, _ = lattice.encode(data, eps=eps, mode="abs", predictors="p1", tau=0.0)
    b1, _, _ = lattice.encode(data, eps=eps, mode="abs", predictors="p1", tau=1.5)
    r0, _ = lattice.decode(b0)
    r1, _ = lattice.decode(b1)
    rngv = data.max() - data.min()
    q = 2 * eps * rngv
    assert np.abs(r0 - data).max() <= q / 2 * (1 + 1e-9)
    assert np.abs(r1 - data).max() <= 1.5 * q * (1 + 1e-9)  # tau*q bound
    assert len(b1) <= len(b0)  # dead-zone must not cost bytes


def test_nan_inf_handling():
    data = _mixed_data(n_vars=2, n_time=500)
    data[0, 10:20] = np.nan
    data[1, 100] = np.inf
    blob, _, _ = lattice.encode(data, eps=1e-3, mode="abs")
    recon, _ = lattice.decode(blob)
    assert np.isfinite(recon).all()


def test_view_png_roundtrip(tmp_path):
    data = _mixed_data(n_vars=4, n_time=800)
    blob, grid, header = lattice.encode(data, eps=1e-3, mode="abs")
    p = str(tmp_path / "view.png")
    lattice.save_view_png(p, grid, header)
    from tracq.codec import ImageCodec
    arr, meta = ImageCodec.load_png(p)
    assert np.array_equal(arr, grid)
    assert meta["codec"] == "tracq2-lattice"


def test_single_timestep():
    data = np.array([[5.0], [7.0]])
    blob, _, _ = lattice.encode(data, eps=1e-3, mode="abs")
    recon, _ = lattice.decode(blob)
    assert recon.shape == (2, 1)
    assert np.abs(recon - data).max() < 1e-2
