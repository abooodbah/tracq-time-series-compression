import numpy as np


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    return float(np.mean(np.abs(a - b)))


def mape(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    denom = np.maximum(np.abs(a), eps)
    return float(np.mean(np.abs((a - b) / denom)))


def smape(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    denom = np.maximum((np.abs(a) + np.abs(b)) / 2.0, eps)
    return float(np.mean(np.abs(a - b) / denom))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("shapes must match")
    if a.size == 0:
        return 0.0
    a_mean = a.mean()
    b_mean = b.mean()
    num = np.sum((a - a_mean) * (b - b_mean))
    den = np.sqrt(np.sum((a - a_mean) ** 2) * np.sum((b - b_mean) ** 2))
    return float(num / den) if den != 0 else 0.0
