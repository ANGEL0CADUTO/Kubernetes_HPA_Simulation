import math
import numpy as np
from scipy.stats import t, chi2



# --- TEST DI INDIPENDENZA (Ljung–Box) ---
def ljung_box_test(x, h=10):
    """
    Ljung–Box test per autocorrelazione fino al lag h.
    Restituisce p-value: se p > 0.05, non si rifiuta l’ipotesi di indipendenza.
    """
    n = len(x)
    if n < h + 2:
        return None
    x = np.asarray(x) - np.mean(x)

    # autocorrelazioni fino a h
    acf = []
    for lag in range(1, h + 1):
        num = np.dot(x[:-lag], x[lag:])
        den = np.dot(x, x)
        acf.append(num / den if den > 0 else 0)

    Q = n * (n + 2) * sum((acf[k - 1] ** 2) / (n - k) for k in range(1, h + 1))
    pval = 1 - chi2.cdf(Q, df=h)
    return pval

# --- FUNZIONE PER CALCOLARE B ---
def compute_batch_size(data, k=64, threshold=0.2):
    n = len(data)
    b = n // k
    while b > 1:
        batches = np.array([
            np.mean(data[i*b:(i+1)*b]) for i in range(k)
        ])
        batches_mean = np.mean(batches)
        num = np.sum((batches[:-1] - batches_mean) * (batches[1:] - batches_mean))
        den = np.sum((batches - batches_mean)**2)
        rho1 = num / den if den > 0 else 0
        if abs(rho1) < threshold:
            return b, rho1
        b += 1
        k = n // b
    return b, rho1

# --- FUNZIONE BATCH MEANS CON WELFORD ---


def batch_means(data, b, k, confidence=0.95):
    from welford import Welford

    n = b * k
    data = np.asarray(data[:n])

    batch_stats = Welford()
    for j in range(k):
        batch = data[j*b:(j+1)*b]
        w = Welford()
        w.add_all(batch)
        batch_stats.add(w.mean)

    mean = batch_stats.mean
    s2 = batch_stats.var_s
    dof = k - 1
    tval = t.ppf((1 + confidence)/2, dof)
    half_width = tval * math.sqrt(s2 / k)

    return {
        "mean": mean,
        "ci": (mean - half_width, mean + half_width),
        "half_width": half_width,
        "confidence_level": confidence,
        "num_batches": k,
        "batch_size": b
    }
