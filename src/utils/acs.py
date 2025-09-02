import math
import numpy as np

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
def batch_means(data, b, k):
    from welford import Welford   # oppure importa la tua classe direttamente

    n = b * k
    data = np.asarray(data[:n])

    batch_stats = Welford()

    for j in range(k):
        batch = data[j*b:(j+1)*b]
        w = Welford()
        w.add_all(batch)
        batch_stats.add(w.mean)

    mean = batch_stats.mean
    s = math.sqrt(batch_stats.var_s)
    ci95 = (
        mean - 1.96 * s / math.sqrt(k),
        mean + 1.96 * s / math.sqrt(k)
    )
    return mean, ci95
