import math
import numpy as np
from scipy.stats import t, chi2

from welford import Welford


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

# --- FUNZIONE PER CALCOLARE B la size ---
def compute_batch_size(data, k_initial_target, threshold):
    n = len(data)

    # Se i dati sono insufficienti per almeno 2 batch, non possiamo procedere
    if n < 2:
        return None, None, None # Restituisce None per b, k, rho1

    b = n // k_initial_target # Calcolo iniziale della dimensione del batch

    # Assicuriamo che b sia almeno 1 (un batch che contiene tutti i dati)
    if b == 0:
        b = 1 # Se b è 0, significa che k_initial_target è più grande di n, quindi b deve essere almeno 1.

    while True: # Ciclo continuo fino a trovare un b o raggiungere un limite
        if b < 1: # La dimensione del batch non può essere minore di 1
            return None, None, None

        current_k = n // b # Questo è il numero effettivo di batch per la dimensione `b` corrente

        # Dobbiamo avere almeno 2 batch per calcolare rho1 e fare il test di indipendenza
        if current_k < 2:
            return None, None, None # Non è possibile trovare un b, k valido

        # Estrai le medie dei batch
        batches = np.array([
            np.mean(data[i*b:(i+1)*b]) for i in range(current_k)
        ])

        batches_mean = np.mean(batches)

        # Calcolo di rho1 (richiede almeno 2 medie di batch)
        if len(batches) < 2:
            return None, None, None

        num = np.sum((batches[:-1] - batches_mean) * (batches[1:] - batches_mean))
        den = np.sum((batches - batches_mean)**2)
        rho1 = num / den if den > 0 else 0

        if abs(rho1) < threshold:
            # Trovato il b ottimale. Restituisci b, il k effettivo e rho1.
            return b, current_k, rho1

        # Incrementa b e prova con una dimensione del batch più grande
        b += 1
        # Evita un loop infinito se b continua a crescere oltre la lunghezza dei dati
        if b > n: # Se b diventa più grande di n, k diventerà 0 o 1, già gestito da current_k < 2
            return None, None, None # Nessuna dimensione del batch valida trovata

# --- FUNZIONE BATCH MEANS CON WELFORD ---
def batch_means(data, b, k, confidence=0.95):
    # 'from .welford import Welford' is now at the top of the file.

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