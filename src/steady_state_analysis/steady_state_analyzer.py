import os
from enum import Enum

import numpy as np
from scipy.stats import t

from src.utils.acs import batch_means, compute_batch_size, ljung_box_test


class SimulationMode(Enum):
    TERMINATING = "terminating"
    STEADY_STATE = "steady_state"


class SteadyStateAnalyzer:
    """
    Classe per l'analisi dello stato stazionario (steady-state) del tempo di risposta.
    Implementa il campionamento per accelerare l'analisi.
    """
    def __init__(self, metrics, config):
        """
        Inizializza l'analizzatore con le metriche della simulazione e la configurazione.

        Args:
            metrics: Istanza della classe Metrics o MetricsWithPriority contenente i dati della simulazione.
            config: Oggetto di configurazione con parametri come CONFIDENCE_LEVEL, WARMUP_METHOD, ecc.
        """
        self.metrics = metrics
        self.config = config
        self.alpha = 1 - getattr(config, "CONFIDENCE_LEVEL", 0.95)
        self.warmup_method = getattr(config, "WARMUP_METHOD", "MSER5")
        self.confidence_level = getattr(config, "CONFIDENCE_LEVEL", 0.95)


    def extract_response_times_values(self) -> list[float]:
        """
        [MODIFICA] Estrae i valori dei tempi di risposta, applicando un campionamento
         per ridurre il volume dei dati. Prende un campione ogni due.

        Returns:
            list[float]: Lista campionata dei valori dei tempi di risposta.
        """
        all_responses = self.metrics.get_all_response_times_with_timestamps()
        # Applica il campionamento (thinning) prendendo un elemento ogni due
        sampled_responses = all_responses[::2]
        return [resp_time for _, resp_time in sampled_responses]

    def extract_full_response_data(self) -> list[tuple[float, float]]:
        """
        Estrae sia i timestamp che i valori dei tempi di risposta, applicando
        un campionamento coerente.

        Returns:
            list[tuple[float, float]]: Lista campionata di tuple (timestamp, tempo_risposta).
        """
        all_responses = self.metrics.get_all_response_times_with_timestamps()
        # Applica il campionamento (thinning) prendendo un elemento ogni due
        return all_responses[::2]


    def estimate_warmup(self, values_for_warmup_analysis: list[float], full_data_with_timestamps: list[tuple[float, float]]) -> float:
        """
        Stima la durata del warm-up in SECONDI. La logica interna non cambia,
        ma opererà su dati già campionati, rendendo il processo molto più veloce.
        """
        if not values_for_warmup_analysis or len(values_for_warmup_analysis) < 2:
            print("  DEBUG(Analyzer): Dati (campionati) insufficienti per stimare il warm-up. Ritorno 0s.")
            return 0.0

        warmup_index = 0
        if self.warmup_method == "WELCH":
            warmup_index = self._welch(np.array(values_for_warmup_analysis))
        elif self.warmup_method == "MSER5":
            warmup_index = self._mser5(np.array(values_for_warmup_analysis))
        else:
            warmup_index = 0

        if warmup_index == 0:
            return 0.0
        elif warmup_index >= len(full_data_with_timestamps):
            # Se l'indice è fuori range, limita il warmup per avere dati a regime
            # NOTA: La durata della simulazione potrebbe non essere più nel config se semplifichiamo,
            # quindi usiamo una stima basata sull'ultimo timestamp.
            max_time = full_data_with_timestamps[-1][0]
            print(f"  WARNING(Analyzer): Warmup_index {warmup_index} fuori range. Assegno un warmup conservativo.")
            return max_time * 0.8
        else:
            estimated_warmup_time = full_data_with_timestamps[warmup_index][0]

        # Limite di sicurezza
        max_time = full_data_with_timestamps[-1][0]
        if estimated_warmup_time >= max_time * 0.95:
            estimated_warmup_time = max_time * 0.8
            print(f"  WARNING(Analyzer): Tempo di warm-up stimato troppo lungo. Ridotto a {estimated_warmup_time:.2f}s.")

        print(f"  DEBUG(Analyzer): Warm-up stimato a {estimated_warmup_time:.2f}s su dati campionati.")
        return estimated_warmup_time

    def _welch(self, x: np.ndarray) -> int:
        n = len(x)
        if n < 50: return 0
        window = max(5, int(min(50, n) // 10))
        if window < 1 or n < window * 2: return 0
        smoothed = np.convolve(x, np.ones(window)/window, mode='valid')
        if len(smoothed) < 2: return 0
        std_smoothed = np.std(smoothed)
        if std_smoothed == 0: return 0
        diffs = np.abs(np.diff(smoothed))
        threshold_welch = std_smoothed / 10.0
        valid_cutoffs = np.where(diffs < threshold_welch)[0]
        cutoff = valid_cutoffs[0] if len(valid_cutoffs) > 0 else 0
        return cutoff if cutoff > 0 else 0

    def _mser5(self, x: np.ndarray) -> int:
        n = len(x)
        if n < 200: return 0
        window = max(5, n // 20)
        if window < 1 or n < window * 2: return 0
        best_t0, best_var = 0, float("inf")
        for t0 in range(0, n - window):
            resid = x[t0:]
            if len(resid) < window: break
            convolved_resid = np.convolve(resid, np.ones(window)/window, mode="valid")
            if len(convolved_resid) < 2: continue
            v = np.var(convolved_resid, ddof=1)
            if v < best_var:
                best_var, best_t0 = v, t0
        if best_var == float("inf"): return 0
        return best_t0

    def steady_state_analysis(self, values: list[float]):
        """
        [SEMPLIFICATO] Esegue l'analisi Batch Means. Ora è più veloce
        perché opera su dati pre-campionati.
        """
        if not values or len(values) < self.config.BATCH_K:
            print(f"  DEBUG(Analyzer): Dati (campionati) insufficienti ({len(values)}) per analisi Batch Means.")
            return None

        # Nota: L'efficienza di questa chiamata dipende criticamente dall'implementazione
        # di compute_batch_size in acs.py. Si assume l'uso della versione robusta.
        b, k, _ = compute_batch_size(
            data=values,
            k_initial_target=256,
            threshold=self.config.BATCH_THRESHOLD
        )

        if b is None or k is None:
            print("  WARNING(Analyzer): compute_batch_size non ha trovato una configurazione (b, k) valida.")
            return None

        results = batch_means(values, b, k, self.confidence_level)

        if results is None:
            print("  WARNING(Analyzer): batch_means ha restituito None.")
            return None

        # Aggiungiamo dettagli diagnostici per la stampa
        batch_means_for_test = [np.mean(values[i*b:(i+1)*b]) for i in range(k)]
        lags = min(10, k - 1) if k > 1 else 0
        pval = ljung_box_test(batch_means_for_test, h=lags) if lags > 0 else None
        results['ljung_box_pvalue'] = pval
        results['independence_ok'] = (pval is not None) and (pval > self.alpha)

        return results

    def print_ci_results(self, results: dict | None, metric_name: str):
        """
        Stampa i risultati dell'analisi dell'intervallo di confidenza.
        """
        if not results:
            print(f"Nessun risultato valido per '{metric_name}'")
            return
        print(f"\n--- Risultati Analisi Batch Means per '{metric_name}' ---")
        print(f"  - Media: {results['mean']:.6f}")
        print(f"  - CI {results['confidence_level']:.0%}: ({results['ci'][0]:.6f}, {results['ci'][1]:.6f})")
        print(f"  - Semi-ampiezza CI: {results['half_width']:.6f}")
        print(f"  - Batch size: {results.get('batch_size','N/A')}, #Batch: {results.get('num_batches','N/A')}")
        p = results.get('ljung_box_pvalue')
        if p is not None:
            print(f"  - Ljung–Box p-value: {p:.4f} ({'OK' if results['independence_ok'] else 'ATTENZIONE: Correlazione residua'})")
        print(f"----------------------------------------------------")