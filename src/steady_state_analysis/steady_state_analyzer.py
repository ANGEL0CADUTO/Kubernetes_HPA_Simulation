


import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

from src.utils.acs import batch_means, compute_batch_size, ljung_box_test


class SimulationMode(Enum):
    TERMINATING = "terminating"
    STEADY_STATE = "steady_state"


class SteadyStateAnalyzer:
    """
    Classe per analisi steady-state o terminating replicata.
    Compatibile con Metrics e MetricsWithPriority del main.py.
    """
    def __init__(self, metrics, config):
        self.metrics = metrics
        self.config = config
        self.mode = getattr(config, "SIMULATION_MODE", SimulationMode.STEADY_STATE)
        self.alpha = 1 - getattr(config, "CONFIDENCE_LEVEL", 0.95)
        self.rel_precision = getattr(config, "REL_PRECISION", None)
        self.abs_precision = getattr(config, "ABS_PRECISION", None)
        self.warmup_method = getattr(config, "WARMUP_METHOD", "MSER5")

    def extract_response_times_values(self) -> list[float]:
        """
        Estrae solo i valori dei tempi di risposta (senza timestamp)
        per l'analisi del warm-up.
        """
        all_responses = self.metrics.get_all_response_times_with_timestamps()
        return [resp_time for _, resp_time in all_responses]

    def extract_full_response_data(self) -> list[tuple[float, float]]:
        """
        Estrae sia i timestamp che i valori dei tempi di risposta.
        """
        return self.metrics.get_all_response_times_with_timestamps()


    def estimate_warmup(self, values_for_warmup_analysis: list[float], full_data_with_timestamps: list[tuple[float, float]]) -> float:
        """
        Stima la durata del warm-up in SECONDI.
        Args:
            values_for_warmup_analysis: Solo i valori (es. tempi di risposta) per l'analisi del warm-up.
            full_data_with_timestamps: I dati completi (timestamp, valore) per convertire l'indice in tempo.
        Returns:
            float: Il tempo di warm-up stimato in secondi.
        """
        if not values_for_warmup_analysis or len(values_for_warmup_analysis) < 2:
            print("  DEBUG(Analyzer): Dati insufficienti per stimare il warm-up, ritorno 0s.")
            return 0.0

        if self.warmup_method == "WELCH":
            warmup_index = self._welch(np.array(values_for_warmup_analysis))
        elif self.warmup_method == "MSER5":
            warmup_index = self._mser5(np.array(values_for_warmup_analysis))
        else: # Default a un warmup minimo se il metodo non è riconosciuto
            warmup_index = 0

        # Ora convertiamo l'indice in un tempo
        if warmup_index >= len(full_data_with_timestamps):

            estimated_warmup_time =self.config.STEADY_SIMULATION_TIME * 0.8
            print(f"  DEBUG(Analyzer): Warmup_index {warmup_index} fuori range ({len(full_data_with_timestamps)}). Assegno warmup di {estimated_warmup_time:.2f}s.")
        else:
            # Il tempo di warm-up è il timestamp di completamento all'indice stimato.
            # Se warmup_index è 0, significa che non c'è warm-up o è molto breve,
            # quindi il tempo di warm-up è il timestamp del primo evento.
            estimated_warmup_time = full_data_with_timestamps[warmup_index][0] if warmup_index > 0 else full_data_with_timestamps[0][0]


        # Per evitare che il warmup sia l'intera simulazione
        if estimated_warmup_time >= self.config.STEADY_SIMULATION_TIME* 0.95:
            estimated_warmup_time = self.config.STEADY_SIMULATION_TIME * 0.8 # Lascia almeno il 20% per lo steady-state
            print(f"  DEBUG(Analyzer): Warmup stimato troppo lungo. Ridotto a {estimated_warmup_time:.2f}s.")

        print(f"  DEBUG(Analyzer): Indice di warm-up stimato: {warmup_index}, Tempo di warm-up finale: {estimated_warmup_time:.2f}s.")
        return estimated_warmup_time

    def _welch(self, x: np.ndarray) -> int:

        window = max(5, int(min(50, len(x)) // 10))
        if window < 1 or len(x) < window * 2:
            return 0
        smoothed = np.convolve(x, np.ones(window)/window, mode='valid')
        if len(smoothed) < 2:
            return 0
        diffs = np.abs(np.diff(smoothed))
        if np.std(smoothed) == 0:
            return 0
        valid_cutoffs = np.where(diffs < np.std(smoothed)/100)[0]
        cutoff = valid_cutoffs[0] if len(valid_cutoffs) > 0 else 0
        return cutoff if cutoff > 0 else 0

    def _mser5(self, x: np.ndarray) -> int:

        n = len(x)
        if n < 200: # MSER5 ha bisogno di più dati
            return 0
        window = max(5, n // 20)
        if window < 1 or n < window * 2:
            return 0
        # ...
        mv = np.convolve(x, np.ones(window)/window, mode="valid")
        if len(mv) < 1:
            return 0

        best_t0, best_var = 0, float("inf")

        for t0 in range(0, n - window * 2):
            resid = x[t0:]

            if len(resid) < window*2:
                break

            convolved_resid = np.convolve(resid, np.ones(window)/window, mode="valid")
            if len(convolved_resid) < 2:
                continue

            v = np.var(convolved_resid, ddof=1)
            if v < best_var:
                best_var, best_t0 = v, t0
        return best_t0

    # --------------------------
    # Batch Means Analysis
    # --------------------------
    def steady_state_analysis(self, values, confidence=0.95, threshold=0.2):
        """Restituisce mean, ci95, batch info, Ljung-Box p-value"""
        b, k = compute_batch_size(values, threshold=threshold)
        if b is None or k is None:
            return None

        mean, ci95 = batch_means(values, b, k, confidence)
        batches = [np.mean(values[i*b:(i+1)*b]) for i in range(k)]
        pval = ljung_box_test(batches, h=min(10, k-1))
        return {
            "mean": mean,
            "ci": ci95,
            "half_width": (ci95[1]-ci95[0])/2,
            "batch_size": b,
            "num_batches": k,
            "ljung_box_pvalue": pval,
            "independence_ok": (pval is None) or (pval > 0.05),
            "confidence_level": confidence
        }

    # --------------------------
    # Terminating (repliche)
    # --------------------------
    def terminating_analysis(self, replications, confidence=0.95):
        n = len(replications)
        if n < 2:
            return None
        mean = np.mean(replications)
        s2 = np.var(replications, ddof=1)
        tval = t.ppf((1 + confidence)/2, df=n-1)
        hw = tval * np.sqrt(s2 / n)
        return {
            "mean": mean,
            "ci": (mean - hw, mean + hw),
            "half_width": hw,
            "confidence_level": confidence,
            "replications": n
        }

    # --------------------------
    # Plot CI
    # --------------------------
    def plot_confidence_interval(self, results, title, output_dir, filename):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.errorbar(x=[0], y=[results["mean"]], yerr=results["half_width"], fmt='o', color='b',
                    capsize=10, markersize=8, elinewidth=3, label='CI')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Valore Medio')
        ax.set_xticks([])
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        # Testo CI
        ci_text = (
            f"Media: {results['mean']:.3f}\n"
            f"CI al {results['confidence_level']:.0%}: [{results['ci'][0]:.3f}, {results['ci'][1]:.3f}]\n"
            f"Batch size: {results.get('batch_size','?')}, "
            f"#Batch: {results.get('num_batches','?')}\n"
            f"Ljung–Box p: {results.get('ljung_box_pvalue','N/A')}"
        )
        ax.text(0.05, 0.95, ci_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # --------------------------
    # Print risultati CI
    # --------------------------
    def print_ci_results(self, results, metric_name):
        if not results:
            print(f"Nessun risultato valido per '{metric_name}'")
            return
        print(f"Batch Means per '{metric_name}':")
        print(f"  - Media: {results['mean']:.4f}")
        print(f"  - CI {results['confidence_level']:.0%}: ({results['ci'][0]:.4f}, {results['ci'][1]:.4f})")
        print(f"  - Batch size: {results.get('batch_size','?')}, #Batch: {results.get('num_batches','?')}")
        if 'ljung_box_pvalue' in results:
            p = results['ljung_box_pvalue']
            print(f"  - Ljung–Box p: {p:.4f} ({'OK' if results['independence_ok'] else '-> ATTENZIONE autocorrelazione'})")





    def calculate_throughput_ci(self,completion_timestamps,warmup_period,confidence_level=0.95,threshold=0.2):
        """
        Calcola il throughput medio (eventi/sec) e il suo intervallo di confidenza
        utilizzando il metodo Batch Means su una serie di timestamp di eventi.

        Args:
            completion_timestamps (list[float]): timestamp degli eventi completati
            warmup_period (float): tempo di warm-up da escludere
            confidence_level (float): livello di confidenza per IC
            threshold (float): soglia per calcolo batch size

        Returns:
            dict | None: dizionario con mean, ci, half_width, total_count, ecc.
        """
        # 1. Filtra i dati per rimuovere il transitorio
        steady_state_timestamps = [t for t in completion_timestamps if t >= warmup_period]

        if not steady_state_timestamps:
            print("Warning: Nessun dato in steady-state per calcolare il throughput.")
            return None

        # 2. Calcola throughput come inverse inter-arrival times
        interarrivals = np.diff(steady_state_timestamps)
        if len(interarrivals) == 0:
            print("Warning: Non ci sono inter-arrival per calcolare il throughput.")
            return None

        throughputs = 1.0 / interarrivals

        # 3. Determina il batch size usando compute_batch_size
        batch_result = compute_batch_size(throughputs, threshold=threshold)
        if not batch_result or len(batch_result) < 2:
            print("Warning: compute_batch_size non ha restituito b e k validi.")
            return None

        b, k = int(batch_result[0]), int(batch_result[1])
        if b <= 0 or k <= 0:
            print("Warning: batch_size o num_batches <= 0, impossibile calcolare batch means.")
            return None

        # 4. Calcola batch means
        results = batch_means(throughputs, b, k, confidence=confidence_level)

        # 5. Test di indipendenza (Ljung-Box)
        batch_values = [np.mean(throughputs[i*b:(i+1)*b]) for i in range(k)]
        pval = ljung_box_test(batch_values, h=min(10, k-1))
        results["ljung_box_pvalue"] = pval
        results["independence_ok"] = (pval is None) or (pval > 0.05)
        results["total_count"] = len(steady_state_timestamps)

        return results







