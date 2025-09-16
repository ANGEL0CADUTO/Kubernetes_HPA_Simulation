
import numpy as np


from src.utils.acs import batch_means, compute_batch_size, ljung_box_test

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