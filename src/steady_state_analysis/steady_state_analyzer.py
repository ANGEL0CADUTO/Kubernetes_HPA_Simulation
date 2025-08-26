
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, stats

from src.config import RequestType



class SteadyStateAnalyzer:
    """
    Una classe dedicata all'analisi di regime permanente (steady-state)
    utilizzando il metodo dei Batch Means.
    """
    def __init__(self, metrics, config):
        self.metrics = metrics
        self.config = config
        self.steady_state_detected = False
        self.steady_state_start_time = None
        self.system_state_history = []  # Per memorizzare lo stato del sistema nel tempo

    def calculate_lag1_autocorrelation(self, batch_means):
        """
        Calcola la lag-1 autocorrelazione di una serie di medie di batch.
        """
        k = len(batch_means)
        if k < 2: return 0.0
        mean_of_means = np.mean(batch_means)
        covariance = sum((batch_means[j] - mean_of_means) * (batch_means[j-1] - mean_of_means) for j in range(1, k)) / (k - 1)
        variance = np.var(batch_means, ddof=0)
        return covariance / variance if variance != 0 else 0.0

    def calculate_batch_means_ci(self, metric_data, warmup_period, num_batches, confidence_level=0.95):
        """
        Calcola la media puntuale, l'IC e l'autocorrelazione.
        """
        if not metric_data: return None
        steady_state_values = [value for timestamp, value in metric_data if timestamp >= warmup_period]
        n = len(steady_state_values)
        if n < num_batches: return None

        batch_size = n // num_batches
        if batch_size == 0: return None

        batch_means = [np.mean(steady_state_values[i*batch_size:(i+1)*batch_size]) for i in range(num_batches)]

        if len(batch_means) < 2: return None

        # Calcolo dell'autocorrelazione
        lag1_autocorr = self.calculate_lag1_autocorrelation(batch_means)

        # Calcolo dell'intervallo di confidenza
        grand_mean = np.mean(batch_means)
        sample_variance = np.var(batch_means, ddof=1)
        degrees_freedom = len(batch_means) - 1
        t_value = t.ppf((1 + confidence_level) / 2, df=degrees_freedom)
        half_width = t_value * np.sqrt(sample_variance / len(batch_means))

        return {
            'mean': grand_mean,
            'ci': (grand_mean - half_width, grand_mean + half_width),
            'half_width': half_width,
            'confidence_level': confidence_level,
            'num_batches': len(batch_means),
            'lag1_autocorrelation': lag1_autocorr # <-- AGGIUNTO!
        }

    def plot_confidence_interval(self, results, title, output_dir, filename):
        """Crea un grafico che visualizza la media e il suo intervallo di confidenza."""
        mean = results['mean']
        half_width = results['half_width']

        fig, ax = plt.subplots(figsize=(6, 5))

        # Disegna il punto della media e la barra d'errore
        ax.errorbar(x=[0], y=[mean], yerr=half_width, fmt='o', color='b',
                    capsize=10, markersize=8, elinewidth=3, label='Intervallo di Confidenza')

        # Estetica del grafico
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Valore Medio')
        ax.set_xticks([]) # Nasconde l'asse x che non serve
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Aggiungi testo per chiarezza
        ci_text = f"Media: {mean:.3f}\nCI al {results['confidence_level']:.0%}: [{results['ci'][0]:.3f}, {results['ci'][1]:.3f}]"
        ax.text(0.05, 0.95, ci_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def calculate_throughput_ci(self, completion_timestamps, warmup_period, num_batches, confidence_level=0.95):
        """
        Calcola il throughput medio (eventi/sec) e il suo intervallo di confidenza
        utilizzando il metodo Batch Means su una serie di timestamp di eventi.
        """
        # 1. Filtra i dati per rimuovere il transitorio
        steady_state_timestamps = [t for t in completion_timestamps if t >= warmup_period]

        if not steady_state_timestamps:
            print("Warning: Nessun dato in steady-state per calcolare il throughput.")
            return None

        # 2. Definisci l'intervallo di tempo dell'analisi steady-state
        start_time = warmup_period
        end_time = steady_state_timestamps[-1]
        total_duration = end_time - start_time

        if total_duration <= 0:
            print("Warning: Durata steady-state non positiva.")
            return None

        # 3. Dividi la DURATA in batch e calcola i tassi dei batch
        batch_duration = total_duration / num_batches
        batch_throughputs = []
        for i in range(num_batches):
            batch_start = start_time + i * batch_duration
            batch_end = batch_start + batch_duration

            # Conta quanti eventi sono caduti in questo intervallo di tempo
            events_in_batch = sum(1 for t in steady_state_timestamps if batch_start <= t < batch_end)

            # Calcola il tasso per questo batch (eventi / secondi)
            batch_rate = events_in_batch / batch_duration
            batch_throughputs.append(batch_rate)

        # 4. Usa la logica Batch Means esistente sui tassi calcolati
        grand_mean_throughput = np.mean(batch_throughputs)
        sample_variance = np.var(batch_throughputs, ddof=1)

        degrees_freedom = num_batches - 1
        if degrees_freedom <= 0: return None # Non si può calcolare se c'è < 2 batch

        t_value = t.ppf((1 + confidence_level) / 2, df=degrees_freedom)

        half_width = t_value * np.sqrt(sample_variance / num_batches)

        ci_lower = grand_mean_throughput - half_width
        ci_upper = grand_mean_throughput + half_width

        return {
            'mean': grand_mean_throughput,
            'ci': (ci_lower, ci_upper),
            'half_width': half_width,
            # Aggiungiamo anche il conteggio totale per le etichette
            'total_count': len(steady_state_timestamps)
        }

    def print_ci_results(self, results, metric_name):
        """Stampa i risultati dell'analisi CI in modo leggibile."""

        if not results:
            print(f"Nessun risultato valido da stampare per '{metric_name}'.")
            return # Il return va DENTRO l'if.

        # Se i risultati esistono, il codice prosegue e stampa tutto.
        print(f"Risultati Batch Means per '{metric_name}':")
        print(f"  - Stima Puntuale della Media: {results['mean']:.4f}")
        print(f"  - Intervallo di Confidenza al {results['confidence_level']:.0%}: ({results['ci'][0]:.4f}, {results['ci'][1]:.4f})")

        if 'lag1_autocorrelation' in results:
            autocorr_val = results['lag1_autocorrelation']
            print(f"  - Lag-1 Autocorrelazione tra Batch: {autocorr_val:.4f}")
            if abs(autocorr_val) > 0.2:
                print("    -> ATTENZIONE: Autocorrelazione > 0.2. I batch potrebbero non essere sufficientemente indipendenti.")



    """ -------------------------------NUOVO---------------------------"""
    #Esempio dei batch means con welford

    def detect_steady_state(self, timestamp, queue_length, active_pods):
        """
        Implementa rilevamento automatico dello steady-state usando test statistici.
        Basato su Welch's method per rilevamento di convergenza.
        """
        self.system_state_history.append((timestamp, queue_length, active_pods))

        # Necessari almeno 100 campioni per test affidabile
        if len(self.system_state_history) < 100:
            return False

        # Test di stazionarietà sulle ultime 50 vs precedenti 50 osservazioni
        recent_data = [x[1] for x in self.system_state_history[-50:]]  # queue_length
        previous_data = [x[1] for x in self.system_state_history[-100:-50]]

        # Test t di Welch per uguaglianza delle medie
        t_stat, p_value = stats.ttest_ind(recent_data, previous_data, equal_var=False)

        # Se p > 0.05, le medie non sono significativamente diverse
        if p_value > 0.05 and not self.steady_state_detected:
            self.steady_state_detected = True
            self.steady_state_start_time = timestamp
            print(f"STEADY-STATE RILEVATO al tempo {timestamp:.2f}s (p-value={p_value:.4f})")

        return self.steady_state_detected

    def batch_means_analysis(self, req_type: RequestType, num_batches=20):
        if not self.steady_state_detected:
            print("ATTENZIONE: Steady-state non rilevato. Risultati potrebbero essere biased.")

        steady_start = self.steady_state_start_time or 0
        data = self.metrics.get_response_times_by_type(req_type)
        steady_data = [(t, v) for (t, v) in data if t >= steady_start]

        if len(steady_data) < num_batches * 10:
            return None

        steady_values = [v for _, v in steady_data]
        batch_size = len(steady_values) // num_batches

        batch_means = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size if i < num_batches - 1 else len(steady_values)
            batch_means.append(np.mean(steady_values[start_idx:end_idx]))

        batch_mean = np.mean(batch_means)
        batch_var = np.var(batch_means, ddof=1)
        dof = len(batch_means) - 1
        t_critical = stats.t.ppf(0.975, dof)
        margin_of_error = t_critical * np.sqrt(batch_var / len(batch_means))

        return {
            'mean': batch_mean,
            'variance': batch_var,
            'confidence_interval': (batch_mean - margin_of_error, batch_mean + margin_of_error),
            'num_batches': len(batch_means),
            'batch_size': batch_size,
            'steady_state_used': self.steady_state_detected
        }



    def test_independence(self, req_type: RequestType, max_lag=20):
        """
        Test di autocorrelazione per verificare indipendenza dei dati.
        Critico per validità dell'analisi statistica.
        """
        if not self.steady_state_detected:
            return None

        steady_start = self.steady_state_start_time or 0
        data_times = self.completion_timestamps[req_type]
        data_values = self.response_times_raw[req_type]

        steady_data = [v for t, v in zip(data_times, data_values) if t >= steady_start]

        if len(steady_data) < 100:
            return None

        # Calcola autocorrelazioni per vari lag
        autocorrs = []
        for lag in range(1, min(max_lag + 1, len(steady_data) // 4)):
            correlation = np.corrcoef(steady_data[:-lag], steady_data[lag:])[0, 1]
            autocorrs.append((lag, correlation))

        # Test se qualche autocorrelazione è significativamente diversa da 0
        significant_correlations = [(lag, corr) for lag, corr in autocorrs if abs(corr) > 0.2]

        return {
            'autocorrelations': autocorrs,
            'significant_correlations': significant_correlations,
            'independence_violated': len(significant_correlations) > 0
        }

    def comprehensive_statistical_report(self):
        """
        Genera un report completo che soddisfa i criteri di Kurkowski.
        """
        print("\n" + "="*80)
        print("REPORT STATISTICO RIGOROSO (secondo Kurkowski et al.)")
        print("="*80)

        print(f"\n1. CONFIGURAZIONE DELLA SIMULAZIONE:")
        print(f"   - Seed Lehmer: {self.config.LEHMER_SEED}")
        print(f"   - Durata simulazione: {self.config.SIMULATION_TIME}s")
        print(f"   - Steady-state rilevato: {'SÌ' if self.steady_state_detected else 'NO'}")
        if self.steady_state_start_time:
            print(f"   - Inizio steady-state: {self.steady_state_start_time:.2f}s")

        print(f"\n2. ANALISI BATCH MEANS (95% Confidence Intervals):")
        for req_type in RequestType:
            batch_analysis = self.batch_means_analysis(req_type)
            if batch_analysis:
                ci_lower, ci_upper = batch_analysis['confidence_interval']
                print(f"   {req_type.name:12}: μ = {batch_analysis['mean']:.4f}s")
                print(f"   {'':12}  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"   {'':12}  Batches: {batch_analysis['num_batches']}")

        print(f"\n3. TEST DI INDIPENDENZA (Autocorrelazione):")
        for req_type in RequestType:
            independence = self.test_independence(req_type)
            if independence:
                status = "VIOLATA" if independence['independence_violated'] else "OK"
                print(f"   {req_type.name:12}: {status}")
                if independence['significant_correlations']:
                    print(f"   {'':12}  Correlazioni significative: {independence['significant_correlations']}")

        print(f"\n4. VALIDAZIONE STATISTICA:")
        print(f"   - Initialization bias: {'Gestito' if self.steady_state_detected else 'NON GESTITO'}")
        print(f"   - Multiple runs: {'Implementato' if hasattr(self, 'run_id') else 'NON IMPLEMENTATO'}")
        print(f"   - Confidence intervals: Implementati")
        print(f"   - Independence test: Implementato")



