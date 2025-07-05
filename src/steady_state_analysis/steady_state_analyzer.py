# src/analysis/steady_state_analyzer.py
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

class SteadyStateAnalyzer:
    """
    Una classe dedicata all'analisi di regime permanente (steady-state)
    utilizzando il metodo dei Batch Means.
    """
    def __init__(self, metrics, config):
        self.metrics = metrics
        self.config = config

    def calculate_batch_means_ci(self, metric_data, warmup_period, num_batches, confidence_level=0.95):
        """
        Calcola la media puntuale e l'intervallo di confidenza per una serie di dati
        utilizzando il metodo Batch Means.

        Args:
            metric_data (list): Lista di tuple (timestamp, valore).
            warmup_period (float): Durata del transitorio da scartare.
            num_batches (int): Numero di batch in cui dividere i dati.
            confidence_level (float): Livello di confidenza desiderato (es. 0.95 per 95%).

        Returns:
            dict: Un dizionario con media, intervallo di confidenza e semi-ampiezza, o None se i dati non sono sufficienti.
        """
        # 1. Rimozione del transitorio (Warm-up)
        steady_state_values = [value for timestamp, value in metric_data if timestamp >= warmup_period]

        n = len(steady_state_values)
        if n < num_batches:
            print(f"Errore: Dati insufficienti per creare {num_batches} batch. Osservazioni disponibili: {n}")
            return None

        # 2. Creazione dei Batch e calcolo delle medie dei batch
        batch_size = n // num_batches
        batch_means = []
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            batch = steady_state_values[start_index:end_index]
            batch_means.append(np.mean(batch))

        # 3. Calcolo della media generale e della varianza campionaria delle medie dei batch
        grand_mean = np.mean(batch_means)
        sample_variance = np.var(batch_means, ddof=1) # ddof=1 per varianza campionaria (diviso per k-1)

        # 4. Calcolo dell'intervallo di confidenza
        degrees_freedom = num_batches - 1
        t_value = t.ppf((1 + confidence_level) / 2, df=degrees_freedom)

        half_width = t_value * np.sqrt(sample_variance / num_batches)

        ci_lower = grand_mean - half_width
        ci_upper = grand_mean + half_width

        return {
            'mean': grand_mean,
            'ci': (ci_lower, ci_upper),
            'half_width': half_width,
            'confidence_level': confidence_level,
            'num_batches': num_batches
        }

    def print_ci_results(self, results, metric_name):
        """Stampa i risultati dell'analisi CI in modo leggibile."""
        print(f"Risultati Batch Means per '{metric_name}':")
        print(f"  - Stima Puntuale della Media: {results['mean']:.4f}")
        print(f"  - Intervallo di Confidenza al {results['confidence_level']:.0%}: ({results['ci'][0]:.4f}, {results['ci'][1]:.4f})")
        print(f"  - Semi-Ampiezza (Half-Width): {results['half_width']:.4f}")

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
