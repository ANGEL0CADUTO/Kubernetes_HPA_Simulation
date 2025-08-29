import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
#from src.config import RequestType
import matplotlib.ticker as mticker

matplotlib.use('Qt5Agg')

class newPlotter:
    """
    Questa classe si occupa di generare i nuovi grafici aggiornati.
    - Analisi del transitorio: grafico che mette a confronto seed diversi.
    - Analisi stazionaria: grafico che mostra le medie usando batch means.
    - Confronto base e soluzione migliorativa: grafici per lo stato stazionario.
    """
    def __init__(self, metrics: 'Metrics', metrics_prio: 'MetricsWithPriority', config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config

    # --- FUNZIONI HELPER STATICHE ---
    # Queste funzioni non dipendono dallo stato dell'istanza (self),
    # quindi le dichiariamo come metodi statici.

    @staticmethod
    def _plot_single_replication_trace(ax: plt.Axes, metrics_data: Any, color: tuple, label: str):
        """
        Funzione helper per calcolare e plottare la traccia di una singola replica.
        """
        all_responses = metrics_data.get_all_response_times_with_timestamps()
        if not all_responses:
            return

        times, values = zip(*all_responses)
        cumulative_average = np.cumsum(values) / np.arange(1, len(values) + 1)
        ax.plot(times, cumulative_average, color=color, label=label, linewidth=2.0, alpha=0.8)

    @staticmethod
    def _style_subplot(ax: plt.Axes, title: str, y_max: float):
        """
        Applica uno stile standardizzato a un subplot.
        """
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_ylabel('Tempo di Risposta Medio (s)', fontsize=14)
        ax.set_ylim(0, y_max)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12, loc='upper right')
        # La griglia è ora gestita dallo stile globale applicato all'inizio
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)


    # --- Funzione Principale ---
    # Questa è un'operazione che aggrega risultati da più simulazioni,
    # quindi ha senso che sia un metodo statico, dato che non usa 'self'.
    # Riceve tutti i dati di cui ha bisogno come argomento.
    @staticmethod
    def plot_replication_traces_per_scenario(
            all_results: Dict[str, Dict[int, Dict[str, Any]]],
            output_dir: str = 'output/aggregated'
    ):
        """
        Crea un grafico per ogni scenario, confrontando il modello Baseline (FIFO)
        con il modello Migliorato (Priorità) attraverso le tracce delle repliche.
        """
        print("\n--- Generazione Grafici delle Tracce delle Repliche ---")
        plt.style.use('seaborn-v0_8-whitegrid')

        for scenario_name, replications in all_results.items():
            num_replications = len(replications)
            if num_replications == 0:
                print(f"Scenario '{scenario_name}' non ha repliche, skippato.")
                continue

            fig, (ax1, ax2) = plt.subplots(
                nrows=2, ncols=1, figsize=(12, 10), sharex=True
            )
            fig.suptitle(f'Tracce delle Repliche per Scenario: "{scenario_name.upper()}"', fontsize=18, weight='bold')

            colors = sns.color_palette("husl", n_colors=num_replications)

            for i, replication_data in enumerate(replications.values()):
                if not replication_data:
                    continue

                seed = replication_data.get('seed', f'Replica {i+1}')
                label_text = f'Seed: {seed}'
                color = colors[i]

                # --- Grafico 1: Modello Baseline ---
                metrics_base = replication_data.get('baseline')
                if metrics_base:
                    newPlotter._plot_single_replication_trace(ax1, metrics_base, color, label_text)

                # --- Grafico 2: Modello con Priorità ---
                metrics_prio = replication_data.get('priority')
                if metrics_prio:
                    newPlotter._plot_single_replication_trace(ax2, metrics_prio, color, label_text)

            # Imposta i tick principali (le etichette numeriche) sull'asse x ogni 50 unità
            ax2.xaxis.set_major_locator(mticker.MultipleLocator(50))
            # Aggiungi un tick minore (senza etichetta) ogni 25 unità
            ax2.xaxis.set_minor_locator(mticker.MultipleLocator(25))

            y_axis_limit = 5.5
            newPlotter._style_subplot(ax1, 'Modello Baseline (FIFO)', y_axis_limit)
            newPlotter._style_subplot(ax2, 'Modello Migliorato (Priorità)', y_axis_limit)

            ax2.set_xlabel('Tempo di Simulazione (s)', fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            output_path = os.path.join(output_dir, f'replication_traces_{scenario_name}.png')
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Grafico per '{scenario_name}' salvato in: {output_path}")