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
from matplotlib.ticker import MaxNLocator

matplotlib.use('Qt5Agg')

class newPlotter:
    """
    Questa classe si occupa di generare i nuovi grafici aggiornati.
    - Analisi del transitorio: grafico che mette a confronto seed diversi.
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
        ax.plot(times, cumulative_average, color=color, label=label, linewidth=1.0, alpha=0.8)

    @staticmethod
    def _style_subplot(ax: plt.Axes, title: str, y_max: float):
        """
        Applica uno stile standardizzato a un subplot.
        """
        ax.set_title(title, fontsize=16, pad=10)
        ax.set_ylabel('Tempo di Risposta Medio (s) [Scala Log]', fontsize=14) # Aggiorna l'etichetta

        ax.set_yscale('log')  # Imposta la scala logaritmica sull'asse Y
        ax.set_ylim(bottom=0.01, top=y_max)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)

        ax.set_xlim(0, 30)

    # --- Funzione Principale ---
    # Traccia il trend del tempo medio di risposta per i vari seed
    # mette a confronto baseline e migliorato
    # usa la funzione ausiliaria per stampare il singolo replication traces
    # Analisi del transitorio: mostra lo stabilizzarsi del comportamento del sistema confrontando le repliche
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

            y_axis_limit = 10
            newPlotter._style_subplot(ax1, 'Modello Baseline (FIFO)', y_axis_limit)
            newPlotter._style_subplot(ax2, 'Modello Migliorato (Priorità)', y_axis_limit)

            ax2.set_xlabel('Tempo di Simulazione (s)', fontsize=14)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            output_path = os.path.join(output_dir, f'replication_traces_{scenario_name}.png')
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Grafico per '{scenario_name}' salvato in: {output_path}")



    # GRAFICO WORKER + PODS IN SOVRIMPRESSIONE
    def plot_worker_queue_evolution(
            self,
            all_results: dict,
            scenario_name: str,
            replica_idx: int,
            output_dir: str = 'output/worker_analysis',
            overlay_pods: bool = True
    ):
        """
        Crea un grafico che mostra l'evoluzione temporale della lunghezza della coda
        per ogni Worker Node, evidenziando la formazione di hotspot.
        Opzionalmente, sovrappone l'andamento del numero di pod per ogni worker.


        """
        print(f"Generazione grafico evoluzione code worker per '{scenario_name}', replica {replica_idx}...")

        # --- 1. Estrazione Dati ---
        # Prendiamo i dati della simulazione Baseline, ma potrebbe essere parametrizzato
        try:
            replica_data = all_results[scenario_name][replica_idx]
            metrics_per_worker_base = replica_data['baseline'].metrics_per_worker
            seed = replica_data.get('seed', f'Replica {replica_idx+1}')
            num_workers = len(metrics_per_worker_base)
        except (KeyError, IndexError):
            print(f"Errore: Dati non trovati per scenario '{scenario_name}', replica {replica_idx}.")
            return

        # --- 2. Preparazione Figura e Assi ---
        fig, ax1 = plt.subplots(figsize=(16, 8))
        fig.suptitle(f'Evoluzione Code Worker (FIFO) - Scenario: {scenario_name.upper()} - Seed: {seed}',
                     fontsize=18, fontweight='bold')

        # Asse Y primario (sinistra) per la lunghezza della coda
        ax1.set_xlabel('Tempo di Simulazione (s)', fontsize=14)
        ax1.set_ylabel('N. Richieste in Coda (Scala Log)', fontsize=14, color='black')
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=1)  # La scala logaritmica non può iniziare da 0
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        # --- 3. Plot delle Code ---
        # Genera una palette di colori distinta per i worker
        colors = sns.color_palette("husl", n_colors=num_workers)

        for i in range(num_workers):
            metrics = metrics_per_worker_base[i]
            if metrics.queue_length_history:
                # La history è una lista di tuple (timestamp, queue_length)
                timestamps, lengths = zip(*metrics.queue_length_history)
                ax1.plot(timestamps, lengths, label=f'Coda Worker {i}', color=colors[i], linewidth=2.5)

        # --- 4. Plot Opzionale dei Pod (se overlay_pods è True) ---
        ax2 = None
        if overlay_pods:
            # Crea un secondo asse Y (destra) che condivide l'asse X con il primo
            ax2 = ax1.twinx()
            ax2.set_ylabel('Numero di Pod Attivi', fontsize=14, color='dimgray')
            # Forza i tick dell'asse Y dei pod ad essere interi
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.tick_params(axis='y', labelcolor='dimgray')
            # Imposta un limite massimo ragionevole per i pod
            max_pods_config = getattr(self.config, 'MAX_PODS', 50)
            ax2.set_ylim(bottom=0, top=max_pods_config * 1.1)

            for i in range(num_workers):
                metrics = metrics_per_worker_base[i]
                if metrics.pod_count_history:
                    timestamps, counts = zip(*metrics.pod_count_history)
                    # Usa lo stesso colore della coda ma con più trasparenza (più chiaro)
                    # e uno stile di linea diverso per evitare confusione.
                    ax2.plot(timestamps, counts, label=f'Pod Worker {i}', color=colors[i],
                             linestyle=':', linewidth=2, alpha=0.8)

        # --- 5. Creazione Legenda Unificata e Salvataggio ---
        # Combina le legende di entrambi gli assi in una sola
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))

        # Ordina le etichette per worker ID per una legenda più pulita
        all_labels = labels1 + labels2
        all_lines = lines1 + lines2
        sorted_legend = sorted(zip(all_labels, all_lines), key=lambda x: x[0].split(' ')[-1])

        # Estrai le etichette e le linee ordinate
        if sorted_legend:
            sorted_labels, sorted_lines = zip(*sorted_legend)
            ax1.legend(sorted_lines, sorted_labels, loc='upper left', fontsize=12, title="Metriche per Worker")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta il layout per il titolo

        # Crea la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)

        # Nome del file dinamico in base all'opzione di overlay
        filename_suffix = "pods_overlay" if overlay_pods else "queues_only"
        filename = f'worker_queues_{scenario_name}_rep{replica_idx}_{filename_suffix}.png'
        save_path = os.path.join(output_dir, filename)

        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")




