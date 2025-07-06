import os
import matplotlib.pyplot as plt
import numpy as np
from src.config import RequestType

class ValidationPlotter:
    """
    Questa classe è dedicata alla creazione di grafici specifici per la fase di
    Verifica e Validazione del modello, come l'analisi di stabilità al variare del carico.
    """
    def __init__(self, all_metrics_data: dict, config):
        """
        Args:
            all_metrics_data (dict): Un dizionario che contiene i risultati di tutte le run.
                                     Formato: {'tasso_70': {'baseline': metrics, 'priority': metrics}, ...}
            config: Il modulo di configurazione.
        """
        self.all_metrics = all_metrics_data
        self.config = config

    def _save_plot(self, output_dir, filename, fig):
        """Helper per creare la directory e salvare la figura."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_stability_analysis_grid(self, output_dir="output/validation"):
        """
        Crea un grafico a griglia che confronta l'andamento del tempo di risposta medio
        cumulativo per i diversi scenari di carico (lambda), per entrambi i modelli.
        Questo è il grafico FONDAMENTALE per la validazione.
        """
        print("Generazione grafico di analisi di stabilità...")

        scenarios = self.all_metrics.keys()
        num_scenarios = len(scenarios)

        fig, axes = plt.subplots(num_scenarios, 2, figsize=(16, 6 * num_scenarios), sharex=True, sharey=True)
        fig.suptitle("Analisi di Stabilità: Tempo di Risposta Medio Cumulativo vs. Carico", fontsize=20, fontweight='bold')

        for i, scenario_name in enumerate(scenarios):
            # Grafico per il modello Baseline (colonna sinistra)
            ax_base = axes[i, 0]
            metrics_base = self.all_metrics[scenario_name]['baseline']

            all_responses_base = metrics_base.get_all_response_times_with_timestamps()
            if all_responses_base:
                times, values = zip(*all_responses_base)
                cusum = np.cumsum(values) / np.arange(1, len(values) + 1)
                ax_base.plot(times, cusum, color='r', label=f"λ = {scenario_name.split('_')[1]}")

            ax_base.set_title(f"Baseline - {scenario_name}")
            ax_base.grid(True, linestyle='--', alpha=0.6)
            ax_base.set_ylabel("Tempo Medio (s)")

            # Grafico per il modello con Priorità (colonna destra)
            ax_prio = axes[i, 1]
            metrics_prio = self.all_metrics[scenario_name]['priority']

            all_responses_prio = metrics_prio.get_all_response_times_with_timestamps()
            if all_responses_prio:
                times, values = zip(*all_responses_prio)
                cusum = np.cumsum(values) / np.arange(1, len(values) + 1)
                ax_prio.plot(times, cusum, color='b', label=f"λ = {scenario_name.split('_')[1]}")

            ax_prio.set_title(f"Con Priorità - {scenario_name}")
            ax_prio.grid(True, linestyle='--', alpha=0.6)

        # Aggiungi etichette comuni
        axes[-1, 0].set_xlabel("Tempo di Simulazione (s)")
        axes[-1, 1].set_xlabel("Tempo di Simulazione (s)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, "stability_analysis_grid.png", fig)


    def plot_cumulative_loss_analysis(self, output_dir="output/validation"):
            """
            Crea un grafico che mostra l'andamento del numero cumulativo di richieste
            perse (timeout) per i diversi scenari di carico, per entrambi i modelli.
            Questo grafico è ideale per visualizzare l'instabilità.
            """
            print("Generazione grafico di analisi delle perdite cumulative...")

            scenarios = self.all_metrics.keys()
            num_scenarios = len(scenarios)

            fig, axes = plt.subplots(num_scenarios, 2, figsize=(16, 6 * num_scenarios), sharex=True)
            fig.suptitle("Analisi di Instabilità: Richieste Perse Cumulative vs. Carico", fontsize=20, fontweight='bold')

            for i, scenario_name in enumerate(scenarios):
                # Grafico per il modello Baseline (colonna sinistra)
                ax_base = axes[i, 0]
                metrics_base = self.all_metrics[scenario_name]['baseline']

                times, counts = metrics_base.get_cumulative_timeouts()
                if times:
                    ax_base.plot(times, counts, color='r', label=f"λ = {scenario_name.split('_')[1]}")

                ax_base.set_title(f"Baseline - {scenario_name}")
                ax_base.grid(True, linestyle='--', alpha=0.6)
                ax_base.set_ylabel("Numero Cumulativo di Richieste Perse")

                # Grafico per il modello con Priorità (colonna destra)
                ax_prio = axes[i, 1]
                metrics_prio = self.all_metrics[scenario_name]['priority']

                times_p, counts_p = metrics_prio.get_cumulative_timeouts()
                if times_p:
                    ax_prio.plot(times_p, counts_p, color='b', label=f"λ = {scenario_name.split('_')[1]}")

                ax_prio.set_title(f"Con Priorità - {scenario_name}")
                ax_prio.grid(True, linestyle='--', alpha=0.6)

            # Aggiungi etichette comuni
            axes[-1, 0].set_xlabel("Tempo di Simulazione (s)")
            axes[-1, 1].set_xlabel("Tempo di Simulazione (s)")

            # Usa una scala logaritmica sull'asse Y per gestire grandi differenze di scala
            for ax_row in axes:
                for ax in ax_row:
                    ax.set_yscale('log')
                    ax.set_ylim(bottom=1) # La scala log non può iniziare da 0

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self._save_plot(output_dir, "instability_analysis_cumulative_loss.png", fig)

    # Sostituisci il vecchio plot_cumulative_loss_analysis con questo nuovo metodo

    def plot_loss_rate_analysis(self, output_dir="output/validation"):
        """
        Crea un grafico che mostra l'evoluzione del TASSO di perdita (timeout/sec)
        calcolato su una finestra mobile. Ideale per distinguere stabilità da instabilità.
        """
        print("Generazione grafico di analisi del tasso di perdita...")

        scenarios = self.all_metrics.keys()
        num_scenarios = len(scenarios)

        fig, axes = plt.subplots(num_scenarios, 2, figsize=(16, 6 * num_scenarios), sharex=True, sharey=True)
        fig.suptitle("Analisi di Instabilità: Tasso di Perdita (Timeout/s) vs. Carico", fontsize=20, fontweight='bold')

        window_size_sec = 50  # Durata della finestra mobile in secondi

        for i, scenario_name in enumerate(scenarios):
            for j, model_type in enumerate(['baseline', 'priority']):
                ax = axes[i, j]
                metrics = self.all_metrics[scenario_name][model_type]

                timeout_timestamps = sorted([t for t, r in metrics.timeout_history])

                if timeout_timestamps:
                    # Crea i bin per le finestre temporali
                    sim_duration = self.config.SIMULATION_TIME
                    bins = np.arange(0, sim_duration + window_size_sec, window_size_sec)

                    # Conta i timeout in ogni bin
                    counts, _ = np.histogram(timeout_timestamps, bins=bins)

                    # Calcola il tasso (conteggio / durata finestra)
                    rates = counts / window_size_sec

                    # Prendi i punti centrali dei bin per l'asse X
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    color = 'r' if model_type == 'baseline' else 'b'
                    ax.plot(bin_centers, rates, color=color, label=f"λ = {scenario_name.split('_')[1]}")

                ax.set_title(f"{model_type.title()} - {scenario_name}")
                ax.grid(True, linestyle='--', alpha=0.6)
                if j == 0:
                    ax.set_ylabel("Tasso di Perdita (req/s)")

        axes[-1, 0].set_xlabel("Tempo di Simulazione (s)")
        axes[-1, 1].set_xlabel("Tempo di Simulazione (s)")

        # Usiamo una scala lineare, che ora funzionerà perfettamente
        for ax_row in axes:
            for ax in ax_row:
                ax.set_ylim(bottom=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, "instability_analysis_loss_rate.png", fig)

    def generate_validation_report(self, output_dir="output/validation"):
        """Metodo principale per generare tutti i grafici di validazione."""
        print(f"\n--- Generazione Report di Validazione in '{output_dir}' ---")
        self.plot_stability_analysis_grid(output_dir)

        # Chiama il nuovo metodo
        self.plot_loss_rate_analysis(output_dir)
