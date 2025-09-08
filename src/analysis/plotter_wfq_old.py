# File: analysis/plotter_wfq.py (VERSIONE REFATTORIZZATA E MIGLIORATA)

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from src import config # Assumo che il modulo config sia importabile

# Impostazione del backend e dello stile a livello di modulo
matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterWFQ:
    def __init__(self, metrics_base, metrics_prio, metrics_wfq, config_module):
        self.metrics_base = metrics_base
        self.metrics_prio = metrics_prio
        self.metrics_wfq = metrics_wfq
        self.config = config_module

    # --- METODI HELPER PRIVATI ---

    def _save_plot(self, output_dir, filename, fig):
        """Salva la figura in un file, creando la cartella se non esiste."""
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    @staticmethod
    def _get_time_based_moving_average(history: list, window_str: str = '10s') -> pd.Series:
        """Calcola la media mobile basata sul tempo usando pandas."""
        if not history or len(history) < 2:
            return pd.Series(dtype=np.float64)
        times, values = zip(*sorted(history, key=lambda x: x[0]))
        s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
        # Ricampiona a 1s per avere una griglia regolare, poi applica la media mobile
        return s.resample('1s').mean().rolling(window=window_str, min_periods=1).mean()

    @staticmethod
    def _plot_series(ax: plt.Axes, series: pd.Series, **kwargs):
        """Plotta una pandas Series sull'asse specificato."""
        if series is not None and not series.empty:
            time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
            ax.plot(time_in_seconds, series.values, **kwargs)

    def _setup_plot(self, ax: plt.Axes, title: str, ylabel: str, ylim_top: float):
        """Applica uno stile standard a un asse."""
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Tempo (s)", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
        ax.set_ylim(bottom=0, top=ylim_top)

    def _plot_load_profile(self, ax: plt.Axes, load_times: list, load_values: list):
        """Aggiunge il profilo di carico su un asse Y secondario."""
        ax_load = ax.twinx()
        ax_load.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load.set_ylabel("Carico (req/s)", color='gray', fontsize=14)
        ax_load.tick_params(axis='y', labelcolor='gray')
        return ax_load # Ritorna l'asse per la gestione della legenda

    def _combine_and_set_legend(self, ax: plt.Axes, ax_twin: plt.Axes, loc='upper left'):
        """Combina le legende da due assi e le mostra."""
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=loc, fontsize=12)

    @staticmethod
    def _calculate_rate_histogram(timestamps: list, sim_time: float, window_sec: int):
        """Calcola un istogramma di tasso (es. throughput, timeout/sec)."""
        if not timestamps:
            return None, None
        bins = np.arange(0, sim_time + window_sec, window_sec)
        counts, _ = np.histogram(timestamps, bins=bins)
        rate = counts / window_sec
        times = bins[:-1]
        return times, rate

    # --- METODO PRINCIPALE PER LA GENERAZIONE DEI GRAFICI ---

    def generate_final_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):

        sim_time = self.config.SIMULATION_TIME
        TIME_WINDOW_SEC = 10

        # --- 1. CENTRALIZZAZIONE DEGLI STILI ---
        # Definiamo colori e stili in un unico posto per coerenza
        styles = {
            'Baseline': {'color': 'royalblue', 'linestyle': '--', 'lw': 2.5},
            'Priorità': {'color': 'darkred', 'linestyle': '-', 'lw': 2.0},
            'WFQ':      {'color': 'limegreen', 'linestyle': '-', 'lw': 2.5}
        }
        prio_colors = {
            config.Priority.HIGH: 'green',
            config.Priority.MEDIUM: 'orange',
            config.Priority.LOW: 'purple'
        }

        # --- 2. PREPARAZIONE DEI DATI (una sola volta all'inizio) ---
        print("\n--- Preparazione dati per i grafici finali ---")
        base_ma = self._get_time_based_moving_average(self.metrics_base.get_all_response_times_with_timestamps())
        prio_ma_data = {p: self._get_time_based_moving_average(hist) for p, hist in self.metrics_prio.response_times_history_by_prio.items()}
        wfq_ma_data = {p: self._get_time_based_moving_average(hist) for p, hist in self.metrics_wfq.response_times_history_by_prio.items()}

        load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]
        load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]

        # --- 3. GENERAZIONE GRAFICI TEMPO DI RISPOSTA (Grafici 1, 2) ---
        print("--- Generazione grafici Tempo di Risposta ---")
        for prio in [config.Priority.HIGH, config.Priority.LOW]:
            fig, ax = plt.subplots(figsize=(20, 8))

            plot_title = f"Protezione QoS per Priorità {prio.name}" if prio == config.Priority.HIGH else f"Analisi Starvation per Priorità {prio.name}"
            self._setup_plot(ax, f"{plot_title} - {run_prefix}", "Tempo Risposta Medio (s)", 20.0)

            self._plot_series(ax, base_ma, label='Baseline (FIFO)', **styles['Baseline'])
            self._plot_series(ax, prio_ma_data.get(prio), label=f'Priorità Strette ({prio.name})', **styles['Priorità'])
            self._plot_series(ax, wfq_ma_data.get(prio), label=f'WFQ ({prio.name})', **styles['WFQ'])

            ax_load = self._plot_load_profile(ax, load_times, load_values)
            self._combine_and_set_legend(ax, ax_load)

            fig.tight_layout()
            filename = f"{run_prefix}_{1 if prio == config.Priority.HIGH else 2}_{'QoS' if prio == config.Priority.HIGH else 'Starvation'}_{prio.name}.png"
            self._save_plot(output_dir, filename, fig)

        # --- 4. GRAFICO PERFORMANCE INTERNA WFQ (Grafico 3) ---
        print("--- Generazione grafico Performance Interna WFQ ---")
        fig3, ax3 = plt.subplots(figsize=(20, 8))
        self._setup_plot(ax3, f"Performance Interna del DWFQ - {run_prefix}", "Tempo Risposta Medio (s)", 20.0)
        for prio in config.Priority:
            self._plot_series(ax3, wfq_ma_data.get(prio), color=prio_colors.get(prio, 'black'), lw=2.5, label=f'DWFQ - {prio.name}')
        ax_load3 = self._plot_load_profile(ax3, load_times, load_values)
        self._combine_and_set_legend(ax3, ax_load3)
        fig3.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_3_WFQ_Internal_Performance.png", fig3)

        # --- 5. GENERAZIONE GRAFICI DI THROUGHPUT (Grafici 4, 5, 6) ---
        print("--- Generazione grafici di Throughput ---")
        for i, prio in enumerate(config.Priority, 4):
            fig, ax = plt.subplots(figsize=(20, 8))
            self._setup_plot(ax, f"Throughput (req/s) per Priorità {prio.name} - {run_prefix}", "Throughput (req/s)", 85.0)

            # Calcolo carico teorico per questa priorità
            prio_share = sum(prob for rt, prob in self.config.TRAFFIC_PROFILE.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == prio)
            peak_load_prio = peak_load * prio_share
            ax.axhline(y=peak_load_prio, color='gray', linestyle=':', lw=2, alpha=0.9, label=f'Tasso Arrivo Teorico {prio.name} (~{peak_load_prio:.1f} req/s)')

            # Calcolo e plot del throughput per ogni modello
            for model_name, metrics_obj in [('Baseline', self.metrics_base), ('Priorità', self.metrics_prio), ('WFQ', self.metrics_wfq)]:
                if model_name == 'Baseline':
                    ts_list = [ts for rt, hist in metrics_obj.response_times_history.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == prio for ts, _ in hist]
                else: # Per Priorità e WFQ
                    ts_list = [ts for ts, _ in metrics_obj.response_times_history_by_prio.get(prio, [])]

                times, rate = self._calculate_rate_histogram(ts_list, sim_time, TIME_WINDOW_SEC)
                if times is not None:
                    label = f"{model_name} (FIFO)" if model_name == 'Baseline' else model_name
                    ax.plot(times, rate, label=f'{label} - Throughput {prio.name}', drawstyle='steps-post', **styles[model_name])

            ax.legend(loc='upper left', fontsize=12)
            fig.tight_layout()
            self._save_plot(output_dir, f"{run_prefix}_{i}_Throughput_{prio.name}.png", fig)

        # --- 6. GENERAZIONE GRAFICI TASSO DI TIMEOUT (Grafici 7, 8, 9) ---
        print("--- Generazione grafici Tasso di Timeout ---")
        for i, prio in enumerate(config.Priority, 7):
            fig, ax = plt.subplots(figsize=(20, 8))
            self._setup_plot(ax, f"Tasso di Timeout (req/s) per Priorità {prio.name} - {run_prefix}", "Tasso di Timeout (req/s)", ylim_top=None) # Ylim auto

            # Calcolo e plot del tasso di timeout per ogni modello
            for model_name, metrics_obj in [('Baseline', self.metrics_base), ('Priorità', self.metrics_prio), ('WFQ', self.metrics_wfq)]:
                ts_list = [ts for ts, rt in metrics_obj.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == prio]
                times, rate = self._calculate_rate_histogram(ts_list, sim_time, TIME_WINDOW_SEC)
                if times is not None:
                    label = f"{model_name} (FIFO)" if model_name == 'Baseline' else model_name
                    ax.plot(times, rate, label=label, drawstyle='steps-post', **styles[model_name])

            ax_load = self._plot_load_profile(ax, load_times, load_values)
            self._combine_and_set_legend(ax, ax_load)
            fig.tight_layout()
            self._save_plot(output_dir, f"{run_prefix}_{i}_Timeout_Rate_{prio.name}.png", fig)