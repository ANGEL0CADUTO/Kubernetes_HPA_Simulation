

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from src import config
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterSPS:
    def __init__(self, metrics_base, metrics_prio, metrics_sps, config_module):
        self.metrics_base = metrics_base
        self.metrics_prio = metrics_prio
        self.metrics_sps = metrics_sps
        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico finale salvato in: {save_path}")



    def generate_final_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):

        sim_time = self.config.SIMULATION_TIME
        TIME_WINDOW = '10s'

        def get_time_based_moving_average(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            # Resample ogni secondo e poi calcola la media mobile per evitare linee ingannevoli
            return s.resample('1s').mean().rolling(window=TIME_WINDOW, min_periods=1).mean()

        # Preparazione Dati Corretta
        base_ma = get_time_based_moving_average(self.metrics_base.get_all_response_times_with_timestamps())
        prio_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_prio.response_times_history_by_prio.items()}
        sps_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_sps.response_times_history_by_prio.items()}

        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                ax.plot(time_in_seconds, series.values, **kwargs)

        # --- Grafico 1: Protezione QoS per Priorità HIGH (Ripristinato) ---
        fig1, ax1 = plt.subplots(figsize=(20, 8)); ax1.set_title(f"Protezione QoS per Priorità HIGH - {run_prefix}", fontsize=18)
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tempo Risposta Medio (s)"); ax1.grid(True); ax1.set_ylim(bottom=0, top=20.0)
        plot_series(ax1, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        plot_series(ax1, prio_ma_data.get(config.Priority.HIGH), color='darkred', lw=2, label='Priorità Strette (HIGH)')
        plot_series(ax1, sps_ma_data.get(config.Priority.HIGH), color='limegreen', lw=2, label='SPS (HIGH)')
        ax_load1 = ax1.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]
        load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load1.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load1.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_QoS_HIGH.png", fig1)

        # --- Grafico 2: Analisi Starvation per Priorità LOW (Ripristinato) ---
        fig2, ax2 = plt.subplots(figsize=(20, 8)); ax2.set_title(f"Analisi Starvation per Priorità LOW - {run_prefix}", fontsize=18)
        ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Tempo Risposta Medio (s)"); ax2.grid(True); ax2.set_ylim(bottom=0, top=20.0)
        plot_series(ax2, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        plot_series(ax2, prio_ma_data.get(config.Priority.LOW), color='darkred', lw=2, label='Priorità Strette (LOW)')
        plot_series(ax2, sps_ma_data.get(config.Priority.LOW), color='limegreen', lw=2, label='SPS (LOW)')
        ax_load2 = ax2.twinx(); ax_load2.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load2.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig2.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_Starvation_LOW.png", fig2)

        # --- Grafico 3: Performance Interna del SPS (Aggiuntivo) ---
        fig3, ax3 = plt.subplots(figsize=(20, 8)); ax3.set_title(f"Performance Interna del SPS - {run_prefix}", fontsize=18)
        ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Tempo Risposta Medio (s)"); ax3.grid(True); ax3.set_ylim(bottom=0, top=20.0)
        colors = {config.Priority.HIGH: 'green', config.Priority.MEDIUM: 'orange', config.Priority.LOW: 'purple'}
        for prio in config.Priority:
            plot_series(ax3, sps_ma_data.get(prio), color=colors.get(prio, 'black'), lw=2, label=f'SPS - {prio.name}')
        ax_load3 = ax3.twinx(); ax_load3.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load3.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax3.get_legend_handles_labels(); lines2, labels2 = ax_load3.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig3.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_3_SPS_Internal_Performance.png", fig3)