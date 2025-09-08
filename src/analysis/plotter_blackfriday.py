# File: analysis/plotter_blackfriday.py

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from src import config
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterBlackFriday:
    def __init__(self, metrics_base_agg, metrics_prio_agg, metrics_wfq_agg,
                 metrics_per_worker_base, metrics_per_worker_prio, metrics_per_worker_wfq,
                 config_module):
        self.metrics_base_agg = metrics_base_agg
        self.metrics_prio_agg = metrics_prio_agg # Potrebbe essere None
        self.metrics_wfq_agg = metrics_wfq_agg
        self.metrics_per_worker_base = metrics_per_worker_base
        self.metrics_per_worker_prio = metrics_per_worker_prio # Potrebbe essere None
        self.metrics_per_worker_wfq = metrics_per_worker_wfq
        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def generate_final_dashboards(self, output_dir, run_prefix, lambda_func):
        print(f"\n--- [1/2] Generazione Dashboard Aggregati (Cluster) per la Replica: {run_prefix} ---")
        self._plot_cluster_aggregate_dashboards(output_dir, run_prefix, lambda_func)

        print(f"\n--- [2/2] Generazione Dashboard Analisi Hotspot (Worker 0) per la Replica: {run_prefix} ---")
        self._plot_hotspot_analysis_dashboards(output_dir, run_prefix, lambda_func)

    def _plot_cluster_aggregate_dashboards(self, output_dir, run_prefix, lambda_func):
        sim_time = self.config.SIMULATION_TIME
        TIME_WINDOW_STR = '60s' # Finestra più grande per simulazione lunga

        def get_ma(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('10s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()

        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                ax.plot(time_in_seconds, series.values, **kwargs)

        base_history_high_only = [h for rt, h_list in self.metrics_base_agg.response_times_history.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH for h in h_list]
        base_ma_high = get_ma(base_history_high_only)
        wfq_ma_high = get_ma(self.metrics_wfq_agg.response_times_history_by_prio.get(config.Priority.HIGH, []))

        fig1, ax1 = plt.subplots(figsize=(20, 8))
        ax1.set_title(f"Protezione QoS Aggregata (Cluster) per Priorità HIGH - {run_prefix}", fontsize=18)
        plot_series(ax1, base_ma_high, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Solo HIGH')
        plot_series(ax1, wfq_ma_high, color='limegreen', lw=2, label='WFQ - Solo HIGH')

        ax_load1 = ax1.twinx(); load_times = np.linspace(0, sim_time, num=2000); load_values = [lambda_func(t) for t in load_times]
        ax_load1.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Applicato')

        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tempo Risposta Medio (s)"); ax1.grid(True); ax1.set_ylim(bottom=0, top=5.0)
        ax_load1.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_agg_QoS_HIGH.png", fig1)

    def _plot_hotspot_analysis_dashboards(self, output_dir, run_prefix, lambda_func):
        worker_id_to_analyze = 0
        sim_time = self.config.SIMULATION_TIME
        metrics_w0_base = self.metrics_per_worker_base[worker_id_to_analyze]
        metrics_w0_wfq = self.metrics_per_worker_wfq[worker_id_to_analyze]

        fig, ax = plt.subplots(figsize=(20, 8))
        ax.set_title(f"Protezione QoS su Hotspot (Worker {worker_id_to_analyze}) - {run_prefix}", fontsize=18)

        TIME_WINDOW_STR = '60s'
        def get_ma(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('10s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()

        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                ax.plot(time_in_seconds, series.values, **kwargs)

        base_w0_history_high_only = [h for rt, h_list in metrics_w0_base.response_times_history.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH for h in h_list]
        base_ma_w0_high = get_ma(base_w0_history_high_only)
        wfq_ma_high_w0 = get_ma(metrics_w0_wfq.response_times_history_by_prio.get(config.Priority.HIGH, []))

        plot_series(ax, base_ma_w0_high, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Solo HIGH')
        plot_series(ax, wfq_ma_high_w0, color='limegreen', lw=2, label='WFQ - Solo HIGH')

        ax.set_xlabel("Tempo (s)"); ax.set_ylabel("Tempo Risposta Medio Locale (s)"); ax.grid(True); ax.legend()
        ax.set_ylim(bottom=0, top=5.0)

        ax_load = ax.twinx()
        load_times = np.linspace(0, sim_time, num=2000); load_values = [lambda_func(t) for t in load_times]
        ax_load.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Totale Applicato al Cluster')
        ax_load.set_ylabel("Carico (req/s)", color='gray');

        lines, labels = ax.get_legend_handles_labels(); lines2, labels2 = ax_load.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper left')

        fig.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_hotspot_QoS_HIGH.png", fig)