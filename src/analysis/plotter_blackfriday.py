# File: analysis/plotter_blackfriday.py (COMPLETO E CORRETTO)

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from src import config
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterBlackFriday:
    def __init__(self, metrics_base, metrics_prio, metrics_wfq, config_module):
        self.metrics_base = metrics_base
        self.metrics_prio = metrics_prio # Può essere None
        self.metrics_wfq = metrics_wfq
        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico finale salvato in: {save_path}")

    def generate_final_dashboards(self, output_dir, run_prefix, lambda_func):

        sim_time = self.config.SIMULATION_TIME
        TIME_WINDOW_STR = '10s' # Finestra più grande per simulazione lunga
        TIME_WINDOW_SEC = 10

        # --- FUNZIONI HELPER ---
        def get_time_based_moving_average(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('20s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()

        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                ax.plot(time_in_seconds, series.values, **kwargs)

        def _extract_seconds(data):
            if not data: return []
            return [float(p[0]) for p in data]

        # --- PREPARAZIONE DATI ---
        base_ma = get_time_based_moving_average(self.metrics_base.get_all_response_times_with_timestamps())

        prio_ma_data = {}
        if self.metrics_prio:
            prio_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_prio.response_times_history_by_prio.items()}

        wfq_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_wfq.response_times_history_by_prio.items()}

        # --- PREPARAZIONE LINEA DI CARICO COMPLESSA ---
        load_times = np.linspace(0, sim_time, num=2000)
        load_values = [lambda_func(t) for t in load_times]
        max_load = max(load_values) if load_values else 1

        # --- Grafico 1: Protezione QoS per Priorità HIGH ---
        fig1, ax1 = plt.subplots(figsize=(20, 8)); ax1.set_title(f"Protezione QoS per Priorità HIGH - {run_prefix}", fontsize=18)
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tempo Risposta Medio (s)"); ax1.grid(True); ax1.set_ylim(bottom=0, top=5.0)
        plot_series(ax1, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        if self.metrics_prio: plot_series(ax1, prio_ma_data.get(config.Priority.HIGH), color='darkred', lw=2, label='Priorità Strette (HIGH)')
        plot_series(ax1, wfq_ma_data.get(config.Priority.HIGH), color='limegreen', lw=2, label='DWFQ (HIGH)')
        ax_load1 = ax1.twinx(); ax_load1.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load1.set_ylabel("Carico (req/s)", color='gray'); ax_load1.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_QoS_HIGH.png", fig1)

        # --- Grafico 2: Analisi Starvation per Priorità LOW ---
        fig2, ax2 = plt.subplots(figsize=(20, 8)); ax2.set_title(f"Analisi Starvation per Priorità LOW - {run_prefix}", fontsize=18)
        ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Tempo Risposta Medio (s)"); ax2.grid(True); ax2.set_ylim(bottom=0, top=20.0)
        plot_series(ax2, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        if self.metrics_prio: plot_series(ax2, prio_ma_data.get(config.Priority.LOW), color='darkred', lw=2, label='Priorità Strette (LOW)')
        plot_series(ax2, wfq_ma_data.get(config.Priority.LOW), color='limegreen', lw=2, label='DWFQ (LOW)')
        ax_load2 = ax2.twinx(); ax_load2.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load2.set_ylabel("Carico (req/s)", color='gray'); ax_load2.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig2.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_Starvation_LOW.png", fig2)

        # --- Grafico 3: Performance Interna del WFQ ---
        fig3, ax3 = plt.subplots(figsize=(20, 8)); ax3.set_title(f"Performance Interna del DWFQ - {run_prefix}", fontsize=18)
        ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Tempo Risposta Medio (s)"); ax3.grid(True); ax3.set_ylim(bottom=0, top=20.0)
        colors = {config.Priority.HIGH: 'green', config.Priority.MEDIUM: 'orange', config.Priority.LOW: 'purple'}
        for prio in config.Priority:
            plot_series(ax3, wfq_ma_data.get(prio), color=colors.get(prio, 'black'), lw=2.5, label=f'DWFQ - {prio.name}')
        ax_load3 = ax3.twinx(); ax_load3.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load3.set_ylabel("Carico (req/s)", color='gray'); ax_load3.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax3.get_legend_handles_labels(); lines2, labels2 = ax_load3.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig3.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_3_WFQ_Internal_Performance.png", fig3)

        # --- Grafici Throughput (4, 5, 6) ---
        bins = np.arange(0, sim_time + TIME_WINDOW_SEC, TIME_WINDOW_SEC)
        def calculate_throughput_histogram(timestamps):
            if not timestamps: return None, None
            counts, _ = np.histogram(timestamps, bins=bins); throughput = counts / TIME_WINDOW_SEC
            return bins[:-1], throughput

        # Grafico 4: HIGH
        fig4, ax4 = plt.subplots(figsize=(20, 8)); ax4.set_title(f"Throughput (req/s) per Priorità HIGH - {run_prefix}", fontsize=18)
        ax4.set_xlabel("Tempo (s)"); ax4.set_ylabel("Throughput (req/s)"); ax4.grid(True); ax4.set_ylim(bottom=0,top = 100.0)
        base_ts_high = [ts for rt, p in config.REQUEST_TYPE_TO_PRIORITY.items() if p == config.Priority.HIGH for ts in _extract_seconds(self.metrics_base.response_times_history.get(rt,[]))]
        prio_ts_high = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.HIGH, [])) if self.metrics_prio else []
        wfq_ts_high = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.HIGH, []))
        t, tp = calculate_throughput_histogram(base_ts_high); ax4.plot(t, tp, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio: t, tp = calculate_throughput_histogram(prio_ts_high); ax4.plot(t, tp, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, tp = calculate_throughput_histogram(wfq_ts_high); ax4.plot(t, tp, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None
        ax_load4 = ax4.twinx(); ax_load4.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load4.set_ylabel("Carico (req/s)", color='gray'); ax_load4.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax4.get_legend_handles_labels(); lines2, labels2 = ax_load4.get_legend_handles_labels()
        ax4.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig4.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_4_Throughput_HIGH.png", fig4)

        # Grafico 5: MEDIUM
        fig5, ax5 = plt.subplots(figsize=(20, 8)); ax5.set_title(f"Throughput (req/s) per Priorità MEDIUM - {run_prefix}", fontsize=18)
        ax5.set_xlabel("Tempo (s)"); ax5.set_ylabel("Throughput (req/s)"); ax5.grid(True); ax5.set_ylim(bottom=0,top = 70.0)
        base_ts_medium = [ts for rt, p in config.REQUEST_TYPE_TO_PRIORITY.items() if p == config.Priority.MEDIUM for ts in _extract_seconds(self.metrics_base.response_times_history.get(rt,[]))]
        prio_ts_medium = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.MEDIUM, [])) if self.metrics_prio else []
        wfq_ts_medium  = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.MEDIUM, []))
        t, tp = calculate_throughput_histogram(base_ts_medium); ax5.plot(t, tp, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio: t, tp = calculate_throughput_histogram(prio_ts_medium); ax5.plot(t, tp, color='darkorange', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, tp = calculate_throughput_histogram(wfq_ts_medium); ax5.plot(t, tp, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None
        ax_load5 = ax5.twinx(); ax_load5.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load5.set_ylabel("Carico (req/s)", color='gray'); ax_load5.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax5.get_legend_handles_labels(); lines2, labels2 = ax_load5.get_legend_handles_labels()
        ax5.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig5.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_5_Throughput_MEDIUM.png", fig5)

        # Grafico 6: LOW
        fig6, ax6 = plt.subplots(figsize=(20, 8)); ax6.set_title(f"Throughput (req/s) per Priorità LOW - {run_prefix}", fontsize=18)
        ax6.set_xlabel("Tempo (s)"); ax6.set_ylabel("Throughput (req/s)"); ax6.grid(True); ax6.set_ylim(bottom=0, top = 70.0)
        base_ts_low = [ts for rt, p in config.REQUEST_TYPE_TO_PRIORITY.items() if p == config.Priority.LOW for ts in _extract_seconds(self.metrics_base.response_times_history.get(rt,[]))]
        prio_ts_low = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.LOW, [])) if self.metrics_prio else []
        wfq_ts_low  = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.LOW, []))
        t, tp = calculate_throughput_histogram(base_ts_low); ax6.plot(t, tp, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio: t, tp = calculate_throughput_histogram(prio_ts_low); ax6.plot(t, tp, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, tp = calculate_throughput_histogram(wfq_ts_low); ax6.plot(t, tp, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None
        ax_load6 = ax6.twinx(); ax_load6.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load6.set_ylabel("Carico (req/s)", color='gray'); ax_load6.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax6.get_legend_handles_labels(); lines2, labels2 = ax_load6.get_legend_handles_labels()
        ax6.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig6.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_6_Throughput_LOW.png", fig6)

        # --- Grafici Timeout (7, 8, 9) ---
        def calculate_rate_histogram(timestamps):
            if not timestamps: return None, None
            bins = np.arange(0, sim_time + TIME_WINDOW_SEC, TIME_WINDOW_SEC); counts, _ = np.histogram(timestamps, bins=bins)
            return bins[:-1], counts / TIME_WINDOW_SEC

        # Grafico 7: Timeout HIGH
        fig7, ax7 = plt.subplots(figsize=(20, 8)); ax7.set_title(f"Tasso di Timeout (req/s) per Priorità HIGH - {run_prefix}", fontsize=18)
        ax7.set_xlabel("Tempo (s)"); ax7.set_ylabel("Tasso di Timeout (req/s)"); ax7.grid(True); ax7.set_ylim(bottom=0,top = max_load * 1.1)
        base_ts_high_to = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH]
        prio_ts_high_to = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH] if self.metrics_prio else []
        wfq_ts_high_to = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH]
        t, r = calculate_rate_histogram(base_ts_high_to); ax7.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio: t, r = calculate_rate_histogram(prio_ts_high_to); ax7.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_high_to); ax7.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None
        ax_load7 = ax7.twinx(); ax_load7.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load7.set_ylabel("Carico (req/s)", color='gray'); ax_load7.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax7.get_legend_handles_labels(); lines2, labels2 = ax_load7.get_legend_handles_labels()
        ax7.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig7.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_7_Timeout_Rate_HIGH.png", fig7)

        # --- Grafico 8: Timeout MEDIUM ---
        fig8, ax8 = plt.subplots(figsize=(20, 8)); ax8.set_title(f"Tasso di Timeout (req/s) per Priorità MEDIUM - {run_prefix}", fontsize=18)
        ax8.set_xlabel("Tempo (s)"); ax8.set_ylabel("Tasso di Timeout (req/s)"); ax8.grid(True); ax8.set_ylim(bottom=0, top=max_load * 1.1)

        base_ts_medium_to = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.MEDIUM]
        prio_ts_medium_to = []
        if self.metrics_prio:
            prio_ts_medium_to = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.MEDIUM]
        wfq_ts_medium_to = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.MEDIUM]

        t, r = calculate_rate_histogram(base_ts_medium_to); ax8.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio:
            t, r = calculate_rate_histogram(prio_ts_medium_to); ax8.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_medium_to); ax8.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None

        ax_load8 = ax8.twinx(); ax_load8.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load8.set_ylabel("Carico (req/s)", color='gray'); ax_load8.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax8.get_legend_handles_labels(); lines2, labels2 = ax_load8.get_legend_handles_labels()
        ax8.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig8.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_8_Timeout_Rate_MEDIUM.png", fig8)

        # --- Grafico 9: Timeout LOW ---
        fig9, ax9 = plt.subplots(figsize=(20, 8)); ax9.set_title(f"Tasso di Timeout (req/s) per Priorità LOW - {run_prefix}", fontsize=18)
        ax9.set_xlabel("Tempo (s)"); ax9.set_ylabel("Tasso di Timeout (req/s)"); ax9.grid(True); ax9.set_ylim(bottom=0, top = max_load * 1.1)
        base_ts_low_to = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.LOW]
        prio_ts_low_to = []
        if self.metrics_prio:
            prio_ts_low_to = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.LOW]
        wfq_ts_low_to = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.LOW]

        t, r = calculate_rate_histogram(base_ts_low_to); ax9.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        if self.metrics_prio:
            t, r = calculate_rate_histogram(prio_ts_low_to); ax9.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_low_to); ax9.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None

        ax_load9 = ax9.twinx(); ax_load9.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load9.set_ylabel("Carico (req/s)", color='gray'); ax_load9.set_ylim(bottom=0, top=max_load * 1.1)
        lines, labels = ax9.get_legend_handles_labels(); lines2, labels2 = ax_load9.get_legend_handles_labels()
        ax9.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig9.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_9_Timeout_Rate_LOW.png", fig9)