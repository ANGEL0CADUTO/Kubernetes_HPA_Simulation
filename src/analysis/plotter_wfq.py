# File: src_analysis_plotter_wfq.txt (VERSIONE AGGIORNATA E COMPLETA)

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

class PlotterWFQ:
    def __init__(self, metrics_base_agg, metrics_prio_agg, metrics_wfq_agg,
                 metrics_per_worker_base, metrics_per_worker_prio, metrics_per_worker_wfq,
                 config_module):
        # Metriche aggregate
        self.metrics_base_agg = metrics_base_agg
        self.metrics_prio_agg = metrics_prio_agg
        self.metrics_wfq_agg = metrics_wfq_agg

        # Metriche per-worker
        self.metrics_per_worker_base = metrics_per_worker_base
        self.metrics_per_worker_prio = metrics_per_worker_prio
        self.metrics_per_worker_wfq = metrics_per_worker_wfq

        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def generate_final_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):
        print(f"\n--- [1/2] Generazione Dashboard Aggregati (Cluster) per la Replica: {run_prefix} ---")
        self._plot_cluster_aggregate_dashboards(output_dir, run_prefix, peak_start, peak_end, base_load, peak_load)

        print(f"\n--- [2/2] Generazione Dashboard Analisi Hotspot (Per-Worker) per la Replica: {run_prefix} ---")
        self._plot_hotspot_analysis_dashboards(output_dir, run_prefix, peak_start, peak_end, base_load, peak_load)

    # ==============================================================================
    # SEZIONE 1: DASHBOARD AGGREGATI A LIVELLO DI CLUSTER (INVARIATA)
    # ==============================================================================
    def _plot_cluster_aggregate_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):
        # ... (Questa sezione rimane identica a quella che mi hai fornito)
        self._plot_aggregated_response_time_trends(output_dir, run_prefix, peak_start, peak_end, base_load, peak_load)
        self._plot_aggregated_throughput_per_priority(output_dir, run_prefix)
        self._plot_aggregated_timeout_rates_per_priority(output_dir, run_prefix, peak_start, peak_end, base_load, peak_load)

    def _plot_aggregated_response_time_trends(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):
        # ... (codice invariato)
        sim_time = self.config.SIMULATION_TIME; TIME_WINDOW_STR = '10s'
        def get_ma(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0])); s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('1s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()
        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds(); ax.plot(time_in_seconds, series.values, **kwargs)
        base_history_high_only = []
        for req_type, history in self.metrics_base_agg.response_times_history.items():
            if self.config.REQUEST_TYPE_TO_PRIORITY.get(req_type) == config.Priority.HIGH: base_history_high_only.extend(history)
        base_history_high_only.sort(key=lambda x: x[0]); base_ma_high = get_ma(base_history_high_only)
        base_ma_all = get_ma(self.metrics_base_agg.get_all_response_times_with_timestamps())
        prio_ma_high = get_ma(self.metrics_prio_agg.response_times_history_by_prio.get(config.Priority.HIGH, [])); prio_ma_low = get_ma(self.metrics_prio_agg.response_times_history_by_prio.get(config.Priority.LOW, []))
        wfq_ma_high = get_ma(self.metrics_wfq_agg.response_times_history_by_prio.get(config.Priority.HIGH, [])); wfq_ma_low = get_ma(self.metrics_wfq_agg.response_times_history_by_prio.get(config.Priority.LOW, []))
        fig1, ax1 = plt.subplots(figsize=(20, 8)); ax1.set_title(f"Protezione QoS Aggregata (Cluster) per Priorità HIGH - {run_prefix}", fontsize=18)
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tempo Risposta Medio (s)"); ax1.grid(True); ax1.set_ylim(bottom=0, top=15.0)
        plot_series(ax1, base_ma_high, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Solo HIGH'); plot_series(ax1, prio_ma_high, color='darkred', lw=2, label='Priorità Strette - Solo HIGH'); plot_series(ax1, wfq_ma_high, color='limegreen', lw=2, label='WFQ - Solo HIGH')
        ax_load1 = ax1.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]; load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load1.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Applicato'); ax_load1.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels(); ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_agg_QoS_HIGH.png", fig1)
        fig2, ax2 = plt.subplots(figsize=(20, 8)); ax2.set_title(f"Analisi Starvation Aggregata (Cluster) per Priorità LOW - {run_prefix}", fontsize=18)
        ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Tempo Risposta Medio (s)"); ax2.grid(True); ax2.set_ylim(bottom=0, top=15.0)
        plot_series(ax2, base_ma_all, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Media Totale'); plot_series(ax2, prio_ma_low, color='darkred', lw=2, label='Priorità Strette - Solo LOW'); plot_series(ax2, wfq_ma_low, color='limegreen', lw=2, label='WFQ - Solo LOW')
        ax_load2 = ax2.twinx(); ax_load2.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Applicato'); ax_load2.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load2.get_legend_handles_labels(); ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig2.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_agg_Starvation_LOW.png", fig2)

    def _plot_aggregated_throughput_per_priority(self, output_dir, run_prefix):
        # ... (codice invariato)
        fig, ax = plt.subplots(figsize=(16, 9)); plot_data = []
        base_served_by_prio = defaultdict(int)
        for req_type, req_list in self.metrics_base_agg.response_times_data.items():
            prio = self.config.REQUEST_TYPE_TO_PRIORITY[req_type]; base_served_by_prio[prio] += len(req_list)
        for prio in config.Priority:
            plot_data.append({'Priorità': prio.name, 'Conteggio': base_served_by_prio[prio], 'Scenario': 'Baseline (FIFO)'})
            plot_data.append({'Priorità': prio.name, 'Conteggio': self.metrics_prio_agg.requests_completed_by_priority[prio], 'Scenario': 'Priorità Strette'})
            plot_data.append({'Priorità': prio.name, 'Conteggio': self.metrics_wfq_agg.requests_completed_by_priority[prio], 'Scenario': 'WFQ'})
        df = pd.DataFrame(plot_data)
        sns.barplot(data=df, x='Priorità', y='Conteggio', hue='Scenario', order=[p.name for p in config.Priority], hue_order=['Baseline (FIFO)', 'Priorità Strette', 'WFQ'], palette={'Baseline (FIFO)': 'royalblue', 'Priorità Strette': 'darkred', 'WFQ': 'limegreen'}, ax=ax)
        ax.set_title(f"Throughput Aggregato (Cluster) per Priorità - {run_prefix}", fontsize=18); ax.set_xlabel("Classe di Priorità", fontsize=14); ax.set_ylabel("Numero Totale di Richieste Servite", fontsize=14)
        for container in ax.containers: ax.bar_label(container, fmt='%d', padding=3, fontsize=9)
        ax.legend(title='Scenario'); ax.grid(True, axis='y', linestyle='--', alpha=0.6); fig.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_3_agg_throughput_per_priority.png", fig)

    def _plot_aggregated_timeout_rates_per_priority(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):
        # ... (codice invariato)
        fig, axes = plt.subplots(3, 1, figsize=(20, 24), sharex=True, sharey=True); sim_time = self.config.SIMULATION_TIME; TIME_WINDOW_SEC = 10
        bins = np.arange(0, sim_time + TIME_WINDOW_SEC, TIME_WINDOW_SEC)
        def get_timeout_rate(timeout_history, priority_filter):
            timestamps = [ts for ts, req in timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(getattr(req, 'req_type', req), None) == priority_filter]
            if not timestamps: return None, None
            counts, _ = np.histogram(timestamps, bins=bins); return bins[:-1], counts / TIME_WINDOW_SEC
        priorities = [config.Priority.HIGH, config.Priority.MEDIUM, config.Priority.LOW]
        for i, prio in enumerate(priorities):
            ax = axes[i]
            t_base, r_base = get_timeout_rate(self.metrics_base_agg.timeout_history, prio)
            if t_base is not None: ax.plot(t_base, r_base, color='royalblue', label='Baseline (FIFO)', drawstyle='steps-post', lw=2.5, linestyle='--')
            t_prio, r_prio = get_timeout_rate(self.metrics_prio_agg.timeout_history, prio)
            if t_prio is not None: ax.plot(t_prio, r_prio, color='darkred', label='Priorità Strette', drawstyle='steps-post', lw=2)
            t_wfq, r_wfq = get_timeout_rate(self.metrics_wfq_agg.timeout_history, prio)
            if t_wfq is not None: ax.plot(t_wfq, r_wfq, color='limegreen', label='WFQ', drawstyle='steps-post', lw=2.5)
            ax.set_title(f"Tasso di Timeout Aggregato (Cluster) per Priorità {prio.name}", fontsize=16); ax.set_ylabel("Timeout / sec"); ax.grid(True); ax.legend(); ax.set_ylim(bottom=0)
            ax_load = ax.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]; load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
            ax_load.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Applicato'); ax_load.set_ylabel("Carico (req/s)", color='gray'); ax_load.set_ylim(bottom=0)
        axes[-1].set_xlabel("Tempo (s)"); fig.suptitle(f"Analisi Tasso di Perdita Aggregato (Cluster) - {run_prefix}", fontsize=20, fontweight='bold'); fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, f"{run_prefix}_4_agg_timeout_rate.png", fig)

    # ==============================================================================
    # SEZIONE 2: DASHBOARD DI ANALISI HOTSPOT (PER-WORKER) --- MODIFICATA
    # ==============================================================================

    def _plot_hotspot_analysis_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):
        # --- NUOVO GRAFICO DI DIVERGENZA DELLE CODE ---
        # Chiamiamo il nuovo metodo per ogni sistema per creare 3 grafici separati
        self._plot_worker_queue_divergence(
            metrics_per_worker=self.metrics_per_worker_base,
            system_name="Baseline (FIFO)",
            run_prefix=run_prefix,
            output_dir=output_dir,
            overlay_pods=True
        )
        self._plot_worker_queue_divergence(
            metrics_per_worker=self.metrics_per_worker_prio,
            system_name="Priorita Strette",
            run_prefix=run_prefix,
            output_dir=output_dir,
            overlay_pods=True
        )
        self._plot_worker_queue_divergence(
            metrics_per_worker=self.metrics_per_worker_wfq,
            system_name="WFQ",
            run_prefix=run_prefix,
            output_dir=output_dir,
            overlay_pods=True
        )

        # --- GRAFICI ESISTENTI (INVARIATI) ---
        worker_id_to_analyze = 0; sim_time = self.config.SIMULATION_TIME
        metrics_w0_base = self.metrics_per_worker_base[worker_id_to_analyze]; metrics_w0_prio = self.metrics_per_worker_prio[worker_id_to_analyze]; metrics_w0_wfq = self.metrics_per_worker_wfq[worker_id_to_analyze]
        fig1, ax1 = plt.subplots(figsize=(20, 8)); ax1.set_title(f"Evoluzione Coda Locale su Hotspot (Worker {worker_id_to_analyze}) - {run_prefix}", fontsize=18)
        if metrics_w0_base.queue_length_history: times, lengths = zip(*metrics_w0_base.queue_length_history); ax1.plot(times, lengths, label='Baseline (FIFO)', color='royalblue', lw=2)
        if metrics_w0_prio.queue_lengths: ax1.plot(metrics_w0_prio.timestamps, metrics_w0_prio.queue_lengths, label='Priorità Strette', color='darkred', lw=2)
        if metrics_w0_wfq.queue_lengths: ax1.plot(metrics_w0_wfq.timestamps, metrics_w0_wfq.queue_lengths, label='WFQ', color='limegreen', lw=2)
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("N. Richieste in Coda Locale"); ax1.grid(True); ax1.legend(); ax1.set_yscale('log'); ax1.set_ylim(bottom=1)
        self._save_plot(output_dir, f"{run_prefix}_5_hotspot_queue_length.png", fig1)
        fig2, ax2 = plt.subplots(figsize=(20, 8)); ax2.set_title(f"Protezione QoS su Hotspot (Worker {worker_id_to_analyze}) - {run_prefix}", fontsize=18)
        TIME_WINDOW_STR = '10s'
        def get_ma(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0])); s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('1s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()
        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds(); ax.plot(time_in_seconds, series.values, **kwargs)
        base_w0_history_high_only = []
        for req_type, history in metrics_w0_base.response_times_history.items():
            if self.config.REQUEST_TYPE_TO_PRIORITY.get(req_type) == config.Priority.HIGH: base_w0_history_high_only.extend(history)
        base_w0_history_high_only.sort(key=lambda x: x[0]); base_ma_w0_high = get_ma(base_w0_history_high_only)
        prio_ma_high_w0 = get_ma(metrics_w0_prio.response_times_history_by_prio.get(config.Priority.HIGH, [])); wfq_ma_high_w0 = get_ma(metrics_w0_wfq.response_times_history_by_prio.get(config.Priority.HIGH, []))
        plot_series(ax2, base_ma_w0_high, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Solo HIGH'); plot_series(ax2, prio_ma_high_w0, color='darkred', lw=2, label='Priorità Strette - Solo HIGH'); plot_series(ax2, wfq_ma_high_w0, color='limegreen', lw=2, label='WFQ - Solo HIGH')
        ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Tempo Risposta Medio Locale (s)"); ax2.grid(True); ax2.legend(); ax2.set_ylim(bottom=0, top=15.0)
        ax_load = ax2.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]; load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Totale Applicato al Cluster'); ax_load.set_ylabel("Carico (req/s)", color='gray');
        lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load.get_legend_handles_labels(); ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        fig2.tight_layout(); self._save_plot(output_dir, f"{run_prefix}_6_hotspot_QoS_HIGH.png", fig2)

    # --- NUOVO METODO HELPER PER IL GRAFICO DI DIVERGENZA ---
    def _plot_worker_queue_divergence(
            self,
            metrics_per_worker: list,
            system_name: str,
            run_prefix: str,
            output_dir: str,
            overlay_pods: bool = True
    ):
        """
        Crea un grafico che mostra l'evoluzione temporale della lunghezza della coda
        per ogni Worker Node per un dato sistema (es. Baseline, WFQ).
        """
        print(f"Generazione grafico evoluzione code worker per sistema '{system_name}'...")

        if not metrics_per_worker:
            print(f"  -> Dati per-worker non disponibili per '{system_name}'. Grafico saltato.")
            return

        num_workers = len(metrics_per_worker)
        fig, ax1 = plt.subplots(figsize=(16, 8))
        fig.suptitle(f'Evoluzione Code Worker - Sistema: {system_name} - {run_prefix}',
                     fontsize=18, fontweight='bold')

        ax1.set_xlabel('Tempo di Simulazione (s)', fontsize=14)
        ax1.set_ylabel('N. Richieste in Coda (Scala Log)', fontsize=14, color='black')
        ax1.set_yscale('log'); ax1.set_ylim(bottom=1)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        colors = sns.color_palette("husl", n_colors=num_workers)

        for i in range(num_workers):
            metrics = metrics_per_worker[i]
            if hasattr(metrics, 'queue_length_history') and metrics.queue_length_history:
                timestamps, lengths = zip(*metrics.queue_length_history)
                ax1.plot(timestamps, lengths, label=f'Coda Worker {i}', color=colors[i], linewidth=2.5)
            # Adattamento per MetricsWithPriority
            elif hasattr(metrics, 'queue_lengths') and metrics.queue_lengths:
                ax1.plot(metrics.timestamps, metrics.queue_lengths, label=f'Coda Worker {i}', color=colors[i], linewidth=2.5)


        ax2 = None
        if overlay_pods:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Numero di Pod Attivi', fontsize=14, color='dimgray')
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.tick_params(axis='y', labelcolor='dimgray')
            max_pods_config = getattr(self.config, 'MAX_PODS', 50)
            ax2.set_ylim(bottom=0, top=max_pods_config * 1.1)

            for i in range(num_workers):
                metrics = metrics_per_worker[i]
                if hasattr(metrics, 'pod_count_history') and metrics.pod_count_history:
                    timestamps, counts = zip(*metrics.pod_count_history)
                    ax2.plot(timestamps, counts, label=f'Pod Worker {i}', color=colors[i],
                             linestyle=':', linewidth=2, alpha=0.8)
                # Adattamento per MetricsWithPriority
                elif hasattr(metrics, 'pod_counts') and metrics.pod_counts:
                    ax2.plot(metrics.timestamps, metrics.pod_counts, label=f'Pod Worker {i}', color=colors[i],
                             linestyle=':', linewidth=2, alpha=0.8)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
        all_labels = labels1 + labels2; all_lines = lines1 + lines2
        if all_labels:
            sorted_legend = sorted(zip(all_labels, all_lines), key=lambda x: x[0].split(' ')[-1])
            sorted_labels, sorted_lines = zip(*sorted_legend)
            ax1.legend(sorted_lines, sorted_labels, loc='upper left', fontsize=12, title="Metriche per Worker")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        filename_system = system_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename_suffix = "pods_overlay" if overlay_pods else "queues_only"
        filename = f"{run_prefix}_7_worker_divergence_{filename_system}_{filename_suffix}.png"

        self._save_plot(output_dir, filename, fig)