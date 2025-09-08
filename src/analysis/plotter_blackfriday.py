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
# MODIFICA: Import mancanti aggiunti per usare isinstance()
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority

matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterBlackFriday:
    def __init__(self, metrics_base_agg, metrics_prio_agg, metrics_wfq_agg,
                 metrics_per_worker_base, metrics_per_worker_prio, metrics_per_worker_wfq,
                 config_module):
        self.metrics_base_agg = metrics_base_agg
        self.metrics_prio_agg = metrics_prio_agg
        self.metrics_wfq_agg = metrics_wfq_agg
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

    def generate_final_dashboards(self, output_dir, run_prefix, lambda_func):
        """
        Orchestratore principale che genera tutti i set di grafici.
        """
        print(f"\n--- [1/4] Generazione Dashboard Aggregati (Cluster) ---")
        self._plot_cluster_aggregate_dashboards(output_dir, run_prefix, lambda_func)

        print(f"\n--- [2/4] Generazione Dashboard Analisi Hotspot (Worker 0) ---")
        self._plot_hotspot_analysis_dashboards(output_dir, run_prefix, lambda_func)

        print(f"\n--- [3/4] Generazione Grafici Diagnostici di Evoluzione dei Worker ---")
        self._plot_worker_queue_evolution(output_dir, run_prefix)
        self._plot_tradeoff_dashboard(output_dir, run_prefix)

        print(f"\n--- [4/4] Generazione Grafici Diagnostici Avanzati ---")
        self._plot_latency_heatmap(output_dir, run_prefix)
        self._plot_hpa_scaling_trace(output_dir, run_prefix)
        self._plot_cumulative_loss_trace(output_dir, run_prefix)

    def _plot_cluster_aggregate_dashboards(self, output_dir, run_prefix, lambda_func):
        sim_time = self.config.SIMULATION_TIME
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

    def _plot_worker_queue_evolution(self, output_dir, run_prefix):
        """
        MODIFICATO: Semplificata la gestione dei timestamp per garantire l'allineamento
        e la corretta visualizzazione degli assi.
        """
        scenarios = {
            'Baseline (FIFO)': self.metrics_per_worker_base,
            'DWFQ': self.metrics_per_worker_wfq
        }

        for scenario_name, metrics_per_worker in scenarios.items():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 25), sharex=True)
            fig.suptitle(f'Evoluzione Code e Pod per Worker - Scenario: {scenario_name} - {run_prefix}',
                         fontsize=18, fontweight='bold')

            # --- Sottografico 1: Lunghezza delle Code ---
            ax1.set_ylabel('N. Richieste in Coda (Media per 30s)', fontsize=14)
            ax1.set_yscale('log')
            ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax1.set_title("Andamento Lunghezza Code Locali (Identificazione Hotspot)", fontsize=16)
            ax1.set_ylim(bottom=1, top=10000)

            num_workers = len(metrics_per_worker)
            colors = sns.color_palette("husl", n_colors=num_workers)

            for i in range(num_workers):
                metrics = metrics_per_worker[i]
                timestamps, lengths = [], []
                if isinstance(metrics, (Metrics, MetricsWithPriority)):
                    # La logica di estrazione dei dati grezzi rimane la stessa
                    if hasattr(metrics, 'queue_length_history') and metrics.queue_length_history:
                        timestamps, lengths = zip(*metrics.queue_length_history)
                    elif hasattr(metrics, 'timestamps') and metrics.timestamps:
                        timestamps, lengths = metrics.timestamps, metrics.queue_lengths

                if timestamps and lengths:
                    s = pd.Series(lengths, index=pd.to_datetime(timestamps, unit='s'))
                    resampled_mean = s.resample('30S').mean()
                    resampled_max = s.resample('30S').max()

                    # MODIFICA: Plotta direttamente usando l'indice DatetimeIndex di Pandas.
                    # Per l'asse X, usiamo i valori numerici dei timestamp.
                    time_values = (resampled_mean.index - pd.to_datetime(0, unit='s')).total_seconds()
                    ax1.plot(time_values, resampled_mean.values, label=f'Coda Worker {i} (Media)', color=colors[i], linewidth=2.5)
                    ax1.fill_between(time_values, resampled_mean.values, resampled_max.values, color=colors[i], alpha=0.2)

            # --- Sottografico 2: Numero di Pod ---
            ax2.set_ylabel('Numero di Pod Attivi', fontsize=14)
            ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax2.set_xlabel('Tempo di Simulazione (s)', fontsize=14)
            ax2.set_title("Reazione dell'HPA (Scaling Mirato)", fontsize=16)
            ax2.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)

            for i in range(num_workers):
                metrics = metrics_per_worker[i]
                timestamps, pod_counts = [], []
                if isinstance(metrics, (Metrics, MetricsWithPriority)):
                    if hasattr(metrics, 'pod_count_history') and metrics.pod_count_history:
                        timestamps, pod_counts = zip(*metrics.pod_count_history)
                    elif hasattr(metrics, 'timestamps') and metrics.timestamps:
                        timestamps, pod_counts = metrics.timestamps, metrics.pod_counts

                if timestamps and pod_counts:
                    ax2.plot(timestamps, pod_counts, label=f'Pod Worker {i}', color=colors[i], linestyle='-', drawstyle='steps-post', linewidth=2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12, title="Metriche per Worker")
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            filename = f"3_worker_evolution_{scenario_name.replace(' ', '_').lower()}.png"
            self._save_plot(output_dir, f"{run_prefix}_{filename}", fig)


    def _plot_tradeoff_dashboard(self, output_dir, run_prefix):
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=False)
        fig.suptitle(f'Analisi dei Trade-Off per Priorità (Cluster Aggregato) - {run_prefix}',
                     fontsize=18, fontweight='bold')

        ax1 = axes[0]
        served_data = []
        base_served_by_prio = defaultdict(int)
        for rt, h in self.metrics_base_agg.response_times_history.items():
            prio = self.config.REQUEST_TYPE_TO_PRIORITY[rt]
            base_served_by_prio[prio] += len(h)

        for prio in config.Priority:
            served_data.append({'Priorità': prio.name, 'Conteggio': base_served_by_prio[prio], 'Scenario': 'Baseline (FIFO)'})
            served_data.append({'Priorità': prio.name, 'Conteggio': self.metrics_wfq_agg.requests_completed_by_priority[prio], 'Scenario': 'DWFQ'})

        df_served = pd.DataFrame(served_data)
        sns.barplot(data=df_served, x='Priorità', y='Conteggio', hue='Scenario',
                    order=[p.name for p in config.Priority],
                    palette={'Baseline (FIFO)': 'royalblue', 'DWFQ': 'limegreen'}, ax=ax1)
        ax1.set_title('Richieste Servite con Successo', fontsize=16)
        ax1.set_ylabel('Numero Totale di Richieste Servite', fontsize=12)
        ax1.set_xlabel('Classe di Priorità', fontsize=12)
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=3, fontsize=10)
        ax1.legend().set_title("Scenario")
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax2 = axes[1]
        lost_data = []
        base_lost_by_prio = defaultdict(int)
        # CORREZIONE: usiamo getattr per sicurezza
        for ts, req_obj in self.metrics_base_agg.timeout_history:
            req_type = getattr(req_obj, 'req_type', req_obj) # Gestisce sia oggetti che enum
            prio = self.config.REQUEST_TYPE_TO_PRIORITY[req_type]
            base_lost_by_prio[prio] += 1

        for prio in config.Priority:
            lost_data.append({'Priorità': prio.name, 'Conteggio': base_lost_by_prio[prio], 'Scenario': 'Baseline (FIFO)'})
            lost_data.append({'Priorità': prio.name, 'Conteggio': self.metrics_wfq_agg.requests_timed_out_by_priority[prio], 'Scenario': 'DWFQ'})

        df_lost = pd.DataFrame(lost_data)
        sns.barplot(data=df_lost, x='Priorità', y='Conteggio', hue='Scenario',
                    order=[p.name for p in config.Priority],
                    palette={'Baseline (FIFO)': 'royalblue', 'DWFQ': 'limegreen'}, ax=ax2)
        ax2.set_title('Richieste Perse (Timeout o Shedding)', fontsize=16)
        ax2.set_ylabel('Numero Totale di Richieste Perse', fontsize=12)
        ax2.set_xlabel('Classe di Priorità', fontsize=12)
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%d', padding=3, fontsize=10)
        ax2.legend().set_title("Scenario")
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

        y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(top=y_max * 1.1)
        ax2.set_ylim(top=y_max * 1.1)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, f"{run_prefix}_4_tradeoff_dashboard.png", fig)



    # ==============================================================================
    # NUOVI METODI DI PLOTTING AVANZATI
    # ==============================================================================

    def _plot_latency_heatmap(self, output_dir, run_prefix):
        """
        NUOVO: Genera una heatmap per confrontare la latenza media delle richieste HIGH
        su ogni worker per ogni scenario.
        """
        heatmap_data = []
        scenarios = {
            'Baseline (FIFO)': self.metrics_per_worker_base,
            'DWFQ': self.metrics_per_worker_wfq
        }

        for scenario_name, metrics_per_worker in scenarios.items():
            for i, worker_metrics in enumerate(metrics_per_worker):

                # Estrai i tempi di risposta solo per le richieste HIGH
                high_responses = []
                if isinstance(worker_metrics, MetricsWithPriority):
                    high_responses.extend(worker_metrics.response_times_by_priority.get(config.Priority.HIGH, []))
                elif isinstance(worker_metrics, Metrics):
                    for rt, history in worker_metrics.response_times_history.items():
                        if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH:
                            high_responses.extend([val for ts, val in history])

                mean_latency = np.mean(high_responses) if high_responses else 0
                heatmap_data.append({
                    'Scenario': scenario_name,
                    'Worker': f'Worker {i}',
                    'Latenza Media HIGH (s)': mean_latency
                })

        df = pd.DataFrame(heatmap_data)
        pivot_df = df.pivot(index='Scenario', columns='Worker', values='Latenza Media HIGH (s)')

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", linewidths=.5, cmap="coolwarm_r", ax=ax)
        ax.set_title(f'Heatmap Latenza Richieste HIGH per Worker - {run_prefix}', fontsize=16)
        self._save_plot(output_dir, f"{run_prefix}_5_latency_heatmap.png", fig)

    def _plot_hpa_scaling_trace(self, output_dir, run_prefix):
        """
        NUOVO: Genera un grafico a barre impilate per mostrare la distribuzione
        dei pod tra i worker nel tempo.
        """
        scenarios = {
            'Baseline (FIFO)': self.metrics_per_worker_base,
            'DWFQ': self.metrics_per_worker_wfq
        }

        for scenario_name, metrics_per_worker in scenarios.items():
            all_data = []
            for i, worker_metrics in enumerate(metrics_per_worker):
                history = []
                if isinstance(worker_metrics, MetricsWithPriority):
                    if worker_metrics.timestamps and worker_metrics.pod_counts:
                        history = list(zip(worker_metrics.timestamps, worker_metrics.pod_counts))
                elif isinstance(worker_metrics, Metrics):
                    history = worker_metrics.pod_count_history

                if history:
                    df_worker = pd.DataFrame(history, columns=['timestamp', f'Worker {i}'])
                    df_worker['timestamp'] = pd.to_datetime(df_worker['timestamp'], unit='s')
                    df_worker = df_worker.set_index('timestamp')
                    all_data.append(df_worker)

            if not all_data: continue

            # Concatena i dati e riempie i valori mancanti per creare una serie temporale continua
            combined_df = pd.concat(all_data, axis=1).ffill().fillna(0)

            fig, ax = plt.subplots(figsize=(20, 12))
            # Usa il plotting integrato di Pandas per barre impilate
            combined_df.plot(kind='bar', stacked=True, ax=ax, width=1.0,
                             color=sns.color_palette("husl", n_colors=len(metrics_per_worker)))

            ax.set_title(f'Traccia di Scaling HPA per Worker - Scenario: {scenario_name} - {run_prefix}', fontsize=16)
            ax.set_ylabel('Numero Totale di Pod Attivi', fontsize=12)
            ax.set_xlabel('Tempo di Simulazione (s)', fontsize=12)

            # Semplifica le etichette dell'asse X per evitare sovraffollamento
            tick_labels = [item.get_text() for item in ax.get_xticklabels()]
            simplified_ticks = [label.split(' ')[1].split('.')[0] for label in tick_labels]
            ax.set_xticklabels(simplified_ticks, rotation=90)
            # Mostra un tick ogni N per leggibilità
            n = max(1, len(simplified_ticks) // 50)
            [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]

            ax.legend(title='Worker')
            self._save_plot(output_dir, f"{run_prefix}_6_hpa_scaling_trace_{scenario_name.replace(' ', '_').lower()}.png", fig)

    def _plot_cumulative_loss_trace(self, output_dir, run_prefix):
        """
        NUOVO: Mostra il numero cumulativo di richieste perse nel tempo,
        distinte per priorità e scenario.
        """
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.set_title(f'Traccia Cumulativa Richieste Perse per Priorità - {run_prefix}', fontsize=16)

        scenarios = {
            'Baseline (FIFO)': (self.metrics_base_agg, '-'),
            'DWFQ': (self.metrics_wfq_agg, '--')
        }
        colors = {config.Priority.HIGH: 'red', config.Priority.MEDIUM: 'orange', config.Priority.LOW: 'gray'}

        for scenario_name, (metrics, linestyle) in scenarios.items():
            for prio in config.Priority:

                history = metrics.timeout_history
                # Filtra i timeout per la priorità corrente
                timestamps = sorted([ts for ts, req in history if self.config.REQUEST_TYPE_TO_PRIORITY.get(getattr(req, 'req_type', req), None) == prio])

                if timestamps:
                    cumulative_counts = np.arange(1, len(timestamps) + 1)
                    ax.plot(timestamps, cumulative_counts,
                            label=f'{scenario_name} - {prio.name}',
                            color=colors[prio], linestyle=linestyle, lw=2.5)

        ax.set_xlabel('Tempo di Simulazione (s)', fontsize=12)
        ax.set_ylabel('Numero Cumulativo di Richieste Perse', fontsize=12)
        ax.set_yscale('log'); ax.set_ylim(bottom=1)
        ax.grid(True, which='both', linestyle='--'); ax.legend(title='Scenario - Priorità')
        self._save_plot(output_dir, f"{run_prefix}_7_cumulative_loss_trace.png", fig)


    def plot_blackfriday_replication_traces(self, all_results: dict, lambda_func, output_dir: str):
        """
        NUOVO: Genera due grafici separati per lo scenario Black Friday.
        - Un grafico per il Baseline (FIFO) con le tracce di N repliche.
        - Un grafico per il DWFQ con le tracce di N repliche.
        """
        sim_time = self.config.SIMULATION_TIME
        num_replications = len(all_results)

        # Dizionario per contenere i dati delle medie mobili per ogni replica e modello
        traces = {'baseline': [], 'wfq': []}

        # Funzione helper per calcolare la media mobile (già presente nel tuo codice)
        def get_ma(history):
            TIME_WINDOW_STR = '60s'
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('10s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()

        # 1. Calcola le tracce per ogni replica
        for i in range(num_replications):
            rep_data = all_results[i]

            # Traccia per il Baseline
            metrics_base = rep_data['baseline']
            base_history_high_only = [h for rt, h_list in metrics_base.response_times_history.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH for h in h_list]
            traces['baseline'].append(get_ma(base_history_high_only))

            # Traccia per il DWFQ
            metrics_wfq = rep_data['wfq']
            wfq_history_high = metrics_wfq.response_times_history_by_prio.get(config.Priority.HIGH, [])
            traces['wfq'].append(get_ma(wfq_history_high))

        # 2. Crea i due grafici
        for model in ['baseline', 'wfq']:
            fig, ax = plt.subplots(figsize=(20, 8))

            model_name_display = "Baseline (FIFO)" if model == 'baseline' else "DWFQ"
            ax.set_title(f"Tracce delle Repliche per {model_name_display} - Scenario Black Friday", fontsize=18)

            colors = sns.color_palette("viridis", n_colors=num_replications)

            # Plotta ogni traccia di replica
            for i in range(num_replications):
                series = traces[model][i]
                seed = all_results[i]['seed']
                if series is not None and not series.empty:
                    time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                    ax.plot(time_in_seconds, series.values, color=colors[i], lw=1.5, alpha=0.9, label=f'Seed: {seed}')

            # Aggiungi il profilo di carico
            ax_load = ax.twinx()
            load_times = np.linspace(0, sim_time, num=2000)
            load_values = [lambda_func(t) for t in load_times]
            ax_load.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico Applicato')

            ax.set_xlabel("Tempo (s)", fontsize=14)
            ax.set_ylabel("Tempo Risposta Medio (s) - HIGH Prio", fontsize=14)
            ax.grid(True)
            ax.set_ylim(bottom=0, top=5.0) # Imposta un limite Y fisso per confrontabilità

            ax_load.set_ylabel("Carico (req/s)", color='gray', fontsize=14)

            # Combina le legende
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_load.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=10)

            fig.tight_layout()
            self._save_plot(output_dir, f"blackfriday_traces_{model}.png", fig)

