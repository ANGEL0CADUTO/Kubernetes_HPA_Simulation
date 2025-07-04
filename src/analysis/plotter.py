from os import makedirs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

from src.utils.metrics import Metrics
from src.config import *
from src.utils.metrics_with_priority import MetricsWithPriority

matplotlib.use('Qt5Agg')

class Plotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config

    # --- METODI HELPER INTERNI (necessari solo per il nuovo dashboard) ---
    def _calculate_overall_avg(self, times_by_type: dict):
        """Metodo helper per calcolare la media complessiva da un dizionario di liste."""
        all_times = [t for times_list in times_by_type.values() for t in times_list]
        return np.mean(all_times) if all_times else 0

    def _get_all_raw_data(self, data_by_type: dict):
        """Metodo helper per appiattire un dizionario di liste in un'unica lista."""
        return [item for sublist in data_by_type.values() for item in sublist]

    # --- IL NUOVO DASHBOARD DI CONFRONTO (UNICA PARTE NUOVA) ---
    def plot_comparison_dashboard(self):
        """
        Crea un dashboard di confronto 1x2 che confronta direttamente le performance
        dei due scenari in modo chiaro e intuitivo.
        """
        print("Generazione del dashboard di confronto semplificato...")

        # --- Setup della Figura e Colori ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
        fig.suptitle("Confronto Performance: Con Priorità vs. Senza Priorità", fontsize=24, fontweight='bold', y=1.02)

        colors = {
            'prio': '#0077b6',     # Blu vibrante
            'no_prio': '#f4a261'   # Arancione sabbia
        }

        # --- 1. GRAFICO A SINISTRA: Confronto Diretto dei Tempi di Risposta per Tipo ---
        plot_data = []
        all_req_types = set(self.metrics.response_times_data.keys()) | set(self.metrics_prio.response_times_by_req_type.keys())

        for req_type in sorted(list(all_req_types), key=lambda x: x.name):
            resp_times_no_prio = self.metrics.response_times_data.get(req_type)
            if resp_times_no_prio:
                plot_data.append({
                    'Categoria': req_type.name,
                    'Tempo Medio (s)': np.mean(resp_times_no_prio),
                    'Scenario': 'Senza Priorità'
                })

            resp_times_prio = self.metrics_prio.response_times_by_req_type.get(req_type)
            if resp_times_prio:
                plot_data.append({
                    'Categoria': req_type.name,
                    'Tempo Medio (s)': np.mean(resp_times_prio),
                    'Scenario': 'Con Priorità'
                })

        if plot_data:
            df_resp_time = pd.DataFrame(plot_data)
            sns.barplot(data=df_resp_time, x='Categoria', y='Tempo Medio (s)', hue='Scenario', ax=ax1, palette=list(colors.values()))

            ax1.set_title('Tempi di Risposta Medi per Tipo di Richiesta', fontsize=18, fontweight='bold', pad=20)
            ax1.set_xlabel('Tipo di Richiesta', fontsize=14)
            ax1.set_ylabel('Tempo Medio (s)', fontsize=14)
            ax1.tick_params(axis='x', rotation=45, labelsize=12)
            ax1.tick_params(axis='y', labelsize=12)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

            legend = ax1.legend(title='Scenario', fontsize=12)
            legend.get_title().set_fontweight('bold')

            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Nessun dato sui tempi di risposta disponibile.', ha='center', va='center')

        # --- 2. GRAFICO A DESTRA: Confronto Metriche Chiave con % di Miglioramento ---
        metrics_to_compare = ['Throughput (req/s)', 'Tempo Attesa Medio (s)', '% Timeout']

        total_served_prio = sum(self.metrics_prio.requests_completed_by_priority.values())
        total_generated_prio = sum(self.metrics_prio.requests_generated_by_priority.values())
        total_timeouts_prio = sum(self.metrics_prio.requests_timed_out_by_priority.values())

        throughput_prio = total_served_prio / self.config.SIMULATION_TIME if self.config.SIMULATION_TIME > 0 else 0
        avg_wait_prio = self._calculate_overall_avg(self.metrics_prio.wait_times_by_req_type)
        timeout_perc_prio = (total_timeouts_prio / total_generated_prio) * 100 if total_generated_prio > 0 else 0

        total_timeouts_no_prio = sum(self.metrics.requests_timed_out_data.values())
        throughput_no_prio = self.metrics.total_requests_served / self.config.SIMULATION_TIME if self.config.SIMULATION_TIME > 0 else 0
        avg_wait_no_prio = self._calculate_overall_avg(self.metrics.wait_times_data)
        timeout_perc_no_prio = (total_timeouts_no_prio / self.metrics.total_requests_generated) * 100 if self.metrics.total_requests_generated > 0 else 0

        values_prio = [throughput_prio, avg_wait_prio, timeout_perc_prio]
        values_no_prio = [throughput_no_prio, avg_wait_no_prio, timeout_perc_no_prio]

        x = np.arange(len(metrics_to_compare))
        width = 0.35

        bars1 = ax2.bar(x - width/2, values_prio, width, label='Con Priorità', color=colors['prio'])
        bars2 = ax2.bar(x + width/2, values_no_prio, width, label='Senza Priorità', color=colors['no_prio'])

        ax2.set_title('Confronto Metriche Chiave', fontsize=18, fontweight='bold', pad=20)
        ax2.set_ylabel('Valore', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_to_compare, fontsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

        legend2 = ax2.legend(title='Scenario', fontsize=12)
        legend2.get_title().set_fontweight('bold')

        y_max = ax2.get_ylim()[1]
        for i, metric_name in enumerate(metrics_to_compare):
            val_prio = values_prio[i]
            val_no_prio = values_no_prio[i]

            if val_no_prio > 0.0001:
                if 'Throughput' in metric_name:
                    improvement = ((val_prio - val_no_prio) / val_no_prio) * 100
                else:
                    improvement = ((val_no_prio - val_prio) / val_no_prio) * 100

                sign = '+' if improvement >= 0 else ''
                color = 'green' if improvement >= 0 else 'red'
                text = f'Δ: {sign}{improvement:.1f}%'

                ax2.text(i, y_max * 0.9, text, ha='center', va='bottom', fontsize=14, fontweight='bold', color=color,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self.ensure_plot_dir()
        plt.savefig('plots/comparison_dashboard_simplified.png', dpi=300, bbox_inches='tight')
        plt.show()

    # --- I TUOI METODI DI PLOT ORIGINALI (NON TOCCATI) ---
    def plot_pod_history(self):
        plt.figure(figsize=(12, 6))
        if self.metrics.pod_count_history:
            timestamps_senza_priorita = [t for t, _ in self.metrics.pod_count_history]
            pod_counts_senza_priorita = [c for _, c in self.metrics.pod_count_history]
            plt.plot(timestamps_senza_priorita, pod_counts_senza_priorita, color='r', linewidth=4, label='senza priorità', alpha=0.8)
        if self.metrics_prio.timestamps:
            plt.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', linewidth=4, label='Con Priorità', alpha=0.8)
        plt.xlabel('Tempo di simulazione (s)')
        plt.ylabel('Numero di Pod')
        plt.title('Evoluzione del Numero di Pod nel Tempo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(self.config.INITIAL_PODS, self.config.MAX_PODS + 1)
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig('plots/pod_count_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_queue_history(self):
        if not self.metrics.queue_length_history and not self.metrics_prio.queue_lengths:
            return
        plt.figure(figsize=(12, 6))
        if self.metrics.queue_length_history:
            times_senza_priorita = [t for t, _ in self.metrics.queue_length_history]
            lengths_senza_priorita = [l for _, l in self.metrics.queue_length_history]
            plt.plot(times_senza_priorita, lengths_senza_priorita, color='r', linewidth=2, label='Senza Priorità', alpha=0.8)
        if self.metrics_prio.queue_lengths and self.metrics_prio.timestamps:
            plt.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', linewidth=2, label='Con Priorità', alpha=0.8)
        plt.title("Evoluzione della Lunghezza della Coda nel Tempo")
        plt.xlabel("Tempo di Simulazione (s)")
        plt.ylabel("Numero di Richieste in Coda")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig('plots/queue_length_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_response_time_trend(self):
        plt.figure(figsize=(12, 6))
        window_size = 50
        all_responses_senza = []
        for req_type in sorted(self.metrics.response_times_history, key=lambda e: e.name):
            all_responses_senza.extend(self.metrics.response_times_history[req_type])
        all_responses_senza.sort(key=lambda x: x[0])
        if all_responses_senza:
            times_senza = [t for t, _ in all_responses_senza]
            responses_senza = [r for _, r in all_responses_senza]
            if len(responses_senza) >= window_size:
                moving_avg = np.convolve(responses_senza, np.ones(window_size) / window_size, mode='valid')
                plt.plot(times_senza[window_size - 1:], moving_avg, label='Senza Priorità', color='r', alpha=0.8)
        all_responses_prio = []
        for req_type in sorted(self.metrics_prio.response_times_by_req_type, key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            resp = self.metrics_prio.response_times_by_req_type[req_type]
            all_responses_prio.extend(zip(times, resp))
        all_responses_prio.sort(key=lambda x: x[0])
        if all_responses_prio:
            times_prio, resp_prio = zip(*all_responses_prio)
            if len(resp_prio) >= window_size:
                avg = np.convolve(resp_prio, np.ones(window_size)/window_size, mode='valid')
                plt.plot(times_prio[window_size - 1:], avg, label='Con Priorità', color='b', alpha=0.8)
        if not all_responses_senza and not all_responses_prio:
            print("Nessun dato sui tempi di risposta da plottare.")
            return
        plt.title("Andamento del Tempo di Risposta Medio nel Tempo")
        plt.xlabel("Tempo di Simulazione (s)")
        plt.ylabel("Tempo di Risposta Medio (s)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig('plots/response_time_trend.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_wait_time_trend(self):
        plt.figure(figsize=(12, 6))
        window_size = 50
        all_waits_senza = []
        for req_type in sorted(self.metrics.wait_times_history, key=lambda e: e.name):
            all_waits_senza.extend(self.metrics.wait_times_history[req_type])
        all_waits_senza.sort(key=lambda x: x[0])
        if all_waits_senza:
            times_senza, waits_senza = zip(*all_waits_senza)
            if len(waits_senza) >= window_size:
                moving_avg = np.convolve(waits_senza, np.ones(window_size) / window_size, mode='valid')
                plt.plot(times_senza[window_size - 1:], moving_avg, label='Senza Priorità', color='r', alpha=0.8)
        all_waits_prio = []
        for req_type in sorted(self.metrics_prio.wait_times_by_req_type, key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            waits = self.metrics_prio.wait_times_by_req_type[req_type]
            all_waits_prio.extend(zip(times, waits))
        all_waits_prio.sort(key=lambda x: x[0])
        if all_waits_prio:
            times_prio, waits_prio = zip(*all_waits_prio)
            if len(waits_prio) >= window_size:
                moving_avg = np.convolve(waits_prio, np.ones(window_size) / window_size, mode='valid')
                plt.plot(times_prio[window_size - 1:], moving_avg, label='Con Priorità', color='b', alpha=0.8)
        if not all_waits_senza and not all_waits_prio:
            print("Nessun dato per i tempi di attesa.")
            return
        plt.title("Andamento del Tempo Medio di Attesa nel Tempo")
        plt.xlabel("Tempo di Simulazione (s)")
        plt.ylabel("Tempo di Attesa Medio (s)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig("plots/wait_time_trend.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_response_time_histogram(self):
        no_priority_req_types = list(self.metrics.response_times_data.keys())
        priority_req_types= list(self.metrics_prio.response_times_by_req_type.keys())
        all_req_types=[]
        for req_type in no_priority_req_types:
            if req_type not in all_req_types:
                all_req_types.append(req_type)
        for req_type in priority_req_types:
            if req_type not in all_req_types:
                all_req_types.append(req_type)
        all_req_types=sorted(all_req_types, key=lambda e: e.name)
        if not all_req_types:
            print("Nessun dato per l'istogramma comparativo dei tempi di risposta.")
            return
        num_types = len(all_req_types)
        cols= min(4, num_types)
        rows=(num_types+ cols-1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 6*rows))
        if num_types==1:
            axes= [axes]
        elif rows==1:
            axes= axes if isinstance(axes,np.ndarray) else [axes]
        else:
            axes=axes.flatten()
        for i, req_type in enumerate(all_req_types):
            ax = axes[i]
            no_priority_data = self.metrics.response_times_data.get(req_type, [])
            priority_data=self.metrics_prio.response_times_by_req_type.get(req_type, [])
            if no_priority_data and priority_data:
                all_data= no_priority_data + priority_data
                bins=np.linspace(min(all_data), max(all_data), 30)
                ax.hist(no_priority_data, bins=bins, alpha=0.6, label='Senza priorità', color="skyblue", edgecolor='black')
                ax.hist(priority_data, bins=bins, alpha=0.6, label='Con priorità',color='lightcoral',edgecolor='black')
                mean_no_priority = np.mean(no_priority_data)
                mean_priority = np.mean(priority_data)
                ax.axvline(mean_no_priority, color='blue', linestyle='--',linewidth=2,label=f'Media No priority: {mean_no_priority}s')
                ax.axvline(mean_priority,color='red', linestyle='--',linewidth=2,label=f'Media Priority: {mean_priority}s')
                improvement=((mean_no_priority-mean_priority)/mean_no_priority)*100
                ax.text(0.02, 0.98, f'Miglioramento: {improvement:.1f}%', transform=ax.transAxes, va='top', ha='left', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            elif no_priority_data:
                ax.hist(no_priority_data,bins=30,alpha=0.6,label='Senza priorità',color='skyblue',edgecolor='black')
            elif priority_data:
                ax.hist(priority_data,bins=30,alpha=0.6,label='Con priorità',color='lightcoral',edgecolor='black')
            else:
                ax.test(0.5,0.5,'Nessun dato disponibile',transform=ax.transAxes,ha='center',va='center')
            ax.set_title(f"{req_type.name}",fontsize=14,fontweight='bold')
            ax.set_xlabel("Tempo di Risposta (s)",fontsize=12)
            ax.set_ylabel("Frequenza",fontsize=12)
            legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('gray')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=10)
        for j in range(i+1,len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Confronto sovrapposto tempi di risposta:senza priorità vs priorità", fontsize=18,fontweight='bold',y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        self.ensure_plot_dir()
        plt.savefig('plots/response_time_overlay_comparison_histogram.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_request_heatmap(self, bin_size=10):
        req_types = sorted(self.metrics.response_times_history.keys(), key=lambda e: e.name)
        all_data = []
        for req_type in req_types:
            for timestamp, _ in self.metrics.response_times_history[req_type]:
                all_data.append((req_type.name, int(timestamp // bin_size) * bin_size))
        if not all_data:
            print("Nessun dato per la heatmap.")
            return
        df = pd.DataFrame(all_data, columns=["Tipo", "TimeBin"])
        heatmap_data = df.groupby(["Tipo", "TimeBin"]).size().unstack(fill_value=0)
        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, linecolor='gray')
        plt.title("Heatmap: Intensità delle Richieste nel Tempo")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Tipo di Richiesta")
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig("plots/request_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_cumulative_requests(self):
        served = self.metrics.total_requests_served
        generated = self.metrics.total_requests_generated
        if served == 0:
            print("Nessuna richiesta servita da plottare.")
            return
        plt.figure(figsize=(12, 6))
        served_timeline = [0]
        generated_timeline = [0]
        time_points = [0]
        current_served = 0
        current_generated = 0
        for ts, _ in sorted(self.metrics.queue_length_history):
            current_served += 1
            current_generated += 1
            time_points.append(ts)
            served_timeline.append(current_served)
            generated_timeline.append(current_generated)
        plt.plot(time_points, generated_timeline, label="Richieste Generate", color='blue')
        plt.plot(time_points, served_timeline, label="Richieste Servite", color='green')
        plt.title("Andamento Cumulativo delle Richieste nel Tempo")
        plt.xlabel("Tempo di Simulazione (s)")
        plt.ylabel("Numero Cumulativo")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        self.ensure_plot_dir()
        plt.savefig("plots/cumulative_requests.png", dpi=300, bbox_inches='tight')
        plt.show()

    # --- METODO DI REPORTING (AGGIORNATO PER CHIAMARE IL NUOVO DASHBOARD) ---
    def generate_comprehensive_report(self):
        self.plot_comparison_dashboard()
        self.plot_pod_history()
        self.plot_queue_history()
        self.plot_wait_time_trend()
        self.plot_response_time_trend()
        self.plot_cumulative_requests()
        self.plot_response_time_histogram()
        self.plot_request_heatmap()

    # --- METODO DI UTILITÀ (NON TOCCATO) ---
    def ensure_plot_dir(self):
        makedirs('plots', exist_ok=True)