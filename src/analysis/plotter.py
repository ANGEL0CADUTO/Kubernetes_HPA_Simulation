from os import makedirs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority

matplotlib.use('Qt5Agg')
plt.style.use('ggplot')

def ensure_plot_dir():
    makedirs('plots', exist_ok=True)


def _get_all_raw_data(data_by_type: dict):
    """Metodo helper per appiattire un dizionario di liste in un'unica lista."""
    return [item for sublist in data_by_type.values() for item in sublist]


def _calculate_overall_avg(times_by_type: dict):
    """Metodo helper per calcolare la media complessiva da un dizionario di liste."""
    all_times = [t for times_list in times_by_type.values() for t in times_list]
    return np.mean(all_times) if all_times else 0


class Plotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config



    # --- IL NUOVO DASHBOARD DI CONFRONTO (UNICA PARTE NUOVA) ---
    def plot_comparison_dashboard(self):
        """
        Crea un dashboard di confronto 1x2 che confronta direttamente le performance
        dei due scenari in modo chiaro e intuitivo.
        """
        print("Generazione del dashboard di confronto semplificato...")

        # --- Setup della Figura e Colori ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_facecolor('#f9f9f9')
        ax2.set_facecolor('#f9f9f9')  # colore chiaro di sfondo dietro le barre o linee

        fig.suptitle("Confronto Performance: Con Priorità vs. Senza Priorità",fontsize=18)

        colors = {
            'prio':  '#0000ff',     # blue
            'no_prio': '#ff0000'   # red
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

            ax1.set_title('Tempi di Risposta Medi per Tipo di Richiesta')
            ax1.set_xlabel('Tipo di Richiesta')
            ax1.set_ylabel('Tempo Medio (s)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

            legend = ax1.legend(title='Scenario')
            legend.get_title().set_fontweight('bold')

            for container in ax1.containers:
                ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=6)
        else:
            ax1.text(0.5, 0.5, 'Nessun dato sui tempi di risposta disponibile.', ha='center', va='center')

        # --- 2. GRAFICO A DESTRA: Confronto Metriche Chiave con % di Miglioramento ---
        metrics_to_compare = ['Throughput (req/s)', 'Tempo Attesa Medio (s)', '% Timeout']

        total_served_prio = sum(self.metrics_prio.requests_completed_by_priority.values())
        total_generated_prio = sum(self.metrics_prio.requests_generated_by_priority.values())
        total_timeouts_prio = sum(self.metrics_prio.requests_timed_out_by_priority.values())

        throughput_prio = total_served_prio / self.config.SIMULATION_TIME if self.config.SIMULATION_TIME > 0 else 0
        avg_wait_prio = _calculate_overall_avg(self.metrics_prio.wait_times_by_req_type)
        timeout_perc_prio = (total_timeouts_prio / total_generated_prio) * 100 if total_generated_prio > 0 else 0

        total_timeouts_no_prio = sum(self.metrics.requests_timed_out_data.values())
        throughput_no_prio = self.metrics.total_requests_served / self.config.SIMULATION_TIME if self.config.SIMULATION_TIME > 0 else 0
        avg_wait_no_prio = _calculate_overall_avg(self.metrics.wait_times_data)
        timeout_perc_no_prio = (total_timeouts_no_prio / self.metrics.total_requests_generated) * 100 if self.metrics.total_requests_generated > 0 else 0

        values_prio = [throughput_prio, avg_wait_prio, timeout_perc_prio]
        values_no_prio = [throughput_no_prio, avg_wait_no_prio, timeout_perc_no_prio]

        x = np.arange(len(metrics_to_compare))
        width = 0.35

        bars1 = ax2.bar(x - width/2, values_prio, width, label='Con Priorità', color=colors['prio'])
        bars2 = ax2.bar(x + width/2, values_no_prio, width, label='Senza Priorità', color=colors['no_prio'])

        ax2.set_title('Confronto Metriche Chiave', pad=15)
        ax2.set_ylabel('Valore')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_to_compare)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

        legend2 = ax2.legend(title='Scenario')
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

                ax2.text(i, y_max * 0.9, text, ha='center', va='bottom', fontsize=12, fontweight='bold', color=color,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor=color))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        ensure_plot_dir()
        plt.savefig('plots/comparison_dashboard_simplified.png', dpi=300, bbox_inches='tight')
        plt.show()

    # --- I TUOI METODI DI PLOT ORIGINALI (NON TOCCATI) ---
    def plot_pod_history(self):
        plt.figure(figsize=(12, 6))
          # colore chiaro di sfondo dietro le linee
        if self.metrics.pod_count_history:
            timestamps_senza_priorita = [t for t, _ in self.metrics.pod_count_history]
            pod_counts_senza_priorita = [c for _, c in self.metrics.pod_count_history]
            plt.plot(timestamps_senza_priorita, pod_counts_senza_priorita, color='r', linewidth=4, label='senza priorità', alpha=0.8)
        if self.metrics_prio.timestamps:
            plt.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', linewidth=4, label='Con Priorità', alpha=0.8)
        plt.xlabel('Tempo di simulazione (s)')
        plt.ylabel('Numero di Pod')
        plt.title('Evoluzione del Numero di Pod nel Tempo')
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        plt.ylim(self.config.INITIAL_PODS, self.config.MAX_PODS + 1)
        plt.tight_layout()
        ensure_plot_dir()
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
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        ensure_plot_dir()
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
            all_responses_senza.sort(key=lambda x: x[0])  # Ordina per tempo
            times_senza = [t for t, _ in all_responses_senza]
            responses_senza = [r for _, r in all_responses_senza]
            cum_avg_senza = np.cumsum(responses_senza) / np.arange(1, len(responses_senza)+1)
            plt.plot(times_senza, cum_avg_senza, label='Senza Priorità', color='r', alpha=0.8)

        all_responses_prio = []
        for req_type in sorted(self.metrics_prio.response_times_by_req_type, key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            resp = self.metrics_prio.response_times_by_req_type[req_type]
            all_responses_prio.extend(zip(times, resp))
        all_responses_prio.sort(key=lambda x: x[0])
        if all_responses_prio:
            all_responses_prio.sort(key=lambda x: x[0])
            times_prio = [t for t, _ in all_responses_prio]
            responses_prio = [r for _, r in all_responses_prio]
            cum_avg_prio = np.cumsum(responses_prio) / np.arange(1, len(responses_prio)+1)
            plt.plot(times_prio, cum_avg_prio, label='Con Priorità', color='b', alpha=0.8)

        if not all_responses_senza and not all_responses_prio:
                print("Nessun dato sui tempi di risposta da plottare.")
                return
        plt.title("Andamento del Tempo di Risposta Medio nel Tempo")
        plt.xlabel("Tempo di Simulazione (s)")
        plt.ylabel("Tempo di Risposta Medio (s)")
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        ensure_plot_dir()
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
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
        plt.legend(loc='upper right', frameon=False)
        plt.tight_layout()
        ensure_plot_dir()
        plt.savefig("plots/wait_time_trend.png", dpi=300, bbox_inches='tight')
        plt.show()







    # --- METODO DI REPORTING (AGGIORNATO PER CHIAMARE IL NUOVO DASHBOARD) ---
    def generate_comprehensive_report(self):
        self.plot_comparison_dashboard()
        self.plot_pod_history()
        self.plot_queue_history()
        self.plot_wait_time_trend()
        self.plot_response_time_trend()




    # --- METODO DI UTILITÀ (NON TOCCATO) ---
