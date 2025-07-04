# analysis/plotter.py - VERSIONE MERGED

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from matplotlib.ticker import MaxNLocator

matplotlib.use('Qt5Agg')
plt.style.use('ggplot')

# Funzione helper per calcolare le medie, rimane invariata
def _calculate_overall_avg(times_by_type: dict):
    """Metodo helper per calcolare la media complessiva da un dizionario di liste."""
    all_times = [t for times_list in times_by_type.values() for t in times_list]
    return np.mean(all_times) if all_times else 0


class Plotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config

    # --- FUNZIONE HELPER PER IL SALVATAGGIO ---
    def _save_plot(self, output_dir, filename, fig):
        """Helper per creare la directory e salvare la figura."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Chiude la figura per liberare memoria

    def plot_queue_history(self, output_dir='plots', filename='queue_length_history.png'):
        print(f"Generazione storico coda -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        if self.metrics.queue_length_history:
            times_no_prio, lengths_no_prio = zip(*self.metrics.queue_length_history)
            ax.plot(times_no_prio, lengths_no_prio, color='r', linewidth=2, label='Senza Priorità', alpha=0.8)
            ax.axhline(np.mean(lengths_no_prio), color='darkred', linestyle='--', linewidth=1, label=f'Media Senza Priorità: {np.mean(lengths_no_prio):.2f}')
        if self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', linewidth=2, label='Con Priorità', alpha=0.8)
            ax.axhline(np.mean(self.metrics_prio.queue_lengths), color='darkblue', linestyle='--', linewidth=1, label=f'Media Con Priorità: {np.mean(self.metrics_prio.queue_lengths):.2f}')
        ax.set_title("Evoluzione della Lunghezza della Coda nel Tempo"); ax.set_xlabel("Tempo di Simulazione (s)"); ax.set_ylabel("Numero di Richieste in Coda")
        ax.legend(loc='best'); ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_wait_time_trend(self, output_dir='plots', filename='wait_time_trend.png'):
        print(f"Generazione trend tempo di attesa -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        window_size = 50
        all_waits_senza = []
        for req_type in sorted(self.metrics.wait_times_history.keys(), key=lambda e: e.name):
            all_waits_senza.extend(self.metrics.wait_times_history[req_type])
        if all_waits_senza:
            all_waits_senza.sort(key=lambda x: x[0])
            times_senza, waits_senza = zip(*all_waits_senza)
            if len(waits_senza) >= window_size:
                moving_avg = np.convolve(waits_senza, np.ones(window_size) / window_size, mode='valid')
                ax.plot(times_senza[window_size - 1:], moving_avg, label='Senza Priorità (Media Mobile)', color='r', alpha=0.7)
            ax.axhline(np.mean(waits_senza), color='darkred', linestyle='--', linewidth=1, label=f'Media Totale Senza Priorità: {np.mean(waits_senza):.2f}')
        all_waits_prio = []
        for req_type in sorted(self.metrics_prio.wait_times_by_req_type.keys(), key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            waits = self.metrics_prio.wait_times_by_req_type[req_type]
            all_waits_prio.extend(zip(times, waits))
        if all_waits_prio:
            all_waits_prio.sort(key=lambda x: x[0])
            times_prio, waits_prio = zip(*all_waits_prio)
            if len(waits_prio) >= window_size:
                moving_avg_prio = np.convolve(waits_prio, np.ones(window_size) / window_size, mode='valid')
                ax.plot(times_prio[window_size - 1:], moving_avg_prio, label='Con Priorità (Media Mobile)', color='b', alpha=0.7)
            ax.axhline(np.mean(waits_prio), color='darkblue', linestyle='--', linewidth=1, label=f'Media Totale Con Priorità: {np.mean(waits_prio):.2f}')
        ax.set_title("Andamento del Tempo di Attesa Medio (Media Mobile)")
        ax.set_xlabel("Tempo di Simulazione (s)")
        ax.set_ylabel("Tempo di Attesa Medio (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        fig.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_response_time_trend(self, output_dir='plots', filename='response_time_trend.png'):
        print(f"Generazione trend tempo di risposta -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        all_responses_senza = []
        for req_type in sorted(self.metrics.response_times_history.keys(), key=lambda e: e.name):
            all_responses_senza.extend(self.metrics.response_times_history[req_type])
        if all_responses_senza:
            all_responses_senza.sort(key=lambda x: x[0])
            times_senza, responses_senza = zip(*all_responses_senza)
            cum_avg_senza = np.cumsum(responses_senza) / np.arange(1, len(responses_senza) + 1)
            ax.plot(times_senza, cum_avg_senza, label='Senza Priorità (Media Cumulativa)', color='r', alpha=0.8)
        all_responses_prio = []
        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            resp = self.metrics_prio.response_times_by_req_type[req_type]
            all_responses_prio.extend(zip(times, resp))
        if all_responses_prio:
            all_responses_prio.sort(key=lambda x: x[0])
            times_prio, responses_prio = zip(*all_responses_prio)
            cum_avg_prio = np.cumsum(responses_prio) / np.arange(1, len(responses_prio) + 1)
            ax.plot(times_prio, cum_avg_prio, label='Con Priorità (Media Cumulativa)', color='b', alpha=0.8)
        ax.set_title("Andamento del Tempo di Risposta Medio Cumulativo nel Tempo")
        ax.set_xlabel("Tempo di Simulazione (s)")
        ax.set_ylabel("Tempo di Risposta Medio (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best')
        fig.tight_layout()
        self._save_plot(output_dir, filename, fig)
        # --- METODI DI PLOTTING AGGIORNATI CON PARAMETRI DI OUTPUT ---

    def plot_comparison_dashboard(self, output_dir='plots', filename='comparison_dashboard.png'):
        """Crea un dashboard di confronto 1x2."""
        print(f"Generazione del dashboard di confronto -> {os.path.join(output_dir, filename)}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Confronto Performance: Con Priorità vs. Senza Priorità", fontsize=20, fontweight='bold')

        # ... [TUTTA LA LOGICA INTERNA DEL PLOT RIMANE IDENTICA] ...
        ax1.set_facecolor('#f9f9f9'); ax2.set_facecolor('#f9f9f9')
        colors = {'prio': '#0000ff', 'no_prio': '#ff0000'}

        # Grafico 1: Tempi di risposta
        plot_data = []
        all_req_types = set(self.metrics.response_times_data.keys()) | set(self.metrics_prio.response_times_by_req_type.keys())
        for req_type in sorted(list(all_req_types), key=lambda x: x.name):
            if req_type in self.metrics.response_times_data and self.metrics.response_times_data[req_type]:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': np.mean(self.metrics.response_times_data[req_type]), 'Scenario': 'Senza Priorità'})
            if req_type in self.metrics_prio.response_times_by_req_type and self.metrics_prio.response_times_by_req_type[req_type]:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': np.mean(self.metrics_prio.response_times_by_req_type[req_type]), 'Scenario': 'Con Priorità'})

        if plot_data:
            df_resp_time = pd.DataFrame(plot_data)
            sns.barplot(data=df_resp_time, x='Categoria', y='Tempo Medio (s)', hue='Scenario', hue_order=['Senza Priorità', 'Con Priorità'], palette=[colors['no_prio'], colors['prio']], ax=ax1)
            ax1.set_title('Tempi di Risposta Medi per Tipo di Richiesta', fontsize=14)
            ax1.set_xlabel('Tipo di Richiesta', fontsize=12); ax1.set_ylabel('Tempo Medio (s)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45, labelsize=10); plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")
            for container in ax1.containers: ax1.bar_label(container, fmt='%.2f', padding=3, fontsize=8)
            ax1.legend(title='Scenario').get_title().set_fontweight('bold')
            ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

        # Grafico 2: Metriche chiave
        metrics_to_compare = ['Tempo di Risposta Medio (s)', 'Tempo Attesa Medio (s)', '% Timeout']
        total_generated_prio = sum(self.metrics_prio.requests_generated_by_req_type.values())
        total_timeouts_prio = sum(self.metrics_prio.requests_timed_out_by_req_type.values())
        avg_response_prio = _calculate_overall_avg(self.metrics_prio.response_times_by_req_type)
        avg_wait_prio = _calculate_overall_avg(self.metrics_prio.wait_times_by_req_type)
        timeout_perc_prio = (total_timeouts_prio / total_generated_prio) * 100 if total_generated_prio > 0 else 0
        total_timeouts_no_prio = sum(self.metrics.requests_timed_out_data.values())
        avg_response_no_prio = _calculate_overall_avg(self.metrics.response_times_data)
        avg_wait_no_prio = _calculate_overall_avg(self.metrics.wait_times_data)
        timeout_perc_no_prio = (total_timeouts_no_prio / self.metrics.total_requests_generated) * 100 if self.metrics.total_requests_generated > 0 else 0
        values_prio = [avg_response_prio, avg_wait_prio, timeout_perc_prio]
        values_no_prio = [avg_response_no_prio, avg_wait_no_prio, timeout_perc_no_prio]
        x = np.arange(len(metrics_to_compare)); width = 0.35
        bars1 = ax2.bar(x - width/2, values_no_prio, width, label='Senza Priorità', color=colors['no_prio'])
        bars2 = ax2.bar(x + width/2, values_prio, width, label='Con Priorità', color=colors['prio'])
        ax2.set_title('Confronto Metriche Chiave', fontsize=14, pad=20)
        ax2.set_ylabel('Valore', fontsize=12); ax2.set_xticks(x); ax2.set_xticklabels(metrics_to_compare, fontsize=10)
        ax2.legend(title='Scenario').get_title().set_fontweight('bold')
        ax2.bar_label(bars1, padding=3, fmt='%.3f', fontsize=9); ax2.bar_label(bars2, padding=3, fmt='%.3f', fontsize=9)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        y_max = ax2.get_ylim()[1]
        for i, metric_name in enumerate(metrics_to_compare):
            val_prio = values_prio[i]; val_no_prio = values_no_prio[i]
            if val_no_prio > 0.0001:
                improvement = ((val_no_prio - val_prio) / val_no_prio) * 100
                sign = '-' if improvement >= 0 else '+'; color = 'green' if improvement >= 0 else 'red'
                text = f'Δ: {sign}{abs(improvement):.1f}%'
                ax2.text(i, y_max * 0.85, text, ha='center', va='bottom', fontsize=12, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, filename, fig)

    def plot_pod_history(self, output_dir='plots', filename='pod_count_history.png'):
        print(f"Generazione storico pod -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        # ... [LOGICA INTERNA IDENTICA] ...
        if self.metrics.pod_count_history:
            timestamps_no_prio, counts_no_prio = zip(*self.metrics.pod_count_history)
            ax.plot(timestamps_no_prio, counts_no_prio, color='r', linewidth=2.5, label='Senza Priorità', alpha=0.8)
        if self.metrics_prio.pod_counts:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', linewidth=2.5, label='Con Priorità', alpha=0.8)
        ax.set_xlabel('Tempo di simulazione (s)'); ax.set_ylabel('Numero di Pod'); ax.set_title('Evoluzione del Numero di Pod nel Tempo')
        ax.legend(loc='best'); ax.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.set_ylim(bottom=0)
        fig.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_queue_history(self, output_dir='plots', filename='queue_length_history.png'):
        print(f"Generazione storico coda -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        # ... [LOGICA INTERNA IDENTICA] ...
        if self.metrics.queue_length_history:
            times_no_prio, lengths_no_prio = zip(*self.metrics.queue_length_history)
            ax.plot(times_no_prio, lengths_no_prio, color='r', linewidth=2, label='Senza Priorità', alpha=0.8)
            ax.axhline(np.mean(lengths_no_prio), color='darkred', linestyle='--', linewidth=1, label=f'Media Senza Priorità: {np.mean(lengths_no_prio):.2f}')
        if self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', linewidth=2, label='Con Priorità', alpha=0.8)
            ax.axhline(np.mean(self.metrics_prio.queue_lengths), color='darkblue', linestyle='--', linewidth=1, label=f'Media Con Priorità: {np.mean(self.metrics_prio.queue_lengths):.2f}')
        ax.set_title("Evoluzione della Lunghezza della Coda nel Tempo"); ax.set_xlabel("Tempo di Simulazione (s)"); ax.set_ylabel("Numero di Richieste in Coda")
        ax.legend(loc='best'); ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_loss_by_type(self, output_dir='plots', filename='loss_comparison_by_type.png'):
        print(f"Generazione grafico perdite per tipo -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 7))
        # ... [LOGICA INTERNA IDENTICA] ...
        all_req_types = (set(self.metrics.requests_timed_out_data.keys()) | set(self.metrics_prio.requests_timed_out_by_req_type.keys()))
        if not all_req_types: plt.close(fig); return
        plot_data = []
        for req_type in sorted(list(all_req_types), key=lambda x: x.name):
            plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(),'Richieste Perse': self.metrics.requests_timed_out_data.get(req_type, 0),'Scenario': 'Senza Priorità'})
            plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(),'Richieste Perse': self.metrics_prio.requests_timed_out_by_req_type.get(req_type, 0),'Scenario': 'Con Priorità'})
        df_losses = pd.DataFrame(plot_data)
        sns.barplot(data=df_losses, x='Categoria', y='Richieste Perse', hue='Scenario', hue_order=['Senza Priorità', 'Con Priorità'], palette=['#ff0000', '#0000ff'], ax=ax)
        ax.set_title('Richieste Perse (Timeout) per Tipo', fontsize=16); fig.suptitle("Confronto Richieste Perse per Tipo", fontsize=20, fontweight='bold')
        ax.set_xlabel('Tipo di Richiesta'); ax.set_ylabel('Numero di Richieste Perse')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6); ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        for container in ax.containers: ax.bar_label(container, fmt='%d', padding=3, fontsize=9)
        ax.legend(title='Scenario', title_fontproperties={'weight': 'bold'})
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, filename, fig)

    def plot_served_by_type(self, output_dir='plots', filename='served_comparison_by_type.png'):
        print(f"Generazione grafico servite per tipo -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(14, 8))

        all_req_types = (set(self.metrics.response_times_data.keys()) | set(self.metrics_prio.response_times_by_req_type.keys()))
        if not all_req_types:
            plt.close(fig)
            return

        plot_data = []
        # Dizionario per calcolare facilmente il delta
        served_counts = {}

        for req_type in sorted(list(all_req_types), key=lambda x: x.name):
            served_no_prio = len(self.metrics.response_times_data.get(req_type, []))
            served_prio = len(self.metrics_prio.response_times_by_req_type.get(req_type, []))

            served_counts[req_type.name] = {'no_prio': served_no_prio, 'prio': served_prio}

            plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Richieste Servite': served_no_prio, 'Scenario': 'Senza Priorità'})
            plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Richieste Servite': served_prio, 'Scenario': 'Con Priorità'})

        df_served = pd.DataFrame(plot_data)
        sns.barplot(data=df_served, x='Categoria', y='Richieste Servite', hue='Scenario', hue_order=['Senza Priorità', 'Con Priorità'], palette=['#ff0000', '#0000ff'], ax=ax)

        ax.set_title('Richieste Servite con Successo per Tipo', fontsize=18)
        fig.suptitle("Confronto Richieste Servite per Tipo", fontsize=22, fontweight='bold')
        ax.set_xlabel('Tipo di Richiesta', fontsize=14)
        ax.set_ylabel('Numero di Richieste Servite', fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)

        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        for container in ax.containers:
            ax.bar_label(container, fmt='%d', padding=3, fontsize=10)

        ax.legend(title='Scenario', title_fontproperties={'weight': 'bold'}, fontsize=12)

        # --- AGGIUNTA LOGICA PER IL DELTA % ---
        y_max = ax.get_ylim()[1]

        # Determina la posizione per le etichette del delta
        for i, req_name_formatted in enumerate(ax.get_xticklabels()):
            req_name_original = req_name_formatted.get_text().replace(' ', '_').upper()

            # Recupera i conteggi per il tipo di richiesta corrente
            counts = served_counts.get(req_name_original)
            if not counts: continue

            val_no_prio = counts['no_prio']
            val_prio = counts['prio']

            if val_no_prio > 0:
                # Calcola il guadagno (o perdita) percentuale
                gain = ((val_prio - val_no_prio) / val_no_prio) * 100

                # Un guadagno è positivo (verde), una perdita è negativa (rossa)
                sign = '+' if gain >= 0 else ''
                color = 'green' if gain >= 0 else 'red'
                text = f'Δ: {sign}{gain:.1f}%'

                ax.text(i, y_max * 0.9, text, ha='center', va='bottom', fontsize=14, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self._save_plot(output_dir, filename, fig)


    # --- METODO DI REPORTING AGGIORNATO ---
    def generate_comprehensive_report(self, output_dir='plots', run_prefix='run'):
        """
        Chiama tutti i metodi di plotting, passando loro i percorsi di output corretti.
        """
        print(f"\n--- Generazione Report Completo per '{run_prefix}' in '{output_dir}' ---")
        self.plot_comparison_dashboard(output_dir=output_dir, filename=f"{run_prefix}_1_dashboard.png")
        self.plot_served_by_type(output_dir=output_dir, filename=f"{run_prefix}_2_served_by_type.png")
        self.plot_loss_by_type(output_dir=output_dir, filename=f"{run_prefix}_3_loss_by_type.png")
        self.plot_wait_time_trend(output_dir=output_dir, filename=f"{run_prefix}_4_wait_time_trend.png")
        self.plot_response_time_trend(output_dir=output_dir, filename=f"{run_prefix}_5_response_time_trend.png")
        self.plot_pod_history(output_dir=output_dir, filename=f"{run_prefix}_6_pod_history.png")
        self.plot_queue_history(output_dir=output_dir, filename=f"{run_prefix}_7_queue_history.png")