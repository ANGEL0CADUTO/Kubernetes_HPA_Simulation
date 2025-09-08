
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from matplotlib.ticker import MaxNLocator

# Import the Welford class from the welford.py file
from src.utils.welford import Welford



# Funzione helper per calcolare le medie, AGGIORNATA PER USARE WELFORD
def _calculate_overall_avg(times_by_type: dict):
    all_times = [t for times_list in times_by_type.values() for t in times_list]
    if not all_times:
        return 0
    # Utilizzo di Welford per calcolare la media
    welford_aggregator = Welford(np.array(all_times))
    return welford_aggregator.mean if welford_aggregator.count > 0 else 0


def _safe_legend(ax, loc='best', **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=loc, **kwargs)


class Plotter:
    def __init__(self, metrics, metrics_prio, config_module):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")


    def plot_queue_history(self, output_dir='plots', filename='queue_length_history.png'):
        print(f"Generazione storico coda -> {os.path.join(output_dir, filename)}")
        fig, ax = plt.subplots(figsize=(12, 6))
        if self.metrics.queue_length_history:
            times_no_prio, lengths_no_prio = zip(*self.metrics.queue_length_history)
            # Calcola la media usando Welford
            welford_no_prio = Welford(np.array(lengths_no_prio)) if lengths_no_prio else Welford()
            mean_no_prio = welford_no_prio.mean if welford_no_prio.count > 0 else 0
            ax.plot(times_no_prio, lengths_no_prio, color='r', linewidth=2, label='Senza Priorità', alpha=0.8)
            ax.axhline(mean_no_prio, color='darkred', linestyle='--', linewidth=1, label=f'Media Senza Priorità: {mean_no_prio:.2f}')
        if self.metrics_prio.queue_lengths:
            # Calcola la media usando Welford
            welford_prio = Welford(np.array(self.metrics_prio.queue_lengths)) if self.metrics_prio.queue_lengths else Welford()
            mean_prio = welford_prio.mean if welford_prio.count > 0 else 0
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', linewidth=2, label='Con Priorità', alpha=0.8)
            ax.axhline(mean_prio, color='darkblue', linestyle='--', linewidth=1, label=f'Media Con Priorità: {mean_prio:.2f}')
        ax.set_title("Evoluzione della Lunghezza della Coda nel Tempo"); ax.set_xlabel("Tempo di Simulazione (s)"); ax.set_ylabel("Numero di Richieste in Coda")
        _safe_legend(ax); ax.grid(True, linestyle='--', alpha=0.6)
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
                moving_avg = pd.Series(waits_senza).rolling(window=window_size).mean().dropna()
                ax.plot(times_senza[window_size - 1:], moving_avg, label='Senza Priorità (Media Mobile)', color='r', alpha=0.7)
            # Calcola la media totale usando Welford
            welford_total_senza = Welford(np.array(waits_senza)) if waits_senza else Welford()
            mean_total_senza = welford_total_senza.mean if welford_total_senza.count > 0 else 0
            ax.axhline(mean_total_senza, color='darkred', linestyle='--', linewidth=1, label=f'Media Totale Senza Priorità: {mean_total_senza:.2f}')
        all_waits_prio = []
        for req_type in sorted(self.metrics_prio.wait_times_by_req_type.keys(), key=lambda e: e.name):
            times = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            waits = self.metrics_prio.wait_times_by_req_type[req_type]
            all_waits_prio.extend(zip(times, waits))
        if all_waits_prio:
            all_waits_prio.sort(key=lambda x: x[0])
            times_prio, waits_prio = zip(*all_waits_prio)
            if len(waits_prio) >= window_size:
                moving_avg_prio = pd.Series(waits_prio).rolling(window=window_size).mean().dropna()
                ax.plot(times_prio[window_size - 1:], moving_avg_prio, label='Con Priorità (Media Mobile)', color='b', alpha=0.7)
            # Calcola la media totale usando Welford
            welford_total_prio = Welford(np.array(waits_prio)) if waits_prio else Welford()
            mean_total_prio = welford_total_prio.mean if welford_total_prio.count > 0 else 0
            ax.axhline(mean_total_prio, color='darkblue', linestyle='--', linewidth=1, label=f'Media Totale Con Priorità: {mean_total_prio:.2f}')
        ax.set_title("Andamento del Tempo di Attesa Medio (Media Mobile)")
        ax.set_xlabel("Tempo di Simulazione (s)")
        ax.set_ylabel("Tempo di Attesa Medio (s)")
        ax.grid(True, linestyle='--', alpha=0.6)
        _safe_legend(ax)
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
        _safe_legend(ax)
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
                # Calcola la media usando Welford
                welford_resp_no_prio = Welford(np.array(self.metrics.response_times_data[req_type])) if self.metrics.response_times_data[req_type] else Welford()
                mean_resp_no_prio = welford_resp_no_prio.mean if welford_resp_no_prio.count > 0 else 0
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': mean_resp_no_prio, 'Scenario': 'Senza Priorità'})
            if req_type in self.metrics_prio.response_times_by_req_type and self.metrics_prio.response_times_by_req_type[req_type]:
                # Calcola la media usando Welford
                welford_resp_prio = Welford(np.array(self.metrics_prio.response_times_by_req_type[req_type])) if self.metrics_prio.response_times_by_req_type[req_type] else Welford()
                mean_resp_prio = welford_resp_prio.mean if welford_resp_prio.count > 0 else 0
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': mean_resp_prio, 'Scenario': 'Con Priorità'})

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
        # Utilizzo di _calculate_overall_avg che ora usa Welford
        avg_response_prio = _calculate_overall_avg(self.metrics_prio.response_times_by_req_type)
        avg_wait_prio = _calculate_overall_avg(self.metrics_prio.wait_times_by_req_type)
        timeout_perc_prio = (total_timeouts_prio / total_generated_prio) * 100 if total_generated_prio > 0 else 0
        total_timeouts_no_prio = sum(self.metrics.requests_timed_out_data.values())
        # Utilizzo di _calculate_overall_avg che ora usa Welford
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
        _safe_legend(ax); ax.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True)); ax.set_ylim(bottom=0)
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


    def generate_comprehensive_report(self, output_dir='plots', run_prefix='run',
                                      peak_start=0, peak_end=0, base_load=0, peak_load=0):
        """
        Chiama tutti i metodi di plotting, passando loro i percorsi di output corretti.
        Questo metodo è pensato per generare report per UN SINGOLO set di metriche
        (es. quelle aggregate o quelle dell'ultima run), inclusa l'analisi dinamica.
        """
        print(f"\n--- Generazione Report Completo per '{run_prefix}' in '{output_dir}' ---")
        self.plot_comparison_dashboard(output_dir=output_dir, filename=f"{run_prefix}_1_dashboard.png")
        self.plot_served_by_type(output_dir=output_dir, filename=f"{run_prefix}_2_served_by_type.png")
        self.plot_loss_by_type(output_dir=output_dir, filename=f"{run_prefix}_3_loss_by_type.png")
        self.plot_wait_time_trend(output_dir=output_dir, filename=f"{run_prefix}_4_wait_time_trend.png")
        self.plot_response_time_trend(output_dir=output_dir, filename=f"{run_prefix}_5_response_time_trend.png")
        self.plot_pod_history(output_dir=output_dir, filename=f"{run_prefix}_6_pod_history.png")
        self.plot_queue_history(output_dir=output_dir, filename=f"{run_prefix}_7_queue_history.png")

        # Include i dashboard di analisi dinamica per questa singola run
        print(f"Generazione dashboard di analisi dinamica per {run_prefix} (parte del report completo)")
        self._plot_single_dynamic_analysis_dashboard(
            metrics_obj=self.metrics,
            metrics_prio_obj=self.metrics_prio,
            output_dir=output_dir,
            run_prefix=f"{run_prefix}_8_dynamic", # Prefisso diverso per i file di output
            peak_start=peak_start,
            peak_end=peak_end,
            base_load=base_load,
            peak_load=peak_load
        )


    def plot_replication_traces_per_scenario(self, all_results: dict, num_replications: int, output_dir='output/aggregated'):
        """
        Crea un GRAFICO SEPARATO PER OGNI SCENARIO DI TRAFFICO.
        Ogni grafico mostra l'andamento del tempo di risposta medio per ogni replica.
        """
        print("\n--- Generazione Grafici delle Tracce delle Repliche (uno per scenario) ---")

        # Ciclo principale: uno per ogni scenario (es. tasso_70, tasso_85)
        for scenario_name, replications in all_results.items():

            # 1. Creiamo una nuova figura per questo specifico scenario
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(21, 12), sharex=True)
            fig.suptitle(f'Tracce delle Repliche per Scenario: "{scenario_name.upper()}"', fontsize=18)

            # 2. Prepariamo i colori per le linee delle repliche
            colors = plt.cm.viridis(np.linspace(0, 1, num_replications))

            # 3. Cicliamo sulle repliche di questo scenario per disegnarle
            for i in range(num_replications):
                color = colors[i]
                rep_data = replications.get(i)
                if not rep_data: continue

                # Prendiamo il seed di questa replica
                seed = rep_data.get('seed', f'Replica {i+1}')
                label_text = f'Seed: {seed}'

                # --- Grafico 1: Modello Baseline ---
                metrics_base = rep_data['baseline']
                all_responses_base = metrics_base.get_all_response_times_with_timestamps()
                if all_responses_base:
                    times, values = zip(*all_responses_base)
                    cum_avg = np.cumsum(values) / np.arange(1, len(values) + 1)
                    ax1.plot(times, cum_avg, color=color, alpha=0.8, label=label_text)

                # --- Grafico 2: Modello con Priorità ---
                metrics_prio = rep_data['priority']
                all_responses_prio = metrics_prio.get_all_response_times_with_timestamps()
                if all_responses_prio:
                    times, values = zip(*all_responses_prio)
                    cum_avg = np.cumsum(values) / np.arange(1, len(values) + 1)
                    ax2.plot(times, cum_avg, color=color, alpha=0.8, label=label_text)

            # 4. Abbelliamo i grafici
            ax1.set_title('Modello Baseline (FIFO)')
            ax1.set_ylabel('Tempo di Risposta Medio (s)')
            ax1.grid(True, linestyle='--', alpha=0.6)
            _safe_legend(ax1)

            ax2.set_title('Modello Migliorato (Priorità)')
            ax2.set_xlabel('Tempo di Simulazione (s)')
            ax2.set_ylabel('Tempo di Risposta Medio (s)')
            ax2.grid(True, linestyle='--', alpha=0.6)
            _safe_legend(ax2)

            # 5. Salviamo il file con un nome specifico per lo scenario
            output_filename = f'replication_traces_{scenario_name}.png'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_path = os.path.join(output_dir, output_filename)
            fig.tight_layout(rect=[0, 0.03, 1, 0.96])
            fig.savefig(save_path, dpi=300)
            plt.close(fig)
            print(f"Grafico delle tracce per '{scenario_name}' salvato in: {save_path}")

    def _plot_single_dynamic_analysis_dashboard(self, metrics_obj, metrics_prio_obj, output_dir='plots', run_prefix='run', peak_start=0, peak_end=0, base_load=0, peak_load=0):
        """
        Crea due dashboard separati per l'analisi del carico dinamico per una singola replica/simulazione.
        """
        print(f"Generazione dashboard di analisi dinamica per {run_prefix} -> {os.path.join(output_dir, f'{run_prefix}_1_effectiveness_comparison.png')}")

        sim_time = self.config.SIMULATION_TIME
        window_size = 50
        # Y_AXIS_LIMIT = 5.0 # Rimosso limite fisso

        # Preparazione Dati Comuni - Using passed objects
        all_responses_base = metrics_obj.get_all_response_times_with_timestamps()
        base_times, base_ma = (None, None)
        if len(all_responses_base) > window_size:
            times, values = zip(*all_responses_base); base_times = pd.Series(times)
            base_ma = pd.Series(values).rolling(window=window_size).mean()

        prio_ma_data = {}
        for prio, history in metrics_prio_obj.response_times_history_by_prio.items():
            if len(history) > window_size:
                history.sort(key=lambda x: x[0]); times, values = zip(*history)
                prio_ma_data[prio] = {'times': pd.Series(times), 'ma': pd.Series(values).rolling(window=window_size).mean()}

        # Grafico 1: Confronto Efficacia
        fig1, ax1 = plt.subplots(figsize=(21, 7)) # Aumentato leggermente la larghezza
        fig1.suptitle(f"Efficacia della Prioritizzazione - {run_prefix}", fontsize=20, fontweight='bold')
        ax1.set_xlabel("Tempo di Simulazione (s)", fontsize=12); ax1.set_ylabel("Tempo di Risposta Medio (s)", fontsize=12)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)

        # Calcolo dinamico del limite Y per ax1
        max_y1 = 0.0
        if base_ma is not None and not base_ma.isnull().all():
            ax1.plot(base_times, base_ma, color='royalblue', linestyle='--', linewidth=2.5, label='Baseline (FIFO) - Media Globale')
            max_y1 = max(max_y1, base_ma.max())

        if self.config.Priority.HIGH in prio_ma_data and not prio_ma_data[self.config.Priority.HIGH]['ma'].isnull().all():
            data = prio_ma_data[self.config.Priority.HIGH]
            ax1.plot(data['times'], data['ma'], color='limegreen', linewidth=2, label='Migliorato - Priorità HIGH')
            max_y1 = max(max_y1, data['ma'].max())

        # Imposta il limite Y per ax1, con un valore predefinito se non ci sono dati
        ax1.set_ylim(bottom=0, top=max(max_y1 * 1.2, 1.0)) # Aumentato il padding a 1.2, e minimo a 1.0


        ax_load1 = ax1.twinx()
        load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]
        load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load1.plot(load_times, load_values, color='red', linestyle=':', linewidth=2.5, alpha=0.9, label='Tasso di Arrivo (Carico)')
        ax_load1.set_ylabel("Tasso di Arrivo (req/s)", color='red', fontsize=12); ax_load1.tick_params(axis='y', labelcolor='red')
        min_load_upper_bound = 0.05 if peak_load == 0 and base_load == 0 else 0.01
        ax_load1.set_ylim(bottom=0, top=max(peak_load * 1.2, base_load * 1.2, min_load_upper_bound))


        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_effectiveness_comparison.png", fig1)

        # Grafico 2: Analisi Dettagliata delle Priorità
        fig2, ax2 = plt.subplots(figsize=(21, 7)); # Aumentato leggermente la larghezza
        fig2.suptitle(f"Esperienza Utente per Classe di Priorità - {run_prefix}", fontsize=20, fontweight='bold')
        ax2.set_xlabel("Tempo di Simulazione (s)", fontsize=12); ax2.set_ylabel("Tempo di Risposta Medio (s)", fontsize=12)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)

        # Calcolo dinamico del limite Y per ax2
        max_y2 = 0.0
        colors = {self.config.Priority.HIGH: 'limegreen', self.config.Priority.MEDIUM: 'gold', self.config.Priority.LOW: 'salmon'}

        for prio_enum in self.config.Priority:
            data = prio_ma_data.get(prio_enum)
            if data is not None and not data['ma'].isnull().all():
                ax2.plot(data['times'], data['ma'], color=colors.get(prio_enum, 'black'), linewidth=2, label=f'Priorità {prio_enum.name}')
                max_y2 = max(max_y2, data['ma'].max())

        # Imposta il limite Y per ax2, con un valore predefinito se non ci sono dati
        ax2.set_ylim(bottom=0, top=max(max_y2 * 1.2, 1.0)) # Aumentato il padding a 1.2, e minimo a 1.0


        ax_load2 = ax2.twinx()
        ax_load2.plot(load_times, load_values, color='red', linestyle=':', linewidth=2.5, alpha=0.9, label='Tasso di Arrivo (Carico)')
        ax_load2.set_ylabel("Tasso di Arrivo (req/s)", color='red', fontsize=12); ax_load2.tick_params(axis='y', labelcolor='red')
        ax_load2.set_ylim(bottom=0, top=max(peak_load * 1.2, base_load * 1.2, min_load_upper_bound))


        lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig2.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_priority_experience.png", fig2)
    def plot_dynamic_analysis_for_replications(self, all_results: dict, output_dir='output/aggregated', arrival_scenarios: dict = None):
        """
        Itera attraverso tutti i risultati delle repliche e genera i dashboard di analisi dinamica per ciascuna.
        Args:
            all_results (dict): Dizionario contenente i risultati di tutte le repliche e scenari.
            output_dir (str): Directory di output per i grafici.
            arrival_scenarios (dict): Dizionario con le funzioni lambda per i tassi di arrivo per ogni scenario.
                                      Usato per determinare base_load e peak_load per il plot.
        """
        print("\n--- Generazione Dashboard di Analisi Dinamica per Ciascuna Replica ---")

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for scenario_name, replications in all_results.items():
            base_load = 0
            peak_load = 0
            peak_start = 0 # Default values
            peak_end = 0   # Default values

            if arrival_scenarios and scenario_name in arrival_scenarios:
                lambda_fn = arrival_scenarios[scenario_name]
                # Poiché le lambda nel main sono costanti, base_load = peak_load = valore della lambda.
                test_time = 0
                base_load = lambda_fn(test_time)
                peak_load = lambda_fn(test_time)
                # Se il tuo config_module ha un modo per ottenere peak_start/end per scenari specifici,
                # potresti recuperarli qui. Es:
                peak_start = getattr(self.config, f'PEAK_START_TIME', 0)
                peak_end = getattr(self.config, f'PEAK_END_TIME', 0)


            for replica_idx, rep_data in replications.items():
                if not rep_data or 'baseline' not in rep_data or 'priority' not in rep_data:
                    print(f"  Dati incompleti per lo scenario '{scenario_name}', replica {replica_idx}. Saltando.")
                    continue

                metrics_baseline = rep_data['baseline']
                metrics_priority = rep_data['priority']
                seed = rep_data.get('seed', f'repl{replica_idx+1}') # Get seed if available

                # Construct a more informative run_prefix for the filenames and titles
                current_run_prefix = f"{scenario_name}_seed_{seed}"

                self._plot_single_dynamic_analysis_dashboard(
                    metrics_obj=metrics_baseline,
                    metrics_prio_obj=metrics_priority,
                    output_dir=output_dir,
                    run_prefix=current_run_prefix,
                    peak_start=peak_start,
                    peak_end=peak_end,
                    base_load=base_load,
                    peak_load=peak_load
                )

    def generate_replication_reports(self, all_results: dict, num_replications: int, arrival_scenarios: dict, output_dir='output/aggregated'):
        """
        Genera un set completo di report per tutte le repliche di simulazione,
        inclusi i grafici delle tracce di replica e i dashboard di analisi dinamica.

        Args:
            all_results (dict): Dizionario contenente i risultati di tutte le repliche e scenari.
                                  Esempio: {'scenario_A': {0: {'seed': 123, 'baseline': Metrics, 'priority': MetricsWithPriority}}}
            num_replications (int): Numero totale di repliche per ogni scenario.
            arrival_scenarios (dict): Dizionario con le funzioni lambda per i tassi di arrivo per ogni scenario.
            output_dir (str): Directory di output per i grafici.
        """
        print(f"\n--- Generazione Report Complessivi per Repliche in '{output_dir}' ---")

        # 1. Genera i grafici delle tracce delle repliche (uno per scenario)
        self.plot_replication_traces_per_scenario(all_results, num_replications, output_dir)

        # 2. Genera i dashboard di analisi dinamica per ciascuna replica
        # I parametri di carico (base_load, peak_load) sono ora derivati all'interno di
        # plot_dynamic_analysis_for_replications per ogni scenario.
        self.plot_dynamic_analysis_for_replications(
            all_results=all_results,
            output_dir=output_dir,
            arrival_scenarios=arrival_scenarios # Passa il dizionario degli scenari di arrivo
        )
