# In src/analysis/plotter.py



import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority


matplotlib.use('Qt5Agg')
plt.style.use('ggplot')

class SteadyStatePlotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config

    def plot_steady_state_loss_ci(self, baseline_results, prio_results, output_dir, filename):
        """
        Crea un grafico a barre che confronta la probabilità di perdita steady-state
        con i rispettivi intervalli di confidenza.
        """
        print("Generazione grafico C.I. per probabilità di perdita...")
        scenarios = ['Senza Priorità', 'Con Priorità']
        means = [baseline_results['mean'], prio_results['mean']]
        half_widths = [baseline_results['half_width'], prio_results['half_width']]

        fig, ax = plt.subplots(figsize=(8, 6))

        colors = ['#ff0000', '#0000ff']
        bars = ax.bar(scenarios, means, yerr=half_widths, color=colors,
                      capsize=10, alpha=0.8, width=0.5)

        ax.set_title('Probabilità di Perdita (Steady State) con IC al 95%', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata')
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2) # Aggiunge spazio sopra
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax.bar_label(bars, fmt='%.4f', padding=3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_steady_state_times_by_type(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir):
        """
        Crea un dashboard che confronta i tempi medi (risposta e attesa) per tipo di richiesta,
        calcolati in steady-state con intervalli di confidenza.
        """
        print("Generazione grafici C.I. per tempi per tipo di richiesta...")

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
        fig.suptitle('Tempi Medi (Steady State) per Tipo con IC al 95%', fontsize=16)

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        for metric_name, ax in zip(['response', 'wait'], axes):
            plot_data = []
            for req_type in all_req_types:
                # Dati e analisi per il modello Baseline
                raw_data_baseline = self.metrics.response_times_history[req_type] if metric_name == 'response' else self.metrics.wait_times_history[req_type]
                ci_baseline = analyzer_baseline.calculate_batch_means_ci(raw_data_baseline, warmup, batches)
                if ci_baseline:
                    plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_baseline['mean'],
                                      'Errore': ci_baseline['half_width'], 'Scenario': 'Senza Priorità'})

                # Dati e analisi per il modello con Priorità
                raw_data_prio = self.metrics_prio.response_times_by_req_type[req_type] if metric_name == 'response' else self.metrics_prio.wait_times_by_req_type[req_type]
                timestamps_prio = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
                if len(timestamps_prio) == len(raw_data_prio):
                    data_with_ts_prio = sorted(zip(timestamps_prio, raw_data_prio), key=lambda x: x[0])
                    ci_prio = analyzer_prio.calculate_batch_means_ci(data_with_ts_prio, warmup, batches)
                    if ci_prio:
                        plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_prio['mean'],
                                          'Errore': ci_prio['half_width'], 'Scenario': 'Con Priorità'})

            if not plot_data: continue

            df = pd.DataFrame(plot_data)
            hue_order = ['Senza Priorità', 'Con Priorità']
            palette = ['#ff0000', '#0000ff']

            # Disegna il barplot principale
            sns.barplot(data=df, x='Categoria', y='Tempo Medio (s)', hue='Scenario',
                        order=category_names, hue_order=hue_order, palette=palette, ax=ax)

            # --- CORREZIONE APPLICATA QUI ---
            # Calcoliamo le posizioni x per le barre e aggiungiamo le barre di errore
            num_categories = len(category_names)
            x_positions = np.arange(num_categories)
            width = 0.4 # Larghezza di ogni singola barra

            for i, scenario in enumerate(hue_order):
                offset = -width / 2 if i == 0 else width / 2
                subset = df[df['Scenario'] == scenario].set_index('Categoria').loc[category_names]
                ax.errorbar(x_positions + offset, subset['Tempo Medio (s)'], yerr=subset['Errore'],
                            fmt='none', c='black', capsize=4, elinewidth=1)
            # --------------------------------

            # Estetica del grafico
            title_str = f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio"
            ax.set_title(title_str)
            ax.set_xlabel('') # Rimuoviamo l'etichetta x per non appesantire
            ax.set_ylabel('Tempo Medio (s)')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.legend(title='Scenario').remove()

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', title='Scenario')

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "steady_state_times_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_steady_state_loss_by_type_ci(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir):
        """
        Crea un grafico a barre che confronta la probabilità di perdita per tipo di richiesta,
        calcolata in steady-state con intervalli di confidenza.
        """
        print("Generazione grafico C.I. per probabilità di perdita per tipo...")

        fig, ax = plt.subplots(1, 1, figsize=(14, 8)) # Ho reso il grafico un po' più grande
        fig.suptitle('Probabilità di Perdita (Steady State) per Tipo con IC al 95%', fontsize=16)

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        plot_data = []
        for req_type in all_req_types:
            # Analisi Baseline
            stream_baseline = self.metrics.get_outcomes_by_type_as_binary_stream(req_type)
            ci_baseline = analyzer_baseline.calculate_batch_means_ci(stream_baseline, warmup, batches)
            if ci_baseline:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Probabilità di Perdita': ci_baseline['mean'],
                                  'Errore': ci_baseline['half_width'], 'Scenario': 'Senza Priorità'})

            # Analisi Priorità
            stream_prio = self.metrics_prio.get_outcomes_by_type_as_binary_stream(req_type)
            ci_prio = analyzer_prio.calculate_batch_means_ci(stream_prio, warmup, batches)
            if ci_prio:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Probabilità di Perdita': ci_prio['mean'],
                                  'Errore': ci_prio['half_width'], 'Scenario': 'Con Priorità'})

        if not plot_data:
            print("Nessun dato sufficiente per generare il grafico delle perdite per tipo.")
            plt.close(fig)
            return

        df = pd.DataFrame(plot_data)
        hue_order = ['Senza Priorità', 'Con Priorità']
        palette = ['#ff0000', '#0000ff']

        sns.barplot(data=df, x='Categoria', y='Probabilità di Perdita', hue='Scenario',
                    order=category_names, hue_order=hue_order, palette=palette, ax=ax)

        num_categories = len(category_names)
        x_positions = np.arange(num_categories)
        width = 0.4

        for i, scenario in enumerate(hue_order):
            offset = -width / 2 if i == 0 else width / 2
            subset = df[df['Scenario'] == scenario].set_index('Categoria').reindex(category_names)

            # Aggiungi barre di errore
            ax.errorbar(x_positions + offset, subset['Probabilità di Perdita'], yerr=subset['Errore'],
                        fmt='none', c='black', capsize=4, elinewidth=1.5)

            # --- Aggiungiamo valore dell'intervallo ---
            for j, (cat_name, row) in enumerate(subset.iterrows()):
                mean = row['Probabilità di Perdita']
                error = row['Errore']

                # Calcola i limiti dell'intervallo
                lower_bound = max(0, mean - error) # Evita valori negativi
                upper_bound = mean + error

                ci_text = f"[{lower_bound:.4f},\n {upper_bound:.4f}]" # Vado a capo per leggibilità

                # Posiziona il testo sopra la barra di errore
                # Usiamo `upper_bound` come coordinata y di base
                ax.text(x_positions[j] + offset, upper_bound, ci_text,
                        ha='center', va='bottom', fontsize=8, color='darkslategrey')
            # -------------------------------------------------------------

        # Estetica
        ax.set_xlabel('Tipo di Richiesta')
        ax.set_ylabel('Probabilità di Perdita Stimata')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.legend(title='Scenario')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)

        # Aumenta dinamicamente lo spazio superiore per far posto al testo
        max_y_lim = df['Probabilità di Perdita'].max() + df['Errore'].max()
        ax.set_ylim(top=max_y_lim * 1.3)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "steady_state_loss_by_type_ci.png")
        plt.savefig(save_path, dpi=300)
        plt.show()



    def plot_convergence_prio_by_type(self, output_dir="plots/transient_analysis"):
        """
        Crea un grafico che mostra la convergenza del tempo di risposta medio cumulativo
        per ogni tipo di richiesta nello scenario CON PRIORITÀ.
        """
        print("Generazione grafico di convergenza per tipo (con priorità)...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Itera su ogni tipo di richiesta presente nelle metriche con priorità
        all_req_types = sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name)

        for req_type in all_req_types:
            # 2. Recupera i dati corretti dalla classe MetricsWithPriority
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])

            # Controlla che i dati siano consistenti
            if not response_times or len(response_times) != len(timestamps):
                continue

            # 3. Combina i dati in tuple (timestamp, valore) e ordinali cronologicamente
            history = sorted(zip(timestamps, response_times), key=lambda x: x[0])

            # Separa di nuovo per il calcolo
            sorted_timestamps = [t for t, v in history]
            sorted_values = [v for t, v in history]

            # 4. Calcola e disegna la media cumulativa
            cumulative_avg = np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1)
            ax.plot(sorted_timestamps, cumulative_avg, label=f'{req_type.name}')

        # 5. Estetica del grafico (titoli, etichette, legenda)
        ax.set_title('Analisi della Convergenza per Tipo di Richiesta (Con Priorità)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)

        # Rendi la legenda più leggibile se ci sono molte linee
        if len(all_req_types) > 5:
            ax.legend(title='Tipo di Richiesta', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=(0, 0, 0.85, 1)) # Aggiusta lo spazio per la legenda esterna
        else:
            ax.legend(title='Tipo di Richiesta')
            plt.tight_layout()

        # Salvataggio
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "prio_convergence_by_type.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_convergence_baseline_overall(self, output_dir="plots/transient_analysis"):
        """
        Crea un grafico che mostra la convergenza del tempo di risposta medio cumulativo
        per l'intero scenario baseline. Utile per identificare il warm-up period.
        """
        print("Generazione grafico di convergenza generale (baseline)...")

        # 1. Raccogli e ordina tutti i dati di risposta dalla metrica baseline
        all_responses = self.metrics.get_all_response_times_with_timestamps()

        if not all_responses:
            print("Nessun dato di risposta per l'analisi di convergenza baseline.")
            return

        # 2. Calcola la media cumulativa (CUSUM)
        timestamps = [t for t, v in all_responses]
        values = [v for t, v in all_responses]
        cumulative_avg = np.cumsum(values) / np.arange(1, len(values) + 1)

        # 3. Disegna il grafico
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(timestamps, cumulative_avg, color='r', label='Tempo Risposta Medio Cumulativo')

        # Estetica
        ax.set_title('Analisi della Convergenza del Tempo di Risposta Medio (Baseline)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend()

        # Aggiungi una linea verticale per indicare il warm-up period scelto
        warmup_period = 250 # Assumendo 250s come da discussione
        ax.axvline(x=warmup_period, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_period}s)')
        ax.legend()

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "baseline_convergence_overall.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        # Aggiungi questo metodo alla classe Plotter

    def plot_convergence_baseline_by_type(self, output_dir="plots/transient_analysis"):
        """
        Crea un grafico che mostra la convergenza del tempo di risposta medio cumulativo
        per ogni tipo di richiesta nello scenario baseline.
        """
        print("Generazione grafico di convergenza per tipo (baseline)...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Itera su ogni tipo di richiesta presente nelle metriche
        for req_type, history in self.metrics.response_times_history.items():
            if not history:
                continue

            # Ordina i dati per questo tipo di richiesta
            history.sort(key=lambda x: x[0])
            timestamps = [t for t, v in history]
            values = [v for t, v in history]

            # Calcola e disegna la media cumulativa
            cumulative_avg = np.cumsum(values) / np.arange(1, len(values) + 1)
            ax.plot(timestamps, cumulative_avg, label=f'{req_type.name}')

        # Estetica
        ax.set_title('Analisi della Convergenza per Tipo di Richiesta (Baseline)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "baseline_convergence_by_type.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_wait_time_comparison_trend(self, output_dir="plots/comparison"):
        """
        Confronta l'evoluzione del tempo di attesa medio cumulativo
        tra lo scenario baseline e quello con priorità.
        """
        print("Generazione grafico di confronto andamento tempo di attesa...")

        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. Dati e curva per la Baseline
        all_waits_baseline = []
        for history in self.metrics.wait_times_history.values():
            all_waits_baseline.extend(history)
        all_waits_baseline.sort(key=lambda x: x[0])

        if all_waits_baseline:
            times_b, values_b = zip(*all_waits_baseline)
            cusum_b = np.cumsum(values_b) / np.arange(1, len(values_b) + 1)
            ax.plot(times_b, cusum_b, color='r', label='Senza Priorità')

        # 2. Dati e curva per lo scenario con Priorità
        all_waits_prio = []
        for prio, history in self.metrics_prio.wait_times_by_priority.items():
            timestamps = self.metrics_prio.completion_timestamps_by_priority[prio]
            if len(timestamps) == len(history):
                all_waits_prio.extend(zip(timestamps, history))
        all_waits_prio.sort(key=lambda x: x[0])

        if all_waits_prio:
            times_p, values_p = zip(*all_waits_prio)
            cusum_p = np.cumsum(values_p) / np.arange(1, len(values_p) + 1)
            ax.plot(times_p, cusum_p, color='b', label='Con Priorità')

        # Estetica
        ax.set_title('Confronto Evoluzione del Tempo di Attesa Medio', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "wait_time_trend_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_times_by_request_type_grid(self, output_dir="plots/comparison"):
        """
        Crea una griglia di grafici, uno per ogni tipo di richiesta.
        Ogni grafico confronta le curve cumulative di tempo di attesa e di risposta
        per i due scenari (baseline vs priorità).
        """
        print("Generazione griglia di confronto per tipo di richiesta...")

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        num_req_types = len(all_req_types)

        # Calcola le dimensioni della griglia (es. 2x3 o 3x2)
        ncols = 3
        nrows = int(np.ceil(num_req_types / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), sharex=True, sharey=True)
        axes = axes.flatten() # Appiattisce l'array di assi per una facile iterazione

        for i, req_type in enumerate(all_req_types):
            ax = axes[i]

            # --- Dati Baseline ---
            # Tempo di risposta
            resp_b = sorted(self.metrics.response_times_history.get(req_type, []), key=lambda x: x[0])
            if resp_b:
                times_rb, values_rb = zip(*resp_b)
                ax.plot(times_rb, np.cumsum(values_rb) / np.arange(1, len(values_rb)+1),
                        color='salmon', linestyle='--', label='Risposta (Baseline)')

            # Tempo di attesa
            wait_b = sorted(self.metrics.wait_times_history.get(req_type, []), key=lambda x: x[0])
            if wait_b:
                times_wb, values_wb = zip(*wait_b)
                ax.plot(times_wb, np.cumsum(values_wb) / np.arange(1, len(values_wb)+1),
                        color='red', label='Attesa (Baseline)')

            # --- Dati con Priorità ---
            # Tempo di risposta
            times_rp = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            values_rp = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            if times_rp and len(times_rp) == len(values_rp):
                resp_p = sorted(zip(times_rp, values_rp), key=lambda x: x[0])
                times_rp, values_rp = zip(*resp_p)
                ax.plot(times_rp, np.cumsum(values_rp) / np.arange(1, len(values_rp)+1),
                        color='lightblue', linestyle='--', label='Risposta (Priorità)')

            # Tempo di attesa
            values_wp = self.metrics_prio.wait_times_by_req_type.get(req_type, [])
            if times_rp and len(times_rp) == len(values_wp):
                wait_p = sorted(zip(times_rp, values_wp), key=lambda x: x[0])
                times_wp, values_wp = zip(*wait_p)
                ax.plot(times_wp, np.cumsum(values_wp) / np.arange(1, len(values_wp)+1),
                        color='blue', label='Attesa (Priorità)')

            ax.set_title(req_type.name.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

        # Nasconde gli assi vuoti se il numero di grafici non riempie la griglia
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.supxlabel('Tempo di Simulazione (s)', y=0.02)
        fig.supylabel('Tempo Medio Cumulativo (s)', x=0.02)

        plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "times_grid_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        # Incolla questi metodi dentro la classe SteadyStatePlotter

    def plot_pod_history_steady_state(self, output_dir, filename="ss_pod_history.png"):
        """
        Plotta l'evoluzione del numero di Pod nel tempo per la simulazione steady-state.
        Utile per visualizzare il comportamento dell'HPA e il transitorio.
        """
        print("Generazione grafico storico dei Pod (Steady-State)...")
        fig, ax = plt.subplots(figsize=(14, 7))

        # Scenario Baseline
        if self.metrics.pod_count_history:
            times_b, pods_b = zip(*self.metrics.pod_count_history)
            ax.plot(times_b, pods_b, color='r', label='Senza Priorità', alpha=0.8, linewidth=1.5)

        # Scenario con Priorità
        if self.metrics_prio.pod_counts:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', label='Con Priorità', alpha=0.8, linewidth=1.5)

        # Aggiungi una linea verticale per indicare il warm-up period
        warmup = self.config.WARM_UP_TO_STEADY
        ax.axvline(x=warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup}s)')

        ax.set_title('Evoluzione del Numero di Pod (Simulazione Lunga)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_queue_history_steady_state(self, output_dir, filename="ss_queue_history.png"):
        """
        Plotta l'evoluzione della lunghezza della coda nel tempo per la simulazione steady-state.
        """
        print("Generazione grafico storico della Coda (Steady-State)...")
        fig, ax = plt.subplots(figsize=(14, 7))

        # Scenario Baseline
        if self.metrics.queue_length_history:
            times_b, queue_b = zip(*self.metrics.queue_length_history)
            ax.plot(times_b, queue_b, color='r', label='Senza Priorità', alpha=0.7, linewidth=1.5)
            # Media dopo il warm-up
            steady_queue_b = [q for t, q in self.metrics.queue_length_history if t >= self.config.WARM_UP_TO_STEADY]
            if steady_queue_b:
                ax.axhline(np.mean(steady_queue_b), color='darkred', linestyle='--', label=f'Media Steady-State (Baseline): {np.mean(steady_queue_b):.2f}')


        # Scenario con Priorità
        if self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', label='Con Priorità', alpha=0.7, linewidth=1.5)
            # Media dopo il warm-up
            steady_queue_p = [q for t, q in zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths) if t >= self.config.WARM_UP_TO_STEADY]
            if steady_queue_p:
                ax.axhline(np.mean(steady_queue_p), color='darkblue', linestyle='--', label=f'Media Steady-State (Priorità): {np.mean(steady_queue_p):.2f}')

        warmup = self.config.WARM_UP_TO_STEADY
        ax.axvline(x=warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup}s)')

        ax.set_title('Evoluzione della Lunghezza della Coda (Simulazione Lunga)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Richieste in Coda')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_convergence_comparison_overall(self, output_dir, filename="ss_overall_convergence.png"):
        """
        Confronta le curve di convergenza del tempo di risposta medio totale
        tra lo scenario baseline e quello con priorità.
        """
        print("Generazione grafico di confronto convergenza generale...")
        fig, ax = plt.subplots(figsize=(14, 7))

        # Dati Baseline
        all_responses_b = self.metrics.get_all_response_times_with_timestamps()
        if all_responses_b:
            times_b, values_b = zip(*all_responses_b)
            cusum_b = np.cumsum(values_b) / np.arange(1, len(values_b) + 1)
            ax.plot(times_b, cusum_b, color='r', label='Senza Priorità', linewidth=2)

        # Dati Priorità
        all_responses_p = self.metrics_prio.get_all_response_times_with_timestamps()
        if all_responses_p:
            times_p, values_p = zip(*all_responses_p)
            cusum_p = np.cumsum(values_p) / np.arange(1, len(values_p) + 1)
            ax.plot(times_p, cusum_p, color='b', label='Con Priorità', linewidth=2)

        warmup = self.config.WARM_UP_TO_STEADY
        ax.axvline(x=warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup}s)')

        ax.set_title('Confronto Convergenza del Tempo di Risposta Medio Totale', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.legend(title='Scenario')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_variance_trend(self, output_dir, filename="ss_variance_trend.png"):
        """
        Plotta la deviazione standard calcolata su una finestra mobile per visualizzare
        la stabilizzazione della variabilità del sistema.
        """
        print("Generazione grafico andamento della varianza (finestra mobile)...")
        fig, ax = plt.subplots(figsize=(14, 7))
        window_size = 500  # Finestra abbastanza grande per smussare le fluttuazioni

        # Dati Baseline
        all_responses_b = self.metrics.get_all_response_times_with_timestamps()
        if len(all_responses_b) > window_size:
            times_b, values_b = zip(*all_responses_b)
            # Calcola la deviazione standard mobile usando pandas per semplicità
            moving_std_b = pd.Series(values_b).rolling(window=window_size).std()
            ax.plot(times_b[window_size-1:], moving_std_b[window_size-1:], color='r', label='Senza Priorità', alpha=0.8)

        # Dati Priorità
        all_responses_p = self.metrics_prio.get_all_response_times_with_timestamps()
        if len(all_responses_p) > window_size:
            times_p, values_p = zip(*all_responses_p)
            moving_std_p = pd.Series(values_p).rolling(window=window_size).std()
            ax.plot(times_p[window_size-1:], moving_std_p[window_size-1:], color='b', label='Con Priorità', alpha=0.8)

        warmup = self.config.WARM_UP_TO_STEADY
        ax.axvline(x=warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup}s)')

        ax.set_title(f'Stabilizzazione della Variabilità (Dev. Std. su Finestra Mobile di {window_size} campioni)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.legend(title='Scenario')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=0) # La deviazione standard non può essere negativa

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300)
        plt.show()


    def generate_steady_state_report(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir="plots/steady_state"):
        """
        Metodo principale che orchestra la generazione di tutti i grafici
        di analisi steady-state.
        """
        print(f"\n--- Generazione Report Completo Steady-State in '{output_dir}' ---")

        # --- SEZIONE 1: Analisi del Transitorio e Convergenza ---
        transient_output_dir = os.path.join(output_dir, "transient_analysis")
        print(f"\n--- 1. Analisi del Transitorio (output in '{transient_output_dir}') ---")
        self.plot_pod_history_steady_state(transient_output_dir)
        self.plot_queue_history_steady_state(transient_output_dir)
        self.plot_convergence_comparison_overall(transient_output_dir)
        self.plot_variance_trend(transient_output_dir)
        # I grafici di convergenza per tipo sono ancora utili
        self.plot_convergence_baseline_by_type(transient_output_dir)
        self.plot_convergence_prio_by_type(transient_output_dir)


        # --- SEZIONE 2: Stime a Regime Permanente (Batch Means) ---
        print(f"\n--- 2. Stime a Regime Permanente (output in '{output_dir}') ---")

        # Calcolo e plot della P_loss aggregata
        all_outcomes_baseline = self.metrics.get_all_outcomes_as_binary_stream()
        baseline_loss_results = analyzer_baseline.calculate_batch_means_ci(all_outcomes_baseline, warmup, batches) if all_outcomes_baseline else None

        all_outcomes_prio = self.metrics_prio.get_all_outcomes_as_binary_stream()
        prio_loss_results = analyzer_prio.calculate_batch_means_ci(all_outcomes_prio, warmup, batches) if all_outcomes_prio else None

        if baseline_loss_results and prio_loss_results:
            self.plot_steady_state_loss_ci(baseline_loss_results, prio_loss_results, output_dir, "ss_1_overall_loss_ci.png")

        # Plot dei tempi e delle perdite per tipo
        self.plot_steady_state_times_by_type(analyzer_baseline, analyzer_prio, warmup, batches, output_dir)
        self.plot_steady_state_loss_by_type_ci(analyzer_baseline, analyzer_prio, warmup, batches, output_dir)


        # --- SEZIONE 3: Confronti Aggiuntivi ---
        comparison_output_dir = os.path.join(output_dir, "comparison")
        print(f"\n--- 3. Confronti Aggiuntivi (output in '{comparison_output_dir}') ---")
        self.plot_wait_time_comparison_trend(output_dir=comparison_output_dir)
        self.plot_times_by_request_type_grid(output_dir=comparison_output_dir)
