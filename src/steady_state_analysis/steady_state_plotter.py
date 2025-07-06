# In src/analysis/plotter.py



import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.config import RequestType

matplotlib.use('Qt5Agg')
plt.style.use('ggplot')

class SteadyStatePlotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config

        # Creiamo una mappa di colori una sola volta, per garantire coerenza
        # tra tutti i grafici.
        self.req_type_colors = {
            RequestType.ADD_TO_CART: '#FF1493',  # DeepPink
            RequestType.ANALYTICS:   '#00BFFF',  # DeepSkyBlue
            RequestType.CHECKOUT:    '#32CD32',  # LimeGreen
            RequestType.LOGIN:       '#FFD700',  # Gold
            RequestType.NAVIGATION:  '#9400D3'   # DarkViolet
        }

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

    # In src/steady_state_analysis/steady_state_plotter.py


    # In src/steady_state_analysis/steady_state_plotter.py

    def plot_steady_state_times_by_type(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir):
        """
        Crea un dashboard che confronta i tempi medi (risposta e attesa) per tipo di richiesta,
        calcolati in steady-state con intervalli di confidenza.
        VERSione GARANTITA per risolvere l'errore di layout.
        """
        print("Generazione grafici C.I. per tempi per tipo di richiesta...")

        # --- FIGURA E ASSI ---
        # Usiamo una dimensione più ragionevole e lasciamo che il layout faccia il suo lavoro
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        # NON usiamo suptitle qui. Lo aggiungeremo alla fine.

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        for metric_name, ax in zip(['response', 'wait'], axes):
            # La logica di raccolta dati rimane IDENTICA alla tua.
            plot_data = []
            for req_type in all_req_types:
                raw_data_baseline = self.metrics.response_times_history[req_type] if metric_name == 'response' else self.metrics.wait_times_history[req_type]
                ci_baseline = analyzer_baseline.calculate_batch_means_ci(raw_data_baseline, warmup, batches)
                if ci_baseline:
                    plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_baseline['mean'], 'Errore': ci_baseline['half_width'], 'Scenario': 'Senza Priorità'})

                raw_data_prio = self.metrics_prio.response_times_by_req_type[req_type] if metric_name == 'response' else self.metrics_prio.wait_times_by_req_type[req_type]
                timestamps_prio = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
                if len(timestamps_prio) == len(raw_data_prio):
                    data_with_ts_prio = sorted(zip(timestamps_prio, raw_data_prio), key=lambda x: x[0])
                    ci_prio = analyzer_prio.calculate_batch_means_ci(data_with_ts_prio, warmup, batches)
                    if ci_prio:
                        plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_prio['mean'], 'Errore': ci_prio['half_width'], 'Scenario': 'Con Priorità'})

            if not plot_data: continue

            df = pd.DataFrame(plot_data)
            hue_order = ['Senza Priorità', 'Con Priorità']
            palette = ['#ff0000', '#0000ff']

            # La logica di plot e annotazione rimane IDENTICA alla tua.
            sns.barplot(data=df, x='Categoria', y='Tempo Medio (s)', hue='Scenario',
                        order=category_names, hue_order=hue_order, palette=palette, ax=ax, dodge=True)

            num_categories = len(category_names)
            x_positions = np.arange(num_categories)
            width = 0.4

            for i, scenario in enumerate(hue_order):
                offset = -width / 2 if i == 0 else width / 2
                subset = df[df['Scenario'] == scenario].set_index('Categoria').reindex(category_names)

                if subset['Tempo Medio (s)'].isnull().all(): continue

                y_coords = subset['Tempo Medio (s)'].fillna(0)
                errors = subset['Errore'].fillna(0)

                ax.errorbar(x_positions + offset, y_coords, yerr=errors,
                            fmt='none', c='black', capsize=5, elinewidth=1.2)

                for k, (cat, row) in enumerate(subset.iterrows()):
                    if pd.isna(row['Tempo Medio (s)']): continue
                    mean_val = row['Tempo Medio (s)']
                    error_val = row['Errore']

                    ax.text(x_positions[k] + offset, mean_val / 2, f'{mean_val:.3f}',
                            ha='center', va='center', color='white', fontsize=7.5, weight='bold')

                    upper_bound = mean_val + error_val
                    ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"

                    padding = ax.get_ylim()[1] * 0.02
                    ax.text(x_positions[k] + offset, upper_bound + padding, ci_text,
                            ha='center', va='center', fontsize=7, color='black')

            # L'estetica del singolo grafico rimane IDENTICA alla tua.
            title_str = f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio"
            ax.set_title(title_str, fontsize=16)
            ax.set_xlabel('')
            ax.set_ylabel('Tempo Medio (s)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
            ax.legend(title='Scenario').remove() # Nascondiamo la legenda del subplot

            # L'aumento del margine superiore rimane IDENTICO al tuo.
            if not df.empty:
                max_y_lim = (df['Tempo Medio (s)'] + df['Errore']).max()
                if not pd.isna(max_y_lim):
                    ax.set_ylim(top=max_y_lim * 1.35)

        # --- GESTIONE CONTROLLATA DEL LAYOUT FINALE (La parte che corregge il problema) ---

        # 1. Aggiungiamo il titolo generale
        fig.suptitle('Tempi Medi (Steady State) per Tipo con IC al 95%', fontsize=20, fontweight='bold')

        # 2. Creiamo una legenda unica per l'intera figura, posizionandola in modo sicuro
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), title='Scenario', fontsize=12)

        # 3. Chiamiamo tight_layout() SENZA argomenti, per una prima sistemazione automatica
        plt.tight_layout()

        # 4. Aggiustiamo manualmente gli spazi per accomodare titolo ed etichette
        #    Questo è il passo cruciale e più robusto di `rect`.
        fig.subplots_adjust(top=0.88, bottom=0.22, left=0.07, right=0.98)

        os.makedirs(output_dir, exist_ok=True)
        # Salviamo con un nome file diverso per non sovrascrivere il vecchio tentativo
        save_path = os.path.join(output_dir, "steady_state_times_comparison_fixed.png")
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
        """AGGIORNATO con colori consistenti e vivaci."""
        print("Generazione grafico di convergenza per tipo (baseline)...")
        fig, ax = plt.subplots(figsize=(12, 7))

        for req_type, history in self.metrics.response_times_history.items():
            if not history: continue
            history.sort(key=lambda x: x[0])
            timestamps, values = zip(*history)
            cumulative_avg = np.cumsum(values) / np.arange(1, len(values) + 1)
            # Usa la mappa di colori definita in __init__
            ax.plot(timestamps, cumulative_avg, label=f'{req_type.name}', color=self.req_type_colors[req_type], linewidth=2)

        ax.set_title('Analisi della Convergenza per Tipo di Richiesta (Baseline)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "baseline_convergence_by_type_colored.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_convergence_prio_by_type(self, output_dir="plots/transient_analysis"):
        """AGGIORNATO con colori consistenti e vivaci."""
        print("Generazione grafico di convergenza per tipo (con priorità)...")
        fig, ax = plt.subplots(figsize=(12, 7))

        all_req_types = sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name)
        for req_type in all_req_types:
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if not response_times or len(response_times) != len(timestamps): continue

            history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
            sorted_timestamps, sorted_values = zip(*history)
            cumulative_avg = np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1)
            # Usa la stessa mappa di colori per coerenza
            ax.plot(sorted_timestamps, cumulative_avg, label=f'{req_type.name}', color=self.req_type_colors[req_type], linewidth=2)

        ax.set_title('Analisi della Convergenza per Tipo di Richiesta (Con Priorità)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "prio_convergence_by_type_colored.png")
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
        """AGGIORNATO con scala logaritmica."""
        print("Generazione grafico storico della Coda (Steady-State)...")
        fig, ax = plt.subplots(figsize=(14, 7))

        # ... (la logica di plot rimane identica) ...
        if self.metrics.queue_length_history:
            times_b, queue_b = zip(*self.metrics.queue_length_history)
            ax.plot(times_b, queue_b, color='r', label='Senza Priorità', alpha=0.7, linewidth=1.5)
            steady_queue_b = [q for t, q in self.metrics.queue_length_history if t >= self.config.WARM_UP_TO_STEADY]
            if steady_queue_b: ax.axhline(np.mean(steady_queue_b), color='darkred', linestyle='--', label=f'Media Steady-State (Baseline): {np.mean(steady_queue_b):.2f}')
        if self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', label='Con Priorità', alpha=0.7, linewidth=1.5)
            steady_queue_p = [q for t, q in zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths) if t >= self.config.WARM_UP_TO_STEADY]
            if steady_queue_p: ax.axhline(np.mean(steady_queue_p), color='darkblue', linestyle='--', label=f'Media Steady-State (Priorità): {np.mean(steady_queue_p):.2f}')

        warmup = self.config.WARM_UP_TO_STEADY
        ax.axvline(x=warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup}s)')

        # ###############################################################
        # ## MODIFICA 3.1: SCALA LOGARITMICA PER LA LUNGHEZZA DELLA CODA ##
        # ###############################################################
        # Commenta/Decommenta questa riga per attivare/disattivare la scala logaritmica.
        ax.set_yscale('log')
        # La scala logaritmica non può mostrare lo zero. Se la tua coda a volte è zero,
        # potremmo voler impostare un limite inferiore molto piccolo per evitare problemi.
        # ax.set_ylim(bottom=0.1) # Decommenta se necessario
        # ###############################################################

        ax.set_title('Evoluzione della Lunghezza della Coda (Simulazione Lunga)', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Richieste in Coda (Scala Log)')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.6)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "ss_queue_history_log.png") # Nome file diverso
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

    def plot_steady_state_throughput_ci(self, analyzer_baseline, analyzer_prio, output_dir):
        """
        Crea un grafico che confronta il numero di richieste soddisfatte per tipo,
        mostrando i totali, la differenza percentuale e gli intervalli di confidenza sul throughput.
        """
        print("Generazione grafico di confronto delle richieste soddisfatte (throughput)...")

        warmup = self.config.WARM_UP_TO_STEADY
        batches = self.config.NUM_BATCHES

        fig, ax = plt.subplots(figsize=(16, 9)) # Formato 16:9, più largo
        fig.suptitle("Confronto Richieste Servite per Tipo - Steady State", fontsize=24, fontweight='bold')

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        plot_data = []
        for req_type in all_req_types:
            # Analisi Baseline
            timestamps_b = sorted([ts for ts, rt in self.metrics.response_times_history.get(req_type, [])])
            results_b = analyzer_baseline.calculate_throughput_ci(timestamps_b, warmup, batches)
            if results_b:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Conteggio': results_b['total_count'],
                                  'CI': results_b['ci'], 'Scenario': 'Senza Priorità'})

            # Analisi Priorità
            timestamps_p = sorted(self.metrics_prio.completion_timestamps_by_req_type.get(req_type, []))
            results_p = analyzer_prio.calculate_throughput_ci(timestamps_p, warmup, batches)
            if results_p:
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Conteggio': results_p['total_count'],
                                  'CI': results_p['ci'], 'Scenario': 'Con Priorità'})

        if not plot_data:
            print("Dati insufficienti per il grafico del throughput.")
            return

        df = pd.DataFrame(plot_data)
        hue_order = ['Senza Priorità', 'Con Priorità']
        palette = ['#ff0000', '#0000ff']

        # Disegniamo le barre usando il CONTEGGIO totale
        sns.barplot(data=df, x='Categoria', y='Conteggio', hue='Scenario',
                    order=category_names, hue_order=hue_order, palette=palette, ax=ax)

        # Aggiungi le etichette numeriche (conteggio) sopra ogni barra
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                        textcoords='offset points')

        # Aggiungi le etichette con la differenza percentuale in alto
        y_top = ax.get_ylim()[1]
        for i, cat_name in enumerate(category_names):
            base_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Senza Priorità')]
            prio_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Con Priorità')]

            if not base_row.empty and not prio_row.empty:
                base_count = base_row.iloc[0]['Conteggio']
                prio_count = prio_row.iloc[0]['Conteggio']

                if base_count > 0:
                    delta_perc = ((prio_count - base_count) / base_count) * 100
                    sign = '+' if delta_perc >= 0 else ''
                    color = 'green' if delta_perc >= 0 else 'red'

                    ax.text(i, y_top * 0.95, f'Δ: {sign}{delta_perc:.1f}%', ha='center', va='center',
                            fontsize=14, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9))

        # Estetica finale
        ax.set_title("Richieste Servite con Successo per Tipo", fontsize=18)
        ax.set_xlabel("Tipo di Richiesta", fontsize=14)
        ax.set_ylabel("Numero di Richieste Servite", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title='Scenario', loc='center left', bbox_to_anchor=(0.01, 0.5))
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_facecolor('#f0f0f0')
        fig.set_facecolor('#f0f0f0')

        # Aumenta lo spazio superiore per le etichette Delta
        ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "steady_state_throughput_comparison.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        # Incolla questo nuovo metodo dentro la classe SteadyStatePlotter

    def plot_batch_mean_queue_trend(self, warmup, batches, output_dir, filename="ss_queue_batch_means_trend.png"):
        """
        Crea un grafico che mostra l'evoluzione delle medie dei batch della lunghezza della coda,
        per visualizzare la stabilità del sistema a regime permanente.
        """
        print("Generazione grafico trend delle medie dei batch della coda...")
        fig, ax = plt.subplots(figsize=(14, 7))

        scenarios = {
            "Senza Priorità": self.metrics.queue_length_history,
            "Con Priorità": list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths))
        }
        colors = {"Senza Priorità": 'r', "Con Priorità": 'b'}

        for scenario_name, data in scenarios.items():
            if not data:
                continue

            # 1. Rimuovi il transitorio
            steady_data = [(t, v) for t, v in data if t >= warmup]
            if not steady_data:
                continue

            # 2. Definisci l'intervallo temporale dello steady-state
            start_time = warmup
            end_time = steady_data[-1][0]
            total_duration = end_time - start_time

            if total_duration <= 0: continue

            # 3. Dividi la DURATA in batch e calcola le medie
            batch_duration = total_duration / batches
            batch_means = []
            batch_timestamps = []

            for i in range(batches):
                batch_start = start_time + i * batch_duration
                batch_end = batch_start + batch_duration

                # Calcola il punto centrale del batch per l'asse X
                batch_center_time = batch_start + (batch_duration / 2)

                # Trova tutti i valori la cui timestamp cade in questo batch
                values_in_batch = [v for t, v in steady_data if batch_start <= t < batch_end]

                if values_in_batch:
                    batch_means.append(np.mean(values_in_batch))
                    batch_timestamps.append(batch_center_time)

            # 4. Disegna i punti e la linea che li connette
            if batch_timestamps:
                ax.plot(batch_timestamps, batch_means, marker='o', linestyle='-',
                        color=colors[scenario_name], label=scenario_name)

        # Estetica del grafico
        ax.set_title('Evoluzione delle Medie per Batch della Lunghezza della Coda', fontsize=16)
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(title='Scenario')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

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
        self.plot_batch_mean_queue_trend(warmup, batches, transient_output_dir)
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


        # --- SEZIONE 3: Confronti Aggiuntivi ---
        comparison_output_dir = os.path.join(output_dir, "comparison")
        print(f"\n--- 3. Confronti Aggiuntivi (output in '{comparison_output_dir}') ---")
        self.plot_wait_time_comparison_trend(output_dir=comparison_output_dir)
        self.plot_times_by_request_type_grid(output_dir=comparison_output_dir)
        self.plot_steady_state_throughput_ci(analyzer_baseline, analyzer_prio, comparison_output_dir)