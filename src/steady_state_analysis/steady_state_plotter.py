# In src/analysis/plotter.py



import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from matplotlib.ticker import MaxNLocator # per forzare assi Y interi
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer

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

        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
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
            return

        df = pd.DataFrame(plot_data)
        hue_order = ['Senza Priorità', 'Con Priorità']
        palette = ['#ff0000', '#0000ff']

        # Disegna il barplot principale
        sns.barplot(data=df, x='Categoria', y='Probabilità di Perdita', hue='Scenario',
                    order=category_names, hue_order=hue_order, palette=palette, ax=ax)

        # Calcola le posizioni x per le barre e aggiungi le barre di errore
        num_categories = len(category_names)
        x_positions = np.arange(num_categories)
        width = 0.4 # Larghezza di ogni singola barra

        for i, scenario in enumerate(hue_order):
            offset = -width / 2 if i == 0 else width / 2
            subset = df[df['Scenario'] == scenario].set_index('Categoria').loc[category_names]
            ax.errorbar(x_positions + offset, subset['Probabilità di Perdita'], yerr=subset['Errore'],
                        fmt='none', c='black', capsize=4, elinewidth=1)

        # Estetica
        ax.set_xlabel('Tipo di Richiesta')
        ax.set_ylabel('Probabilità di Perdita Stimata')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.legend(title='Scenario')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(bottom=0)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "steady_state_loss_by_type_ci.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def generate_steady_state_report(self, baseline_loss_results, prio_loss_results, analyzer_baseline, analyzer_prio, warmup, batches, output_dir="plots/steady_state"):
        """Metodo principale per generare tutti i grafici di analisi steady-state."""
        print("\n--- Generazione Report Steady-State ---")
        if baseline_loss_results and prio_loss_results:
            self.plot_steady_state_loss_ci(baseline_loss_results, prio_loss_results, output_dir, "loss_probability_ci.png")

        self.plot_steady_state_times_by_type(analyzer_baseline, analyzer_prio, warmup, batches, output_dir)
