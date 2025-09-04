# In src/steady_state_analysis/steady_state_plotter.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.config import RequestType

plt.style.use('ggplot')

class SteadyStatePlotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config, use_log_scale_infinite=False):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.config = config
        self.req_type_colors = {
            RequestType.ADD_TO_CART: '#FF1493', RequestType.ANALYTICS: '#00BFFF',
            RequestType.CHECKOUT: '#32CD32', RequestType.LOGIN: '#FFD700',
            RequestType.NAVIGATION: '#9400D3'
        }
        self.use_log_scale_infinite = use_log_scale_infinite

    # ==============================================================================
    # ORCHESTRATORE PRINCIPALE
    # ==============================================================================

    def generate_steady_state_report(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir="plots/steady_state"):
        print(f"\n--- INIZIO Generazione Report Completo in '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)

        print("\n--- [SEZIONE 1/4] Analisi Performance a Regime ---")
        self.plot_steady_state_times_by_type(analyzer_baseline, analyzer_prio, warmup, batches, os.path.join(output_dir, "times_analysis"))
        self.plot_throughput_analysis(analyzer_baseline, analyzer_prio, warmup, batches, os.path.join(output_dir, "throughput_analysis"))

        baseline_loss_results = analyzer_baseline.calculate_batch_means_ci(self.metrics.get_all_outcomes_as_binary_stream(), warmup, batches)
        prio_loss_results = analyzer_prio.calculate_batch_means_ci(self.metrics_prio.get_all_outcomes_as_binary_stream(), warmup, batches)
        if baseline_loss_results and prio_loss_results:
            self.plot_steady_state_loss_ci(baseline_loss_results, prio_loss_results, os.path.join(output_dir, "loss_analysis"))

        print("\n--- [SEZIONE 2/4] Analisi Comportamento del Sistema ---")
        self.plot_pod_history_analysis(os.path.join(output_dir, "pod_history_analysis"))
        self.plot_queue_history_analysis(os.path.join(output_dir, "queue_history_analysis"), use_log_scale=True)
        self.plot_wait_time_trend_analysis(os.path.join(output_dir, "wait_time_trend_analysis"))

        print("\n--- [SEZIONE 3/4] Analisi del Transitorio e della Stabilità ---")
        self.plot_convergence_analysis_overall(os.path.join(output_dir, "convergence_overall_analysis"), warmup_period=warmup)
        self.plot_convergence_analysis_by_type(os.path.join(output_dir, "convergence_by_type_analysis"))
        self.plot_variance_trend_analysis(os.path.join(output_dir, "variance_trend_analysis"))
        self.plot_batch_mean_queue_trend_analysis(warmup, batches, os.path.join(output_dir, "queue_batch_means_analysis"))

        print("\n--- [SEZIONE 4/4] Analisi Dettagliate per Tipo di Richiesta ---")
        self.plot_times_by_request_type_grid(os.path.join(output_dir, "detailed_grid_analysis"))

        print(f"\n--- FINE Generazione Report. Controlla la cartella '{output_dir}'. ---")

    # ==============================================================================
    # METODI DI PLOTTING COMPLETI E CORRETTI
    # ==============================================================================

    def _save_plot(self, output_dir, filename, fig):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def _plot_single_scenario_times(self, analyzer, metrics, scenario_name, warmup, batches, output_dir, file_suffix, color_palette):
        is_prio = isinstance(metrics, MetricsWithPriority)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        for metric_name, ax in zip(['response', 'wait'], axes):
            plot_data = []
            for req_type in all_req_types:
                raw_data = []
                if is_prio:
                    values = metrics.response_times_by_req_type.get(req_type, []) if metric_name == 'response' else metrics.wait_times_by_req_type.get(req_type, [])
                    timestamps = metrics.completion_timestamps_by_req_type.get(req_type, [])
                    if len(values) == len(timestamps):
                        raw_data = sorted(zip(timestamps, values), key=lambda x:x[0])
                else:
                    raw_data = metrics.response_times_history.get(req_type, []) if metric_name == 'response' else metrics.wait_times_history.get(req_type, [])

                if (ci := analyzer.calculate_batch_means_ci(raw_data, warmup, batches)):
                    plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci['mean'], 'Errore': ci['half_width']})

            if not plot_data: continue
            df = pd.DataFrame(plot_data)

            sns.barplot(data=df, x='Categoria', y='Tempo Medio (s)', order=category_names, color=color_palette, ax=ax)

            x_positions = np.arange(len(category_names))
            subset = df.set_index('Categoria').reindex(category_names)

            if not subset.empty:
                y_coords = subset['Tempo Medio (s)'].fillna(0)
                errors = subset['Errore'].fillna(0)
                ax.errorbar(x_positions, y_coords, yerr=errors, fmt='none', c='black', capsize=5, elinewidth=1.2)

                if ax.containers:
                    ax.bar_label(ax.containers[0], fmt='%.3f', padding=3, fontsize=8, weight='bold', color='black')

                for k, row in subset.iterrows():
                    if pd.notna(row['Tempo Medio (s)']):
                        try:
                            cat_index = category_names.index(row.name)
                            mean_val, error_val = row['Tempo Medio (s)'], row['Errore']
                            upper_bound = mean_val + error_val

                            ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"
                            ax.annotate(ci_text,
                                        xy=(x_positions[cat_index], upper_bound),
                                        xytext=(0, 4), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=8, color='black')
                        except ValueError:
                            continue

            ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio", fontsize=16)
            ax.set_ylabel('Tempo Medio (s)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

            # CORREZIONE: Imposta il limite DOPO aver disegnato tutto.
            # Questo garantisce che tutte le etichette siano incluse.
            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(bottom=current_bottom, top=current_top * 1.03) # Aggiungi 20% di spazio in alto

        fig.suptitle(f'Tempi Medi (Steady State) - {scenario_name}', fontsize=20, fontweight='bold')
        plt.tight_layout()
        fig.subplots_adjust(top=0.90, bottom=0.20, left=0.07, right=0.98)
        self._save_plot(output_dir, f"times{file_suffix}.png", fig)

    def plot_steady_state_times_by_type(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir):
        self._plot_single_scenario_times(analyzer_baseline, self.metrics, "Senza Priorità", warmup, batches, output_dir, "_baseline", '#ff0000')
        self._plot_single_scenario_times(analyzer_prio, self.metrics_prio, "Con Priorità", warmup, batches, output_dir, "_prio", '#0000ff')

        print("Generazione grafico di CONFRONTO per tempi per tipo di richiesta...")
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]


        for metric_name, ax in zip(['response', 'wait'], axes):
            if self.use_log_scale_infinite:
                ax.set_yscale('log')
            plot_data = []
            for req_type in all_req_types:
                if (ci_b := analyzer_baseline.calculate_batch_means_ci(self.metrics.response_times_history.get(req_type, []) if metric_name == 'response' else self.metrics.wait_times_history.get(req_type, []), warmup, batches)):
                    plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_b['mean'], 'Errore': ci_b['half_width'], 'Scenario': 'Senza Priorità'})

                vals_p = self.metrics_prio.response_times_by_req_type.get(req_type, []) if metric_name == 'response' else self.metrics_prio.wait_times_by_req_type.get(req_type, [])
                ts_p = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
                if len(vals_p) == len(ts_p):
                    if (ci_p := analyzer_prio.calculate_batch_means_ci(sorted(zip(ts_p, vals_p), key=lambda x:x[0]), warmup, batches)):
                        plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Tempo Medio (s)': ci_p['mean'], 'Errore': ci_p['half_width'], 'Scenario': 'Con Priorità'})

            if not plot_data: continue
            df = pd.DataFrame(plot_data)
            sns.barplot(data=df, x='Categoria', y='Tempo Medio (s)', hue='Scenario', order=category_names, hue_order=['Senza Priorità', 'Con Priorità'], palette=['#ff0000', '#0000ff'], ax=ax, dodge=True)

            num_categories, width = len(category_names), 0.4
            x_positions = np.arange(num_categories)
            for i, scenario in enumerate(['Senza Priorità', 'Con Priorità']):
                offset = -width / 2 if i == 0 else width / 2
                subset = df[df['Scenario'] == scenario].set_index('Categoria').reindex(category_names)
                if subset['Tempo Medio (s)'].isnull().all(): continue
                y_coords, errors = subset['Tempo Medio (s)'].fillna(0), subset['Errore'].fillna(0)
                ax.errorbar(x_positions + offset, y_coords, yerr=errors, fmt='none', c='black', capsize=5, elinewidth=1.2)
                for k, row in subset.iterrows():
                    if pd.notna(row['Tempo Medio (s)']):
                        try:
                            cat_index = category_names.index(row.name)
                            mean_val, error_val = row['Tempo Medio (s)'], row['Errore']
                            upper_bound = mean_val + error_val

                            ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"
                            ax.annotate(ci_text,
                                        xy=(x_positions[cat_index] + offset, upper_bound),
                                        xytext=(0, 3), textcoords='offset points',
                                        ha='center', va='bottom', fontsize=7, color='black')

                            y_limit_top = ax.get_ylim()[1]
                            if mean_val > y_limit_top * 0.1:
                                ax.text(x_positions[cat_index] + offset, mean_val / 2, f'{mean_val:.3f}', ha='center', va='center', color='white', fontsize=7.5, weight='bold')
                        except ValueError:
                            continue

            ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio", fontsize=16)
            ax.set_ylabel('Tempo Medio (s)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            ax.legend(title='Scenario').remove()

            # CORREZIONE: Imposta il limite DOPO aver disegnato tutto.
            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(bottom=current_bottom, top=current_top * 1.03)

        fig.suptitle('Confronto Tempi Medi (Steady State) per Tipo con IC al 95%', fontsize=20, fontweight='bold')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), title='Scenario', fontsize=12)
        plt.tight_layout()
        fig.subplots_adjust(top=0.88, bottom=0.20)
        self._save_plot(output_dir, "times_comparison.png", fig)
    def _plot_single_scenario_loss(self, results, scenario_name, color, output_dir, filename):
        fig, ax = plt.subplots(figsize=(8, 6))

        mean_val = results['mean']
        half_width = results['half_width']

        bar = ax.bar(scenario_name, mean_val, yerr=half_width, color=color, capsize=10, alpha=0.8, width=0.4)
        ax.set_title(f'Probabilità di Perdita (Steady State) - {scenario_name}', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata'); ax.set_xlabel('Scenario')

        ax.set_ylim(bottom=0, top=(mean_val + half_width) * 1.5)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # CORREZIONE: Aggiungi padding per spostare il valore sopra la barra, non dentro.
        ax.bar_label(bar, fmt='%.4f', padding=5, fontsize=10, weight='bold')

        upper_bound = mean_val + half_width
        ci_text = f"IC 95%: [{max(0, mean_val - half_width):.4f}, {upper_bound:.4f}]"
        ax.annotate(ci_text,
                    xy=(0, upper_bound),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)
    def plot_steady_state_loss_ci(self, baseline_results, prio_results, output_dir):
        self._plot_single_scenario_loss(
            baseline_results, 'Senza Priorità', '#ff0000', output_dir, "loss_probability_baseline.png"
        )
        self._plot_single_scenario_loss(
            prio_results, 'Con Priorità', '#0000ff', output_dir, "loss_probability_prio.png"
        )

        print("Generazione grafico di CONFRONTO per probabilità di perdita...")
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scala logaritmica solo per steady state
        if self.use_log_scale_infinite:
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-6, top=max(baseline_results['mean'], prio_results['mean']) * 5)
        else:
            ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.2)

        bars = ax.bar(
            ['Senza Priorità', 'Con Priorità'],
            [baseline_results['mean'], prio_results['mean']],
            yerr=[baseline_results['half_width'], prio_results['half_width']],
            color=['#ff0000', '#0000ff'], capsize=10, alpha=0.8, width=0.5
        )

        ax.set_title('Confronto Probabilità di Perdita (Steady State) con IC al 95%', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.bar_label(bars, fmt='%.4f', padding=3)

        plt.tight_layout()
        self._save_plot(output_dir, "loss_probability_comparison.png", fig)


    def _plot_convergence_baseline_by_type(self, output_dir):
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type, history in self.metrics.response_times_history.items():
            if history:
                history.sort(key=lambda x: x[0])
                timestamps, values = zip(*history)
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), label=f'{req_type.name}', color=self.req_type_colors.get(req_type), linewidth=2)
        ax.set_title('Analisi Convergenza per Tipo (Baseline)'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, "baseline_convergence_by_type.png", fig)

    def _plot_convergence_prio_by_type(self, output_dir):
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
                sorted_timestamps, sorted_values = zip(*history)
                ax.plot(sorted_timestamps, np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1), label=f'{req_type.name}', color=self.req_type_colors.get(req_type), linewidth=2)
        ax.set_title('Analisi Convergenza per Tipo (Con Priorità)'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, "prio_convergence_by_type.png", fig)

    def plot_convergence_analysis_by_type(self, output_dir):
        self._plot_convergence_baseline_by_type(output_dir)
        self._plot_convergence_prio_by_type(output_dir)
        print("Generazione grafico di CONFRONTO di convergenza per tipo...")
        fig, ax = plt.subplots(figsize=(14, 8))
        for req_type, history in self.metrics.response_times_history.items():
            if history:
                timestamps, values = zip(*sorted(history, key=lambda x: x[0]))
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), label=f'{req_type.name} (Baseline)', color=self.req_type_colors.get(req_type), linestyle='-', linewidth=2)
        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                sorted_timestamps, sorted_values = zip(*sorted(zip(timestamps, response_times), key=lambda x: x[0]))
                ax.plot(sorted_timestamps, np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1), label=f'{req_type.name} (Priorità)', color=self.req_type_colors.get(req_type), linestyle='--', linewidth=2.5)
        ax.set_title('Confronto Convergenza per Tipo di Richiesta'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario e Tipo', bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        self._save_plot(output_dir, "convergence_by_type_comparison.png", fig)

    def _plot_convergence_overall(self, all_responses, scenario_name, output_dir, filename, color, warmup_period):
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_responses:
            timestamps, values = zip(*all_responses)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label='Tempo Risposta Medio Cumulativo')
            ax.axvline(x=warmup_period, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_period}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Analisi Convergenza Tempo Risposta Medio ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_convergence_analysis_overall(self, output_dir, warmup_period=300):
        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        self._plot_convergence_overall(all_responses_base, "Senza Priorità", output_dir, "baseline_convergence_overall.png", 'r', warmup_period)
        self._plot_convergence_overall(all_responses_prio, "Con Priorità", output_dir, "prio_convergence_overall.png", 'b', warmup_period)
        print("Generazione grafico di CONFRONTO di convergenza generale...")
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_responses_base:
            timestamps, values = zip(*all_responses_base)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color='r', label='Senza Priorità')
        if all_responses_prio:
            timestamps, values = zip(*all_responses_prio)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color='b', label='Con Priorità')
        if all_responses_base or all_responses_prio:
            ax.axvline(x=warmup_period, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_period}s)')
        ax.set_title('Confronto Convergenza del Tempo di Risposta Medio'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "convergence_overall_comparison.png", fig)

    def _plot_wait_time_trend(self, all_waits, scenario_name, output_dir, filename, color):
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_waits:
            times, values = zip(*all_waits)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label='Tempo Attesa Medio Cumulativo')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Evoluzione del Tempo di Attesa Medio ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_wait_time_trend_analysis(self, output_dir):
        # Dati per la baseline (corretti)
        all_waits_base = sorted([item for sublist in self.metrics.wait_times_history.values() for item in sublist], key=lambda x:x[0])

        # CORREZIONE: Raccogli manualmente i dati per lo scenario con priorità
        all_waits_prio_list = []
        for req_type in self.metrics_prio.wait_times_by_req_type.keys():
            waits = self.metrics_prio.wait_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_prio_list.extend(zip(timestamps, waits))

        # Ordina la lista combinata per timestamp
        all_waits_prio = sorted(all_waits_prio_list, key=lambda x: x[0])

        # Chiamate ai metodi helper (ora con i dati corretti)
        self._plot_wait_time_trend(all_waits_base, "Senza Priorità", output_dir, "wait_time_trend_baseline.png", 'r')
        self._plot_wait_time_trend(all_waits_prio, "Con Priorità", output_dir, "wait_time_trend_prio.png", 'b')

        print("Generazione grafico di CONFRONTO andamento tempo di attesa...")
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_waits_base:
            times, values = zip(*all_waits_base)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color='r', label='Senza Priorità')
        if all_waits_prio:
            times, values = zip(*all_waits_prio)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color='b', label='Con Priorità')

        ax.set_title('Confronto Evoluzione del Tempo di Attesa Medio')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "wait_time_trend_comparison.png", fig)

    def _plot_pod_history(self, times, counts, scenario_name, output_dir, filename, color):
        fig, ax = plt.subplots(figsize=(14, 7))
        if times and counts:
            ax.plot(times, counts, color=color, label='Numero di Pod', alpha=0.8, linewidth=1.5)
            ax.axvline(x=self.config.WARM_UP_TO_STEADY, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({self.config.WARM_UP_TO_STEADY}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Evoluzione del Numero di Pod ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_pod_history_analysis(self, output_dir):
        times_b, pods_b = zip(*self.metrics.pod_count_history) if self.metrics.pod_count_history else ([], [])
        self._plot_pod_history(times_b, pods_b, "Senza Priorità", output_dir, "pod_history_baseline.png", 'r')
        self._plot_pod_history(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, "Con Priorità", output_dir, "pod_history_prio.png", 'b')
        print("Generazione grafico di CONFRONTO storico dei Pod...")
        fig, ax = plt.subplots(figsize=(14, 7))
        if times_b and pods_b: ax.plot(times_b, pods_b, color='r', label='Senza Priorità', alpha=0.8, linewidth=1.5)
        if self.metrics_prio.timestamps and self.metrics_prio.pod_counts: ax.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', label='Con Priorità', alpha=0.8, linewidth=1.5)
        ax.axvline(x=self.config.WARM_UP_TO_STEADY, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({self.config.WARM_UP_TO_STEADY}s)')
        ax.set_title('Confronto Evoluzione del Numero di Pod'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1); ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, "pod_history_comparison.png", fig)

    def _plot_queue_history(self, times, queue_lengths, scenario_name, output_dir, filename, color, use_log_scale):
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'

        # Dizionario per mappare il colore base al colore scuro
        dark_color_map = {'r': 'darkred', 'b': 'darkblue'}

        if times and queue_lengths:
            ax.plot(times, queue_lengths, color=color, label='Lunghezza Coda', alpha=0.7, linewidth=1.5)
            steady_queue = [q for t, q in zip(times, queue_lengths) if t >= self.config.WARM_UP_TO_STEADY]

            # CORREZIONE: Usa il dizionario per ottenere il colore corretto
            dark_color = dark_color_map.get(color, color) # Se il colore non è in mappa, usa il colore originale
            if steady_queue:
                ax.axhline(np.mean(steady_queue), color=dark_color, linestyle='--', label=f'Media Steady-State: {np.mean(steady_queue):.2f}')

            ax.axvline(x=self.config.WARM_UP_TO_STEADY, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({self.config.WARM_UP_TO_STEADY}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)

        if use_log_scale:
            ax.set_yscale('log'); ylabel += ' (Scala Log)'; ax.set_ylim(bottom=0.1)

        ax.set_title(f'Evoluzione Lunghezza della Coda ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_queue_history_analysis(self, output_dir, use_log_scale=True):
        times_b, queue_b = zip(*self.metrics.queue_length_history) if self.metrics.queue_length_history else ([],[])
        self._plot_queue_history(times_b, queue_b, "Senza Priorità", output_dir, f"queue_history_baseline{'_log' if use_log_scale else ''}.png", 'r', use_log_scale)
        self._plot_queue_history(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, "Con Priorità", output_dir, f"queue_history_prio{'_log' if use_log_scale else ''}.png", 'b', use_log_scale)
        print("Generazione grafico di CONFRONTO storico della Coda...")
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'
        if times_b and queue_b:
            ax.plot(times_b, queue_b, color='r', label='Senza Priorità', alpha=0.7, linewidth=1.5)
            if (steady_queue_b := [q for t, q in zip(times_b, queue_b) if t >= self.config.WARM_UP_TO_STEADY]):
                ax.axhline(np.mean(steady_queue_b), color='darkred', linestyle=':', label=f'Media Steady (Baseline): {np.mean(steady_queue_b):.2f}')
        if self.metrics_prio.timestamps and self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', label='Con Priorità', alpha=0.7, linewidth=1.5)
            if (steady_queue_p := [q for t, q in zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths) if t >= self.config.WARM_UP_TO_STEADY]):
                ax.axhline(np.mean(steady_queue_p), color='darkblue', linestyle=':', label=f'Media Steady (Priorità): {np.mean(steady_queue_p):.2f}')
        if use_log_scale:
            ax.set_yscale('log'); ylabel += ' (Scala Log)'; ax.set_ylim(bottom=0.1)
        ax.set_title('Confronto Evoluzione della Lunghezza della Coda'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, f"queue_history_comparison{'_log' if use_log_scale else ''}.png", fig)

    def _plot_variance_trend(self, all_responses, scenario_name, output_dir, filename, color, window_size):
        fig, ax = plt.subplots(figsize=(14, 7))
        if len(all_responses) > window_size:
            times, values = zip(*all_responses)
            moving_std = pd.Series(values).rolling(window=window_size).std()
            ax.plot(times[window_size-1:], moving_std.iloc[window_size-1:], color=color, label='Dev. Std. Mobile', alpha=0.8)
            ax.axvline(x=self.config.WARM_UP_TO_STEADY, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({self.config.WARM_UP_TO_STEADY}s)')
        else:
            ax.text(0.5, 0.5, f"Dati insufficienti (necessari > {window_size})", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Stabilizzazione Varianza ({scenario_name}) - Finestra di {window_size}'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0); ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_variance_trend_analysis(self, output_dir, window_size=500):
        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        self._plot_variance_trend(all_responses_base, "Senza Priorità", output_dir, "variance_trend_baseline.png", 'r', window_size)
        self._plot_variance_trend(all_responses_prio, "Con Priorità", output_dir, "variance_trend_prio.png", 'b', window_size)
        print("Generazione grafico di CONFRONTO andamento della varianza...")
        fig, ax = plt.subplots(figsize=(14, 7))
        if len(all_responses_base) > window_size:
            times, values = zip(*all_responses_base)
            ax.plot(times[window_size-1:], pd.Series(values).rolling(window=window_size).std().iloc[window_size-1:], color='r', label='Senza Priorità', alpha=0.8)
        if len(all_responses_prio) > window_size:
            times, values = zip(*all_responses_prio)
            ax.plot(times[window_size-1:], pd.Series(values).rolling(window=window_size).std().iloc[window_size-1:], color='b', label='Con Priorità', alpha=0.8)
        ax.axvline(x=self.config.WARM_UP_TO_STEADY, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({self.config.WARM_UP_TO_STEADY}s)')
        ax.set_title(f'Confronto Stabilizzazione Varianza (Finestra di {window_size})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0); ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "variance_trend_comparison.png", fig)

    def _plot_batch_mean_queue_single(self, data, scenario_name, warmup, batches, output_dir, color):
        fig, ax = plt.subplots(figsize=(14, 7))
        if not data or not (steady_data := [(t, v) for t, v in data if t >= warmup]):
            ax.text(0.5, 0.5, "Nessun dato disponibile/in steady-state", ha='center', va='center', transform=ax.transAxes)
        else:
            total_duration = steady_data[-1][0] - warmup
            if total_duration > 0:
                batch_duration = total_duration / batches
                batch_means, batch_timestamps = [], []
                for i in range(batches):
                    batch_start, batch_end = warmup + i * batch_duration, warmup + (i + 1) * batch_duration
                    values_in_batch = [v for t, v in steady_data if batch_start <= t < batch_end]
                    if values_in_batch:
                        batch_means.append(np.mean(values_in_batch))
                        batch_timestamps.append(batch_start + (batch_duration / 2))
                if batch_timestamps: ax.plot(batch_timestamps, batch_means, marker='o', linestyle='-', color=color, label="Media per Batch")
        ax.set_title(f'Evoluzione Medie per Batch della Coda ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save_plot(output_dir, f"queue_batch_means_trend_{scenario_name.lower().replace(' ', '_')}.png", fig)

    def plot_batch_mean_queue_trend_analysis(self, warmup, batches, output_dir):
        self._plot_batch_mean_queue_single(self.metrics.queue_length_history, "Senza Priorità", warmup, batches, output_dir, 'r')
        data_prio = list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths)) if self.metrics_prio.queue_lengths else []
        self._plot_batch_mean_queue_single(data_prio, "Con Priorità", warmup, batches, output_dir, 'b')
        print("Generazione grafico CONFRONTO trend delle medie dei batch della coda...")
        fig, ax = plt.subplots(figsize=(14, 7))
        scenarios = {"Senza Priorità": self.metrics.queue_length_history, "Con Priorità": data_prio}
        for scenario_name, data in scenarios.items():
            if data and (steady_data := [(t, v) for t, v in data if t >= warmup]):
                total_duration = steady_data[-1][0] - warmup
                if total_duration > 0:
                    batch_duration = total_duration / batches
                    batch_means, batch_timestamps = [], []
                    for i in range(batches):
                        batch_start, batch_end = warmup + i * batch_duration, warmup + (i + 1) * batch_duration
                        values_in_batch = [v for t, v in steady_data if batch_start <= t < batch_end]
                        if values_in_batch:
                            batch_means.append(np.mean(values_in_batch))
                            batch_timestamps.append(batch_start + (batch_duration / 2))
                    if batch_timestamps: ax.plot(batch_timestamps, batch_means, marker='o', linestyle='-', color=('r' if 'Senza' in scenario_name else 'b'), label=scenario_name)
        ax.set_title('Confronto Evoluzione delle Medie per Batch della Coda'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(title='Scenario'); ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save_plot(output_dir, "queue_batch_means_trend_comparison.png", fig)

    def _plot_throughput_single_scenario(self, analyzer, metrics, scenario_name, warmup, batches, output_dir, color):
        is_prio = isinstance(metrics, MetricsWithPriority)
        fig, ax = plt.subplots(figsize=(16, 9))
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]
        plot_data = []
        for req_type in all_req_types:
            timestamps = sorted(metrics.completion_timestamps_by_req_type.get(req_type, [])) if is_prio else sorted([ts for ts, rt in metrics.response_times_history.get(req_type, [])])
            if (results := analyzer.calculate_throughput_ci(timestamps, warmup, batches)):
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Conteggio': results['total_count']})
        if not plot_data:
            print(f"Dati insufficienti per il grafico del throughput ({scenario_name})."); plt.close(fig); return
        df = pd.DataFrame(plot_data)
        sns.barplot(data=df, x='Categoria', y='Conteggio', order=category_names, color=color, ax=ax)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
        ax.set_title(f"Richieste Servite con Successo per Tipo ({scenario_name})", fontsize=18)
        ax.set_xlabel("Tipo di Richiesta", fontsize=14); ax.set_ylabel("Numero di Richieste Servite", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
        plt.tight_layout()
        self._save_plot(output_dir, f"throughput_{scenario_name.lower().replace(' ', '_')}.png", fig)

    def plot_throughput_analysis(self, analyzer_baseline, analyzer_prio, warmup, batches, output_dir):
        self._plot_throughput_single_scenario(analyzer_baseline, self.metrics, "Senza Priorità", warmup, batches, output_dir, '#ff0000')
        self._plot_throughput_single_scenario(analyzer_prio, self.metrics_prio, "Con Priorità", warmup, batches, output_dir, '#0000ff')
        print("Generazione grafico di CONFRONTO delle richieste soddisfatte (throughput)...")
        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle("Confronto Richieste Servite per Tipo - Steady State", fontsize=24, fontweight='bold')
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]
        plot_data = []
        for req_type in all_req_types:
            timestamps_b = sorted([ts for ts, rt in self.metrics.response_times_history.get(req_type, [])])
            if (results_b := analyzer_baseline.calculate_throughput_ci(timestamps_b, warmup, batches)):
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Conteggio': results_b['total_count'], 'Scenario': 'Senza Priorità'})
            timestamps_p = sorted(self.metrics_prio.completion_timestamps_by_req_type.get(req_type, []))
            if (results_p := analyzer_prio.calculate_throughput_ci(timestamps_p, warmup, batches)):
                plot_data.append({'Categoria': req_type.name.replace('_', ' ').title(), 'Conteggio': results_p['total_count'], 'Scenario': 'Con Priorità'})
        if not plot_data:
            print("Dati insufficienti per il grafico di confronto del throughput."); plt.close(fig); return
        df = pd.DataFrame(plot_data)
        sns.barplot(data=df, x='Categoria', y='Conteggio', hue='Scenario', order=category_names, hue_order=['Senza Priorità', 'Con Priorità'], palette=['#ff0000', '#0000ff'], ax=ax)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
        y_top = ax.get_ylim()[1]
        for i, cat_name in enumerate(category_names):
            base_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Senza Priorità')]
            prio_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Con Priorità')]
            if not base_row.empty and not prio_row.empty:
                base_count, prio_count = base_row.iloc[0]['Conteggio'], prio_row.iloc[0]['Conteggio']
                if base_count > 0:
                    delta_perc = ((prio_count - base_count) / base_count) * 100
                    sign, color = ('+', 'green') if delta_perc >= 0 else ('', 'red')
                    ax.text(i, y_top * 0.95, f'Δ: {sign}{delta_perc:.1f}%', ha='center', va='center', fontsize=14, fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.9))
        ax.set_title("Richieste Servite con Successo per Tipo", fontsize=18)
        ax.set_xlabel("Tipo di Richiesta", fontsize=14); ax.set_ylabel("Numero di Richieste Servite", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title='Scenario', loc='upper left'); ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        self._save_plot(output_dir, "throughput_comparison.png", fig)

    def plot_times_by_request_type_grid(self, output_dir):
        print("Generazione griglia di confronto per tipo di richiesta...")
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        ncols, i = 3, -1
        nrows = int(np.ceil(len(all_req_types) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, req_type in enumerate(all_req_types):
            ax = axes[i]
            if (resp_b := sorted(self.metrics.response_times_history.get(req_type, []), key=lambda x: x[0])):
                times, values = zip(*resp_b)
                ax.plot(times, np.cumsum(values) / np.arange(1, len(values)+1), color='salmon', linestyle='--', label='Risposta (Baseline)')
            if (wait_b := sorted(self.metrics.wait_times_history.get(req_type, []), key=lambda x: x[0])):
                times, values = zip(*wait_b)
                ax.plot(times, np.cumsum(values) / np.arange(1, len(values)+1), color='red', label='Attesa (Baseline)')
            times_p = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if (values_rp := self.metrics_prio.response_times_by_req_type.get(req_type, [])) and len(times_p) == len(values_rp):
                times_s, values_s = zip(*sorted(zip(times_p, values_rp), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='lightblue', linestyle='--', label='Risposta (Priorità)')
            if (values_wp := self.metrics_prio.wait_times_by_req_type.get(req_type, [])) and len(times_p) == len(values_wp):
                times_s, values_s = zip(*sorted(zip(times_p, values_wp), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='blue', label='Attesa (Priorità)')
            ax.set_title(req_type.name.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
        if i != -1:
            for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.supxlabel('Tempo di Simulazione (s)', y=0.02)
        fig.supylabel('Tempo Medio Cumulativo (s)', x=0.02)
        plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))
        self._save_plot(output_dir, "times_grid_comparison.png", fig)