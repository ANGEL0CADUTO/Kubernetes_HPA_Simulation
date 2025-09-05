# In src/steady_state_analysis/steady_state_plotter.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sns import sns

from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.config import RequestType

plt.style.use('ggplot')

class SteadyStatePlotter:
    """
    Classe per generare report grafici delle analisi di stato stazionario.
    Ora riceve il periodo di warm-up e i risultati dei batch means complessivi
    dal main.py, ma continua a calcolare i batch means PER TIPO di richiesta
    tramite gli oggetti SteadyStateAnalyzer passati.

    Assunzione chiave: Gli oggetti metrics (self.metrics, self.metrics_prio)
    passati a questa classe *contengono già solo i dati dello stato stazionario*,
    avendo rimosso il periodo di warm-up distruttivamente nel main.py.
    """
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, config, use_log_scale_infinite=True):
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

    def generate_steady_state_report( self,analyzer_baseline,analyzer_prio,warmup_duration_s, overall_batch_ci_results,output_dir="plots/steady_state"):
        """
        Genera un report completo dello stato stazionario.

        Parameters:
        -----------
        analyzer_baseline : Analyzer
            Analizzatore per lo scenario senza priorità.
        analyzer_prio : Analyzer
            Analizzatore per lo scenario con priorità.
        warmup_duration_s : float
            Tempo (s) in cui lo steady-state è iniziato.
        overall_batch_ci_results : dict
            Risultati complessivi dei batch means:
            {"baseline": (mean, ci, b, k_opt), "priority": (mean, ci, b, k_opt)}
        output_dir : str
            Cartella in cui salvare i grafici e report.
        """
        print(f"\n--- INIZIO Generazione Report Completo in '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)

        # --- Sezione 1/4: Analisi Performance a Regime ---
        print("\n[SEZIONE 1/4] Analisi Performance a Regime")
        times_output_dir = os.path.join(output_dir, "times_analysis")
        throughput_output_dir = os.path.join(output_dir, "throughput_analysis")
        os.makedirs(times_output_dir, exist_ok=True)
        os.makedirs(throughput_output_dir, exist_ok=True)

        self.plot_steady_state_times_by_type(
            analyzer_baseline,
            analyzer_prio,
            warmup_duration_s,
            times_output_dir,
            overall_batch_ci_results
        )
        self.plot_throughput_analysis(
            analyzer_baseline,
            analyzer_prio,
            warmup_duration_s,
            throughput_output_dir
        )

        # Analisi perdite
        baseline_loss = analyzer_baseline.compute_loss_ci()
        prio_loss = analyzer_prio.compute_loss_ci()
        if baseline_loss and prio_loss:
            self.plot_steady_state_loss_ci(
                baseline_loss,
                prio_loss,
                os.path.join(output_dir, "loss_analysis")
            )

        # --- Sezione 2/4: Comportamento del Sistema ---
        print("\n[SEZIONE 2/4] Analisi Comportamento del Sistema")
        self.plot_pod_history_analysis(os.path.join(output_dir, "pod_history_analysis"), warmup_duration_s)
        self.plot_queue_history_analysis(os.path.join(output_dir, "queue_history_analysis"),
                                         use_log_scale=True,
                                         warmup_duration_s=warmup_duration_s)
        self.plot_wait_time_trend_analysis(os.path.join(output_dir, "wait_time_trend_analysis"), warmup_duration_s)

        # --- Sezione 3/4: Transitorio e Stabilità ---
        print("\n[SEZIONE 3/4] Analisi del Transitorio e della Stabilità")
        self.plot_convergence_analysis_overall(os.path.join(output_dir, "convergence_overall_analysis"),
                                               warmup_period_s=warmup_duration_s)
        self.plot_convergence_analysis_by_type(os.path.join(output_dir, "convergence_by_type_analysis"))
        self.plot_variance_trend_analysis(os.path.join(output_dir, "variance_trend_analysis"),
                                          warmup_duration_s=warmup_duration_s)

        # Determina k ottimale per batch queue
        k_for_queue_plot = self.config.BATCH_K
        for key in ["baseline", "priority"]:
            if key in overall_batch_ci_results and len(overall_batch_ci_results[key]) > 3:
                k_for_queue_plot = overall_batch_ci_results[key][3] or k_for_queue_plot
                break

        self.plot_batch_mean_queue_trend_analysis(
            warmup_duration_s,
            k_for_queue_plot,
            os.path.join(output_dir, "queue_batch_means_analysis")
        )

        # --- Sezione 4/4: Analisi Dettagliata per Tipo di Richiesta ---
        print("\n[SEZIONE 4/4] Analisi Dettagliate per Tipo di Richiesta")
        self.plot_times_by_request_type_grid(os.path.join(output_dir, "detailed_grid_analysis"))

        print(f"\n--- FINE Generazione Report. Controlla la cartella '{output_dir}' ---")


    # ==============================================================================
    # METODI DI PLOTTING COMPLETI E CORRETTI
    # ==============================================================================

    def _save_plot(self, output_dir, filename, fig):
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def _plot_single_scenario_times(self, analyzer, metrics, scenario_name,warmup_duration_s, output_dir, file_suffix,color_palette, batch_ci_result):
        """
        Disegna tempi di risposta/attesa medi per scenario,
        usando direttamente i risultati batch pre-calcolati.
        batch_ci_result: tupla (mean, ci, b, k)
        """
        is_prio = isinstance(metrics, MetricsWithPriority)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)

        all_req_types = sorted(
            list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name
        )
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        mean_val, ci95, b, k = batch_ci_result

        # Calcola half_width e uniforma mean_val come lista scalare per categoria
        if isinstance(mean_val, (list, np.ndarray)):
            mean_vals = [float(v) for v in mean_val]
        else:
            mean_vals = [float(mean_val)] * len(all_req_types)

        # half_width calcolato dai CI
        if isinstance(ci95[0], (list, np.ndarray)):
            half_widths = [(float(hi) - float(lo)) / 2 for lo, hi in zip(ci95[0], ci95[1])]
        else:
            half_widths = [(float(ci95[1]) - float(ci95[0])) / 2] * len(all_req_types)

        for metric_name, ax in zip(['response', 'wait'], axes):
            # Costruisci il DataFrame per Seaborn
            plot_data = [{
                'Categoria': req_type.name.replace('_', ' ').title(),
                'Tempo Medio (s)': mean_vals[i],
                'Errore': half_widths[i]
            } for i, req_type in enumerate(all_req_types)]

            df = pd.DataFrame(plot_data)
            df['Categoria'] = df['Categoria'].astype('category')  # opzionale ma sicuro

            sns.barplot(
                data=df, x='Categoria', y='Tempo Medio (s)',
                order=category_names, color=color_palette, ax=ax
            )

            x_positions = np.arange(len(category_names))
            subset = df.set_index('Categoria').reindex(category_names)

            if not subset.empty:
                y_coords = subset['Tempo Medio (s)'].fillna(0)
                errors = subset['Errore'].fillna(0)
                ax.errorbar(
                    x_positions, y_coords, yerr=errors, fmt='none',
                    c='black', capsize=5, elinewidth=1.2
                )

                if ax.containers:
                    ax.bar_label(
                        ax.containers[0], fmt='%.3f',
                        padding=3, fontsize=8, weight='bold', color='black'
                    )

                for idx, row in subset.iterrows():
                    if pd.notna(row['Tempo Medio (s)']):
                        cat_index = category_names.index(row.name)
                        mean_v, err_v = row['Tempo Medio (s)'], row['Errore']
                        upper_bound = mean_v + err_v
                        ci_text = f"[{max(0, mean_v - err_v):.3f}, {upper_bound:.3f}]"
                        ax.annotate(
                            ci_text,
                            xy=(x_positions[cat_index], upper_bound),
                            xytext=(0, 4), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8, color='black'
                        )

            ax.set_title(
                f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio",
                fontsize=16
            )
            ax.set_ylabel('Tempo Medio (s)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(bottom=current_bottom, top=current_top * 1.03)

        fig.suptitle(
            f'Tempi Medi (Steady State) - {scenario_name}',
            fontsize=20, fontweight='bold'
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.90, bottom=0.20, left=0.07, right=0.98)
        self._save_plot(output_dir, f"times{file_suffix}.png", fig)



    def plot_steady_state_times_by_type(self, analyzer_baseline, analyzer_prio,
                                        warmup_duration_s, output_dir, overall_batch_ci_results):
        """
        Confronta tempi medi di risposta/attesa tra baseline e priority,
        usando direttamente i risultati batch già calcolati.
        """
        # Plot per singolo scenario
        self._plot_single_scenario_times(
            analyzer_baseline, self.metrics, "Senza Priorità",
            warmup_duration_s, output_dir, "_baseline", '#ff0000',
            overall_batch_ci_results.get("baseline")
        )
        self._plot_single_scenario_times(
            analyzer_prio, self.metrics_prio, "Con Priorità",
            warmup_duration_s, output_dir, "_prio", '#0000ff',
            overall_batch_ci_results.get("priority")
        )

        # Plot di confronto
        print("Generazione grafico di CONFRONTO per tempi per tipo di richiesta...")

        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        # Estrai batch results
        baseline_mean, baseline_ci, *_ = overall_batch_ci_results.get("baseline", (None, None, None, None))
        prio_mean, prio_ci, *_ = overall_batch_ci_results.get("priority", (None, None, None, None))

        if baseline_mean is None or prio_mean is None:
            print(" Dati non disponibili per baseline o priority, skip confronto.")
            return

        # Uniforma valori scalari/lista
        if isinstance(baseline_mean, (list, np.ndarray)):
            baseline_means = [float(v) for v in baseline_mean]
        else:
            baseline_means = [float(baseline_mean)] * len(all_req_types)

        if isinstance(prio_mean, (list, np.ndarray)):
            prio_means = [float(v) for v in prio_mean]
        else:
            prio_means = [float(prio_mean)] * len(all_req_types)

        if isinstance(baseline_ci[0], (list, np.ndarray)):
            baseline_half = [(float(hi)-float(lo))/2 for lo, hi in zip(baseline_ci[0], baseline_ci[1])]
        else:
            baseline_half = [(float(baseline_ci[1])-float(baseline_ci[0]))/2] * len(all_req_types)

        if isinstance(prio_ci[0], (list, np.ndarray)):
            prio_half = [(float(hi)-float(lo))/2 for lo, hi in zip(prio_ci[0], prio_ci[1])]
        else:
            prio_half = [(float(prio_ci[1])-float(prio_ci[0]))/2] * len(all_req_types)

        # Costruisci DataFrame combinato
        plot_data = []
        for i, req_type in enumerate(all_req_types):
            cat = req_type.name.replace('_', ' ').title()
            plot_data.append({
                'Categoria': cat,
                'Tempo Medio (s)': baseline_means[i],
                'Errore': baseline_half[i],
                'Scenario': 'Senza Priorità'
            })
            plot_data.append({
                'Categoria': cat,
                'Tempo Medio (s)': prio_means[i],
                'Errore': prio_half[i],
                'Scenario': 'Con Priorità'
            })

        df = pd.DataFrame(plot_data)
        df['Categoria'] = df['Categoria'].astype('category')

        # Creazione grafico
        for metric_name, ax in zip(['response', 'wait'], axes):
            if self.use_log_scale_infinite:
                ax.set_yscale('log')

            sns.barplot(
                data=df, x='Categoria', y='Tempo Medio (s)',
                hue='Scenario',
                order=category_names,
                hue_order=['Senza Priorità', 'Con Priorità'],
                palette=['#ff0000', '#0000ff'],
                ax=ax, dodge=True
            )

            num_categories, width = len(category_names), 0.4
            x_positions = np.arange(num_categories)

            for i, scenario in enumerate(['Senza Priorità', 'Con Priorità']):
                offset = -width / 2 if i == 0 else width / 2
                subset = df[df['Scenario'] == scenario].set_index('Categoria').reindex(category_names)

                y_coords = subset['Tempo Medio (s)'].fillna(0)
                errors = subset['Errore'].fillna(0)
                ax.errorbar(
                    x_positions + offset, y_coords, yerr=errors,
                    fmt='none', c='black', capsize=5, elinewidth=1.2
                )

                for idx, row in subset.iterrows():
                    if pd.notna(row['Tempo Medio (s)']):
                        cat_index = category_names.index(row.name)
                        mean_val, error_val = row['Tempo Medio (s)'], row['Errore']
                        upper_bound = mean_val + error_val
                        ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"
                        ax.annotate(
                            ci_text,
                            xy=(x_positions[cat_index] + offset, upper_bound),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7, color='black'
                        )

                        # Testo interno alle barre
                        y_limit_top = ax.get_ylim()[1]
                        if mean_val > y_limit_top * 0.1:
                            ax.text(
                                x_positions[cat_index] + offset, mean_val / 2,
                                f'{mean_val:.3f}', ha='center', va='center',
                                color='white', fontsize=7.5, weight='bold'
                            )

            ax.set_title(
                f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio",
                fontsize=16
            )
            ax.set_ylabel('Tempo Medio (s)', fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            ax.legend(title='Scenario').remove()
            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(bottom=current_bottom, top=current_top * 1.03)

        fig.suptitle(
            'Confronto Tempi Medi (Steady State) per Tipo con IC al 95%',
            fontsize=20, fontweight='bold'
        )
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc='upper right',
            bbox_to_anchor=(0.98, 0.95),
            title='Scenario', fontsize=12
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.88, bottom=0.20)
        self._save_plot(output_dir, "times_comparison.png", fig)



    # =========================================
    # Probabilità di perdita (Loss Probability)
    # =========================================
    def _plot_single_scenario_loss(self, results, scenario_name, color, output_dir, filename):
        fig, ax = plt.subplots(figsize=(8, 6))

        mean_val = results['mean']
        half_width = results['half_width']

        bar = ax.bar(scenario_name, mean_val, yerr=half_width, color=color,
                     capsize=10, alpha=0.8, width=0.4)
        ax.set_title(f'Probabilità di Perdita (Steady State) - {scenario_name}', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata'); ax.set_xlabel('Scenario')

        ax.set_ylim(bottom=0, top=(mean_val + half_width) * 1.5)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        ax.bar_label(bar, fmt='%.4f', padding=5, fontsize=10, weight='bold')

        upper_bound = mean_val + half_width
        ci_text = f"IC 95%: [{max(0, mean_val - half_width):.4f}, {upper_bound:.4f}]"
        ax.annotate(ci_text, xy=(0, upper_bound), xytext=(0, 5),
                    textcoords='offset points', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_steady_state_loss_ci(self, baseline_results, prio_results, output_dir):
        self._plot_single_scenario_loss(
            baseline_results, 'Senza Priorità', '#ff0000', output_dir, "loss_probability_baseline.png"
        )
        self._plot_single_scenario_loss(
            prio_results, 'Con Priorità', '#0000ff', output_dir, "loss_probability_prio.png"
        )

        # Confronto
        fig, ax = plt.subplots(figsize=(8, 6))
        means = [baseline_results['mean'], prio_results['mean']]
        errors = [baseline_results['half_width'], prio_results['half_width']]
        labels = ['Senza Priorità', 'Con Priorità']
        colors = ['#ff0000', '#0000ff']

        bars = ax.bar(labels, means, yerr=errors, color=colors, capsize=10, alpha=0.8, width=0.5)
        ax.set_title('Confronto Probabilità di Perdita (Steady State) con IC al 95%', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.bar_label(bars, fmt='%.4f', padding=3)
        plt.tight_layout()
        self._save_plot(output_dir, "loss_probability_comparison.png", fig)

    # =========================================
    # Convergenza per tipo di richiesta
    # =========================================
    def _plot_convergence_by_type(self, metrics, req_types_sorted, scenario_name, color_map, output_dir, filename, linestyle='-'):
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type in req_types_sorted:
            history = metrics.response_times_history.get(req_type, [])
            if history:
                history = sorted(history, key=lambda x: x[0])
                timestamps, values = zip(*history)
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1),
                        label=req_type.name, color=color_map.get(req_type), linestyle=linestyle, linewidth=2)
        ax.set_title(f'Analisi Convergenza per Tipo ({scenario_name}) - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_convergence_analysis_by_type(self, metrics_base, metrics_prio, color_map, output_dir):
        # Baseline
        req_types_base = sorted(metrics_base.response_times_history.keys(), key=lambda x: x.name)
        self._plot_convergence_by_type(metrics_base, req_types_base, "Senza Priorità", color_map, output_dir, "baseline_convergence_by_type.png")
        # Priorità
        req_types_prio = sorted(metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name)
        self._plot_convergence_by_type(metrics_prio, req_types_prio, "Con Priorità", color_map, output_dir, "prio_convergence_by_type.png", linestyle='--')

        # Confronto
        fig, ax = plt.subplots(figsize=(14, 8))
        for req_type in req_types_base:
            history = metrics_base.response_times_history.get(req_type, [])
            if history:
                timestamps, values = zip(*sorted(history, key=lambda x: x[0]))
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1),
                        label=f'{req_type.name} (Baseline)', color=color_map.get(req_type), linestyle='-', linewidth=2)
        for req_type in req_types_prio:
            history = metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if history and len(history) == len(timestamps):
                timestamps, values = zip(*sorted(zip(timestamps, history), key=lambda x: x[0]))
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1),
                        label=f'{req_type.name} (Priorità)', color=color_map.get(req_type), linestyle='--', linewidth=2.5)
        ax.set_title('Confronto Convergenza per Tipo di Richiesta - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario e Tipo', bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        self._save_plot(output_dir, "convergence_by_type_comparison.png", fig)

    # =========================================
    # Convergenza globale e trend varianza
    # =========================================
    def _plot_cumulative(self, all_responses, scenario_name, color, output_dir, filename, warmup_period_s, ylabel, label):
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_responses:
            timestamps, values = zip(*all_responses)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label=label)
            ax.axvline(x=warmup_period_s, color='k', linestyle=':', linewidth=2, label=f'Inizio Steady-state ({warmup_period_s:.0f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{ylabel} ({scenario_name}) - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_convergence_analysis_overall(self, metrics_base, metrics_prio, output_dir, warmup_period_s):
        all_responses_base = metrics_base.get_all_response_times_with_timestamps()
        all_responses_prio = metrics_prio.get_all_response_times_with_timestamps()
        self._plot_cumulative(all_responses_base, "Senza Priorità", 'r', output_dir, "baseline_convergence_overall.png", warmup_period_s, 'Tempo di Risposta Medio', 'Senza Priorità')
        self._plot_cumulative(all_responses_prio, "Con Priorità", 'b', output_dir, "prio_convergence_overall.png", warmup_period_s, 'Tempo di Risposta Medio', 'Con Priorità')

        # Confronto
        fig, ax = plt.subplots(figsize=(12, 7))
        for all_responses, color, label in [(all_responses_base, 'r', 'Senza Priorità'), (all_responses_prio, 'b', 'Con Priorità')]:
            if all_responses:
                timestamps, values = zip(*all_responses)
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label=label)
        ax.axvline(x=warmup_period_s, color='k', linestyle=':', linewidth=2, label=f'Inizio Steady-state ({warmup_period_s:.0f}s)')
        ax.set_title('Confronto Convergenza del Tempo di Risposta Medio - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "convergence_overall_comparison.png", fig)

    # =========================================
    # Trend tempi di attesa
    # =========================================
    def _plot_wait_time_trend(self, all_waits, scenario_name, output_dir, filename, color, warmup_duration_s):
        self._plot_cumulative(all_waits, scenario_name, color, output_dir, filename, warmup_duration_s, 'Tempo di Attesa Medio Cumulativo', 'Tempo Attesa Medio Cumulativo')

    def plot_wait_time_trend_analysis(self, metrics_base, metrics_prio, output_dir, warmup_duration_s):
        all_waits_base = sorted([item for sublist in metrics_base.wait_times_history.values() for item in sublist], key=lambda x:x[0])
        all_waits_prio = []
        for req_type in metrics_prio.wait_times_by_req_type.keys():
            waits = metrics_prio.wait_times_by_req_type.get(req_type, [])
            timestamps = metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_prio.extend(zip(timestamps, waits))
        all_waits_prio = sorted(all_waits_prio, key=lambda x: x[0])

        self._plot_wait_time_trend(all_waits_base, "Senza Priorità", output_dir, "wait_time_trend_baseline.png", 'r', warmup_duration_s)
        self._plot_wait_time_trend(all_waits_prio, "Con Priorità", output_dir, "wait_time_trend_prio.png", 'b', warmup_duration_s)

        # Confronto
        fig, ax = plt.subplots(figsize=(12, 7))
        for all_waits, color, label in [(all_waits_base, 'r', 'Senza Priorità'), (all_waits_prio, 'b', 'Con Priorità')]:
            if all_waits:
                timestamps, values = zip(*all_waits)
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label=label)
        ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2, label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        ax.set_title('Confronto Evoluzione del Tempo di Attesa Medio - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "wait_time_trend_comparison.png", fig)



    # --- Pod History ---
    def _plot_pod_history(self, times, counts, scenario_name, output_dir, filename, color, warmup_duration_s):
        fig, ax = plt.subplots(figsize=(14, 7))
        if times and counts:
            ax.plot(times, counts, color=color, label='Numero di Pod', alpha=0.8, linewidth=1.5)
            ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                       label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'Evoluzione del Numero di Pod ({scenario_name}) - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(0, self.config.MAX_PODS + 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_pod_history_analysis(self, output_dir, warmup_duration_s):
        times_b, pods_b = zip(*self.metrics.pod_count_history) if self.metrics.pod_count_history else ([], [])
        self._plot_pod_history(times_b, pods_b, "Senza Priorità", output_dir, "pod_history_baseline.png", 'r', warmup_duration_s)
        self._plot_pod_history(self.metrics_prio.timestamps, self.metrics_prio.pod_counts,
                               "Con Priorità", output_dir, "pod_history_prio.png", 'b', warmup_duration_s)

        # Confronto
        fig, ax = plt.subplots(figsize=(14, 7))
        if times_b and pods_b:
            ax.plot(times_b, pods_b, 'r', label='Senza Priorità', alpha=0.8, linewidth=1.5)
        if self.metrics_prio.timestamps and self.metrics_prio.pod_counts:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, 'b', label='Con Priorità', alpha=0.8, linewidth=1.5)
        ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                   label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        ax.set_title('Confronto Evoluzione del Numero di Pod - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(0, self.config.MAX_PODS + 1)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, "pod_history_comparison.png", fig)

    # --- Queue History ---
    def _plot_queue_history(self, times, queue_lengths, scenario_name, output_dir, filename, color, use_log_scale, warmup_duration_s):
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'
        dark_color_map = {'r': 'darkred', 'b': 'darkblue'}

        if times and queue_lengths:
            ax.plot(times, queue_lengths, color=color, label='Lunghezza Coda', alpha=0.7, linewidth=1.5)
            ax.axhline(np.mean(queue_lengths), color=dark_color_map.get(color, color),
                       linestyle='--', label=f'Media Steady-State: {np.mean(queue_lengths):.2f}')
            ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                       label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)

        if use_log_scale:
            ax.set_yscale('log')
            ylabel += ' (Scala Log)'
            ax.set_ylim(bottom=0.1)

        ax.set_title(f'Evoluzione Lunghezza della Coda ({scenario_name}) - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_queue_history_analysis(self, output_dir, use_log_scale=True, warmup_duration_s=None):
        warmup_duration_s = warmup_duration_s or self.config.WARM_UP_TO_STEADY
        times_b, queue_b = zip(*self.metrics.queue_length_history) if self.metrics.queue_length_history else ([], [])
        self._plot_queue_history(times_b, queue_b, "Senza Priorità", output_dir,
                                 f"queue_history_baseline{'_log' if use_log_scale else ''}.png", 'r', use_log_scale, warmup_duration_s)
        self._plot_queue_history(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, "Con Priorità",
                                 output_dir, f"queue_history_prio{'_log' if use_log_scale else ''}.png", 'b', use_log_scale, warmup_duration_s)

        # Confronto
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'
        for times, queue, color, label in [(times_b, queue_b, 'r', 'Senza Priorità'),
                                           (self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, 'b', 'Con Priorità')]:
            if times and queue:
                ax.plot(times, queue, color=color, label=label, alpha=0.7, linewidth=1.5)
                ax.axhline(np.mean(queue), color='darkred' if color=='r' else 'darkblue', linestyle=':',
                           label=f'Media Steady ({label}): {np.mean(queue):.2f}')
        if use_log_scale:
            ax.set_yscale('log'); ylabel += ' (Scala Log)'; ax.set_ylim(bottom=0.1)
        ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                   label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        ax.set_title('Confronto Evoluzione della Lunghezza della Coda - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, f"queue_history_comparison{'_log' if use_log_scale else ''}.png", fig)

    # --- Variance Trend ---
    def _plot_variance_trend(self, all_responses, scenario_name, output_dir, filename, color, window_size, warmup_duration_s):
        fig, ax = plt.subplots(figsize=(14, 7))
        if len(all_responses) > window_size:
            times, values = zip(*all_responses)
            moving_std = pd.Series(values).rolling(window=window_size).std()
            ax.plot(times[window_size-1:], moving_std.iloc[window_size-1:], color=color, label='Dev. Std. Mobile', alpha=0.8)
            ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                       label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        else:
            ax.text(0.5, 0.5, f"Dati insufficienti (necessari > {window_size})", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'Stabilizzazione Varianza ({scenario_name}) - Finestra di {window_size} - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_variance_trend_analysis(self, output_dir,  warmup_duration_s,window_size=500):
        warmup_duration_s = warmup_duration_s or self.config.WARM_UP_TO_STEADY
        self._plot_variance_trend(self.metrics.get_all_response_times_with_timestamps(), "Senza Priorità",
                                  output_dir, "variance_trend_baseline.png", 'r', window_size, warmup_duration_s)
        self._plot_variance_trend(self.metrics_prio.get_all_response_times_with_timestamps(), "Con Priorità",
                                  output_dir, "variance_trend_prio.png", 'b', window_size, warmup_duration_s)

        # Confronto
        fig, ax = plt.subplots(figsize=(14, 7))
        for data, color, label in [(self.metrics.get_all_response_times_with_timestamps(), 'r', 'Senza Priorità'),
                                   (self.metrics_prio.get_all_response_times_with_timestamps(), 'b', 'Con Priorità')]:
            if len(data) > window_size:
                times, values = zip(*data)
                moving_std = pd.Series(values).rolling(window=window_size).std()
                ax.plot(times[window_size-1:], moving_std.iloc[window_size-1:], color=color, label=label, alpha=0.8)
        ax.axvline(x=warmup_duration_s, color='k', linestyle=':', linewidth=2.5,
                   label=f'Inizio Steady-state ({warmup_duration_s:.0f}s)')
        ax.set_title(f'Confronto Stabilizzazione Varianza (Finestra di {window_size}) - Dati Post-Warmup')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "variance_trend_comparison.png", fig)