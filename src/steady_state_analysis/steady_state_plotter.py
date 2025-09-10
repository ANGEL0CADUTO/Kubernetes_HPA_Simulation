import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import make_interp_spline

from src.utils.acs import batch_means, compute_batch_size
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.config import RequestType
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.utils.welford import Welford
from scipy.interpolate import make_interp_spline
from scipy.stats import t

class SteadyStatePlotter:
    """
    Classe responsabile della generazione di grafici e report per l'analisi dello stato stazionario.
    Visualizza i risultati della simulazione per baseline, scenari con priorità e WFQ,
    includendo analisi di convergenza, varianza, probabilità di perdita, throughput e tempi di risposta/attesa.
    """
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, metrics_wfq: MetricsWithPriority, config, use_log_scale_infinite: bool = True):
        """
        Inizializza SteadyStatePlotter con i dati delle metriche e la configurazione.

        Args:
            metrics (Metrics): Oggetto Metrics per lo scenario baseline.
            metrics_prio (MetricsWithPriority): Oggetto MetricsWithPriority per lo scenario con priorità.
            metrics_wfq (MetricsWithPriority): Oggetto MetricsWithPriority per lo scenario WFQ.
            config: Oggetto di configurazione della simulazione.
            use_log_scale_infinite (bool): Se True, abilita la scala logaritmica per alcuni grafici
                                           che potrebbero avere valori molto piccoli (es. probabilità di perdita).
        """
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.metrics_wfq = metrics_wfq
        self.config = config

        # --- Unified Style Definitions ---
        self.scenario_colors = {
            "Senza Priorità": '#E41A1C', # Red for Baseline
            "Con Priorità": '#377EB8',  # Blue for Priority
            "WFQ": '#4DAF4A'         # Green for WFQ
        }
        # MODIFIED: All linestyles changed to solid ('-') as per user request.
        self.scenario_linestyles = {
            "Senza Priorità": '-',
            "Con Priorità": '-',
            "WFQ": '-'
        }
        self.request_type_colors = { # Keep existing for internal request type differentiation
            RequestType.ADD_TO_CART: '#FF1493', # DeepPink
            RequestType.ANALYTICS: '#00BFFF',   # DeepSkyBlue
            RequestType.CHECKOUT: '#32CD32',    # LimeGreen
            RequestType.LOGIN: '#FFD700',       # Gold
            RequestType.NAVIGATION: '#9400D3'  # DarkViolet
        }
        # Line styles for response vs wait within a request type grid (to differentiate metrics within same scenario)
        # These are internal differentiation, so they can remain distinct if desired, but making them solid for now.
        self.response_line_style = '-' # Changed to solid
        self.wait_line_style = '-'     # Changed to solid

        self.warmup_line_style = ':'
        self.warmup_line_color = 'r'  # Colore rosso
        self.batch_mean_marker = 'o'
        self.ci_error_bar_color = 'black'
        # --- End Unified Style Definitions ---

        self.use_log_scale_infinite = use_log_scale_infinite

    # ==============================================================================
    # ORCHESTRATORE PRINCIPALE
    # ==============================================================================

    def generate_steady_state_report(self, warmup: dict, response_time_results: dict, throughput_results: dict, output_dir: str = "plots/steady_state"):
        """
        Orchestra la generazione di tutti i grafici e report per l'analisi dello stato stazionario.

        Args:
            warmup (dict): Dizionario contenente le durate di warm-up stimate per ogni scenario.
                           Es: {"baseline": 100.0, "priority": 120.0, "wfq": 90.0}.
            response_time_results (dict): Dizionario con i risultati del Batch Means per i tempi di risposta (overall).
                                          Contiene le chiavi "baseline", "priority", "wfq", e per ciascuna un dict
                                          con "num_batches", "mean", "ci", etc.
            throughput_results (dict): Dizionario con i risultati del Batch Means per il throughput (overall).
            output_dir (str): Directory dove salvare i grafici generati.
        """
        print(f"\n--- INIZIO Generazione Report Completo in '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)

        print("\n--- [SEZIONE 1/4] Analisi Performance a Regime ---")
        self.plot_steady_state_times_by_type(warmup, os.path.join(output_dir, "times_analysis"))
        self.plot_throughput_analysis(warmup, os.path.join(output_dir, "throughput_analysis"))
        self.plot_steady_state_loss(warmup, os.path.join(output_dir, "loss_analysis"))

        print("\n--- [SEZIONE 2/4] Analisi Comportamento del Sistema ---")
        self.plot_pod_history_analysis(os.path.join(output_dir, "pod_history_analysis"), warmup=warmup)
        self.plot_queue_history_analysis(warmup, os.path.join(output_dir, "queue_history_analysis"), use_log_scale=True)
        self.plot_wait_time_trend_analysis(os.path.join(output_dir, "wait_time_trend_analysis"), warmup=warmup)

        print("\n--- [SEZIONE 3/4] Analisi del Transitorio e della Stabilità ---")
        self.plot_convergence_analysis_overall(os.path.join(output_dir, "convergence_overall_analysis"), warmup_durations=warmup)
        self.plot_convergence_analysis_by_type(os.path.join(output_dir, "convergence_by_type_analysis"), warmup_durations=warmup)
        self.plot_variance_trend_analysis(os.path.join(output_dir, "variance_trend_analysis"), warmup_durations=warmup)
        self.plot_batch_mean_queue_trend_analysis(warmup, response_time_results, os.path.join(output_dir, "queue_batch_means_analysis"))

        print("\n--- [SEZIONE 4/4] Analisi Dettagliate per Tipo di Richiesta ---")
        self.plot_times_by_request_type_grid(os.path.join(output_dir, "detailed_grid_analysis"), warmup_durations=warmup)

        print(f"\n--- FINE Generazione Report. Controlla la cartella '{output_dir}'. ---")

    # ==============================================================================
    # METODI DI PLOTTING COMPLETI E CORRETTI
    # ==============================================================================

    def _save_plot(self, output_dir: str, filename: str, fig: plt.Figure):
        """
        Salva un grafico nella directory specificata e chiude la figura.

        Args:
            output_dir (str): Directory di destinazione.
            filename (str): Nome del file per il grafico.
            fig (plt.Figure): La figura Matplotlib da salvare.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def plot_steady_state_times_by_type(self, warmup: dict, output_dir: str):
        """
        Genera un grafico a barre di confronto per i tempi medi di risposta e attesa
        (per tipo di richiesta) per tutti gli scenari, con intervalli di confidenza al 95%.

        Args:
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
            output_dir (str): Directory dove salvare i grafici.
        """
        print("Generazione grafico di CONFRONTO per tempi per tipo di richiesta...")

        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True, layout="constrained")
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace("_", " ").title() for req in all_req_types]

        baseline_warmup_duration = warmup.get("baseline", 0.0)
        priority_warmup_duration = warmup.get("priority", 0.0)
        wfq_warmup_duration = warmup.get("wfq", 0.0)

        for metric_name, ax in zip(["response", "wait"], axes):
            if self.use_log_scale_infinite:
                ax.set_yscale("log")

            plot_data = []

            def get_batch_means_for_type(metrics_obj, req_type: RequestType, metric_key: str, current_warmup_duration: float, is_prio_or_wfq: bool) -> dict | None:
                """Helper per calcolare i batch means per una metrica specifica e tipo di richiesta."""
                values = []
                if is_prio_or_wfq: # MetricsWithPriority or WFQ metrics objects
                    raw_values = metrics_obj.response_times_by_req_type.get(req_type, []) if metric_key == "response" else metrics_obj.wait_times_by_req_type.get(req_type, [])
                    raw_timestamps = metrics_obj.completion_timestamps_by_req_type.get(req_type, [])
                    if len(raw_values) == len(raw_timestamps):
                        raw_data = sorted(zip(raw_timestamps, raw_values), key=lambda x: x[0])
                        values = [v for t, v in raw_data if t >= current_warmup_duration]
                else: # Base Metrics object
                    raw_data = metrics_obj.response_times_history.get(req_type, []) if metric_key == "response" else metrics_obj.wait_times_history.get(req_type, [])
                    values = [v for t, v in raw_data if t >= current_warmup_duration]

                if len(values) < 30: # Minimum data points for batch means to be meaningful
                    return None

                b, k, _ = compute_batch_size(values, k_initial_target=self.config.BATCH_K,threshold=self.config.BATCH_THRESHOLD)

                if b is None or k is None or b * k == 0: # Ensure valid b, k, and enough data for batches
                    return None
                if b * k > len(values): # This condition should be covered by compute_batch_size, but as a safeguard.
                    return None

                return batch_means(values, b, k, confidence=self.config.CONFIDENCE_LEVEL)

            for req_type in all_req_types:
                # Baseline (Senza Priorità)
                if res_b := get_batch_means_for_type(self.metrics, req_type, metric_name, baseline_warmup_duration, False):
                    plot_data.append({
                        "Categoria": req_type.name.replace("_", " ").title(),
                        "Tempo Medio (s)": float(res_b["mean"]),
                        "Errore": float(res_b["half_width"]),
                        "Scenario": "Senza Priorità",
                    })

                # Con Priorità
                if res_p := get_batch_means_for_type(self.metrics_prio, req_type, metric_name, priority_warmup_duration, True):
                    plot_data.append({
                        "Categoria": req_type.name.replace("_", " ").title(),
                        "Tempo Medio (s)": float(res_p["mean"]),
                        "Errore": float(res_p["half_width"]),
                        "Scenario": "Con Priorità",
                    })

                # WFQ
                if res_wfq := get_batch_means_for_type(self.metrics_wfq, req_type, metric_name, wfq_warmup_duration, True):
                    plot_data.append({
                        "Categoria": req_type.name.replace("_", " ").title(),
                        "Tempo Medio (s)": float(res_wfq["mean"]),
                        "Errore": float(res_wfq["half_width"]),
                        "Scenario": "WFQ",
                    })

            if not plot_data:
                ax.text(0.5, 0.5, "Nessun dato valido per il plotting.", ha='center', va='center', fontsize=12)
                ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio (Nessun Dato)")
                plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
                continue

            df = pd.DataFrame(plot_data)
            df["Categoria"] = df["Categoria"].astype("category").cat.reorder_categories(category_names, ordered=True)
            df["Scenario"] = df["Scenario"].astype("category").cat.reorder_categories(list(self.scenario_colors.keys()), ordered=True)

            sns.barplot(
                data=df,
                x="Categoria", y="Tempo Medio (s)",
                hue="Scenario",
                palette=list(self.scenario_colors.values()),
                ax=ax, dodge=True
            )

            num_categories = len(category_names)

            # Extract bar width from the first bar in the plot, assuming uniform width
            bar_width_per_scenario = 0.2 # Default if no bars are plotted yet
            if ax.containers:
                # Find the first non-empty container to get a bar patch
                for container in ax.containers:
                    if container.patches:
                        bar_width_per_scenario = container.patches[0].get_width()
                        break


            total_group_width = bar_width_per_scenario * len(self.scenario_colors)
            offset_center = -total_group_width / 2 + bar_width_per_scenario / 2

            for i, scenario_name in enumerate(list(self.scenario_colors.keys())):
                offset = offset_center + i * bar_width_per_scenario
                subset = df[df["Scenario"] == scenario_name].set_index("Categoria").reindex(category_names)

                if subset["Tempo Medio (s)"].isnull().all():
                    continue

                x_coords_for_error = np.arange(num_categories) + offset
                y_coords = subset["Tempo Medio (s)"].fillna(0).values
                errors = subset["Errore"].fillna(0).values

                ax.errorbar(x_coords_for_error, y_coords, yerr=errors, fmt="none", c=self.ci_error_bar_color, capsize=5, elinewidth=1.2)

                for cat_idx, category_name in enumerate(category_names):
                    row = subset.loc[category_name]
                    if pd.notna(row["Tempo Medio (s)"]):
                        mean_val, error_val = row["Tempo Medio (s)"], row["Errore"]
                        upper_bound = mean_val + error_val
                        # Ensure lower bound is not negative if using log scale, or for physical quantities
                        lower_bound = max(0, mean_val - error_val)
                        ci_text = f"[{lower_bound:.3f}, {upper_bound:.3f}]"

                        # Position CI text slightly above the error bar cap
                        ax.annotate(
                            ci_text,
                            xy=(x_coords_for_error[cat_idx], upper_bound),
                            xytext=(0, 3), # Small vertical offset
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            color="black"
                        )

            ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio")
            ax.set_xlabel("Tipo di Richiesta")
            ax.set_ylabel("Tempo Medio (s)")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            ax.legend(title="Scenario")

            # Adjust y-limit to ensure annotations are visible
            current_bottom, current_top = ax.get_ylim()
            # If log scale, ensure top is slightly above the highest annotation
            if self.use_log_scale_infinite and ax.get_yscale() == 'log':
                max_ci_upper = df.apply(lambda row: row['Tempo Medio (s)'] + row['Errore'] if pd.notna(row['Tempo Medio (s)']) else 0, axis=1).max()
                if max_ci_upper > 0:
                    ax.set_ylim(top=max_ci_upper * 1.5) # Increase by 50% for log scale
            else:
                ax.set_ylim(bottom=current_bottom, top=current_top * 1.05) # Increased top margin for labels

        fig.suptitle("Confronto Tempi Medi (Steady State) per Tipo con IC al 95%")
        self._save_plot(output_dir, "times_comparison.png", fig)

    def plot_steady_state_loss(self, warmup: dict, output_dir: str):
        """
        Genera il grafico della probabilità di perdita a regime per tutti gli scenari,
        includendo il confronto con intervalli di confidenza al 95%.

        Args:
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
            output_dir (str): Directory dove salvare i grafici.
        """
        print("Generazione grafico di CONFRONTO per probabilità di perdita...")

        # `get_all_outcomes_as_binary_stream` should return a list of (timestamp, 0 or 1)
        baseline_outcomes = self.metrics.get_all_outcomes_as_binary_stream()
        prio_outcomes = self.metrics_prio.get_all_outcomes_as_binary_stream()
        wfq_outcomes = self.metrics_wfq.get_all_outcomes_as_binary_stream()

        def compute_loss_ci(outcomes: list[tuple[float, int]], warmup_period: float, confidence_level: float, threshold: float) -> dict | None:
            """
            Calcola la probabilità di perdita e il suo intervallo di confidenza
            per una serie di risultati binari (0=successo, 1=perdita).
            """
            if not outcomes:
                return None

            # Filter out warmup data using timestamps
            steady_values = [float(v) for t, v in outcomes if t >= warmup_period]

            if len(steady_values) < 30: # Minimum data points for batch means to be meaningful
                print(f"  DEBUG: Dati insufficienti per calcolare loss CI, solo {len(steady_values)} punti dopo warmup.")
                return None

            # MODIFIED: Changed k_initial_target to k_min_target
            b_k_rho_tuple = compute_batch_size(steady_values,k_initial_target=self.config.BATCH_K, threshold=threshold)

            if b_k_rho_tuple is None: # compute_batch_size returns (None, None, None)
                print(f"  DEBUG: compute_batch_size ha restituito None per la perdita. Non calcolo il CI.")
                return None

            b_val, k_val, _ = b_k_rho_tuple
            b, k = int(b_val), int(k_val) # Convert to int after unpacking

            # Additional checks after getting b and k
            if b <= 0 or k <= 1: # k <= 1 means not enough batches for CI
                print(f"  DEBUG: Batch size (b={b}) or num batches (k={k}) insufficiente per loss CI.")
                return None
            if b * k == 0 or b * k > len(steady_values):
                print(f"  DEBUG: b*k ({b*k}) supera la lunghezza dei dati ({len(steady_values)}) per loss CI.")
                return None

            res = batch_means(steady_values, b, k, confidence=confidence_level)

            if res is None:
                return None

            # Ensure mean and half_width are floats, handling potential numpy types
            res['mean'] = float(res['mean']) if isinstance(res['mean'], (int, float, np.number)) else np.nan
            res['half_width'] = float(res['half_width']) if isinstance(res['half_width'], (int, float, np.number)) else np.nan
            res['batch_size'] = int(res.get('batch_size', b))
            res['num_batches'] = int(res.get('num_batches', k))
            return res

        baseline_results = compute_loss_ci(baseline_outcomes, warmup.get("baseline", 0.0), self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD)
        prio_results = compute_loss_ci(prio_outcomes, warmup.get("priority", 0.0), self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD)
        wfq_results = compute_loss_ci(wfq_outcomes, warmup.get("wfq", 0.0), self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD)

        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        plot_data = []
        if baseline_results and not np.isnan(baseline_results['mean']):
            plot_data.append({'Scenario': 'Senza Priorità', 'Mean': baseline_results['mean'], 'Half_Width': baseline_results['half_width']})
        if prio_results and not np.isnan(prio_results['mean']):
            plot_data.append({'Scenario': 'Con Priorità', 'Mean': prio_results['mean'], 'Half_Width': prio_results['half_width']})
        if wfq_results and not np.isnan(wfq_results['mean']):
            plot_data.append({'Scenario': 'WFQ', 'Mean': wfq_results['mean'], 'Half_Width': wfq_results['half_width']})

        if not plot_data:
            ax.text(0.5, 0.5, "Nessun dato disponibile per il confronto di perdita.", ha='center', va='center', fontsize=12)
            ax.set_title('Confronto Probabilità di Perdita (Nessun Dato)')
        else:
            df_loss = pd.DataFrame(plot_data)
            df_loss['Mean'] = df_loss['Mean'].astype(float)
            df_loss['Half_Width'] = df_loss['Half_Width'].astype(float)
            df_loss["Scenario"] = df_loss["Scenario"].astype("category").cat.reorder_categories(list(self.scenario_colors.keys()), ordered=True)

            # Conditional log scale
            # Only apply log scale if all means are strictly positive and no NaNs
            if self.use_log_scale_infinite and (df_loss['Mean'] > 0).all() and not df_loss['Mean'].isnull().any():
                ax.set_yscale('log')
                min_positive_mean = df_loss['Mean'][df_loss['Mean'] > 0].min()
                if not np.isnan(min_positive_mean):
                    # Ensure bottom limit is positive and visible for log scale
                    ax.set_ylim(bottom=max(1e-9, min_positive_mean * 0.1))
            else:
                max_y_val = (df_loss['Mean'] + df_loss['Half_Width']).max()
                if not np.isnan(max_y_val):
                    ax.set_ylim(bottom=0, top=max_y_val * 1.2)
                else:
                    ax.set_ylim(bottom=0, top=1) # Default if max_y_val is NaN

            bars = sns.barplot(data=df_loss, x='Scenario', y='Mean',
                               hue='Scenario', palette=list(self.scenario_colors.values()),
                               ax=ax, legend=False)

            x_coords = np.arange(len(df_loss['Scenario']))
            ax.errorbar(x_coords, df_loss['Mean'], yerr=df_loss['Half_Width'],
                        fmt="none", c=self.ci_error_bar_color, capsize=10, elinewidth=1.5)

            ax.set_title('Confronto Probabilità di Perdita (Steady State) con IC al 95%')
            ax.set_ylabel('Probabilità di Perdita Stimata')
            ax.set_xlabel('Scenario')
            ax.grid(True, axis='y', alpha=0.7)

            for i, bar in enumerate(bars.patches):
                mean_val = float(df_loss['Mean'].iloc[i])
                half_width = float(df_loss['Half_Width'].iloc[i])
                upper_bound = mean_val + half_width

                if np.isnan(mean_val) or np.isnan(half_width):
                    text_val = "N/A"
                    ci_text = "N/A"
                else:
                    text_val = f'{mean_val:.4f}'
                    ci_text = f"[{max(0, mean_val - half_width):.4f}, {upper_bound:.4f}]"

                # Adjust text position for log scale if applicable
                y_pos_mean = mean_val
                y_pos_ci = upper_bound
                if ax.get_yscale() == 'log':
                    # For log scale, positions should be relative to log values, but text still readable.
                    # Place annotations slightly above the bar/error bar in data coordinates.
                    y_pos_mean = mean_val if mean_val > 0 else (ax.get_ylim()[0] * 1.1)
                    y_pos_ci = upper_bound if upper_bound > 0 else (ax.get_ylim()[0] * 1.2)


                ax.annotate(text_val,
                            xy=(bar.get_x() + bar.get_width() / 2, y_pos_mean),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=10, weight='bold')

                ax.annotate(ci_text,
                            xy=(bar.get_x() + bar.get_width() / 2, y_pos_ci),
                            xytext=(0, 14), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)

            # Adjust y-limit after annotations to ensure they fit
            current_bottom, current_top = ax.get_ylim()
            if ax.get_yscale() == 'log':
                max_ci_upper_overall = df_loss.apply(lambda row: row['Mean'] + row['Half_Width'] if pd.notna(row['Mean']) else 0, axis=1).max()
                if max_ci_upper_overall > 0:
                    ax.set_ylim(top=max_ci_upper_overall * 2) # More aggressive scaling for log, if needed
            else:
                ax.set_ylim(top=current_top * 1.2)


        self._save_plot(output_dir, "loss_probability_comparison.png", fig)

    def _plot_convergence_baseline_by_type(self, output_dir: str, warmup_duration: float):
        """
        Genera il grafico della convergenza del tempo di risposta medio cumulativo
        per ogni tipo di richiesta nello scenario Baseline.

        Args:
            output_dir (str): Directory dove salvare il grafico.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for req_type, history in self.metrics.response_times_history.items():
            if history: # Ensure history is not empty
                # Sort history by timestamp (x[0])
                sorted_history = sorted(history, key=lambda x: x[0])
                timestamps, values = zip(*sorted_history)

                # Plot only if there's enough data after warm-up
                if any(t >= warmup_duration for t in timestamps):
                    # Filter data to only include post-warmup for smoother convergence, but plot full for visualization
                    plot_data_times = list(timestamps)
                    plot_data_values = list(values)

                    cumulative_means = np.cumsum(plot_data_values) / np.arange(1, len(plot_data_values) + 1)

                    if len(plot_data_times) > 1:
                        # Use min(3, len(data_points)-1) to ensure k is valid for spline interpolation.
                        # k must be <= len(points) - 1. A k=1 (linear) is a fallback.
                        k_val = min(3, len(plot_data_times) - 1)
                        if k_val >= 1: # Only try to smooth if k_val is valid (i.e., at least 2 points)
                            x_smooth = np.linspace(min(plot_data_times), max(plot_data_times), 500)
                            spl = make_interp_spline(plot_data_times, cumulative_means, k=k_val)
                            y_smooth = spl(x_smooth)
                            ax.plot(x_smooth, y_smooth, label=f'{req_type.name.replace("_", " ").title()} (Smoothed)',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle=self.scenario_linestyles["Senza Priorità"])
                        else: # Fallback to raw plot if only 1 point or less, or k_val invalid
                            ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid
                    else: # Fallback for single point
                        ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid

        ax.set_title('Analisi Convergenza per Tipo (Senza Priorità)')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')
        self._save_plot(output_dir, "baseline_convergence_by_type.png", fig)

    def _plot_convergence_prio_by_type(self, output_dir: str, warmup_duration: float):
        """
        Genera il grafico della convergenza del tempo di risposta medio cumulativo
        per ogni tipo di richiesta nello scenario con Priorità.

        Args:
            output_dir (str): Directory dove salvare il grafico.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                # Sort history by timestamp (x[0])
                history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
                sorted_timestamps, sorted_values = zip(*history)

                if any(t >= warmup_duration for t in sorted_timestamps):
                    plot_data_times = list(sorted_timestamps)
                    plot_data_values = list(sorted_values)

                    cumulative_means = np.cumsum(plot_data_values) / np.arange(1, len(plot_data_values) + 1)

                    if len(plot_data_times) > 1:
                        k_val = min(3, len(plot_data_times) - 1)
                        if k_val >= 1:
                            x_smooth = np.linspace(min(plot_data_times), max(plot_data_times), 500)
                            spl = make_interp_spline(plot_data_times, cumulative_means, k=k_val)
                            y_smooth = spl(x_smooth)
                            ax.plot(x_smooth, y_smooth, label=f'{req_type.name.replace("_", " ").title()} (Smoothed)',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle=self.scenario_linestyles["Con Priorità"])
                        else:
                            ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid
                    else:
                        ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid
        ax.set_title('Analisi Convergenza per Tipo (Con Priorità)')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')
        self._save_plot(output_dir, "prio_convergence_by_type.png", fig)

    def _plot_convergence_wfq_by_type(self, output_dir: str, warmup_duration: float):
        """
        Genera il grafico della convergenza del tempo di risposta medio cumulativo
        per ogni tipo di richiesta nello scenario WFQ.

        Args:
            output_dir (str): Directory dove salvare il grafico.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for req_type in sorted(self.metrics_wfq.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_wfq.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                # Sort history by timestamp (x[0])
                history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
                sorted_timestamps, sorted_values = zip(*history)

                if any(t >= warmup_duration for t in sorted_timestamps):
                    plot_data_times = list(sorted_timestamps)
                    plot_data_values = list(sorted_values)

                    cumulative_means = np.cumsum(plot_data_values) / np.arange(1, len(plot_data_values) + 1)

                    if len(plot_data_times) > 1:
                        k_val = min(3, len(plot_data_times) - 1)
                        if k_val >= 1:
                            x_smooth = np.linspace(min(plot_data_times), max(plot_data_times), 500)
                            spl = make_interp_spline(plot_data_times, cumulative_means, k=k_val)
                            y_smooth = spl(x_smooth)
                            ax.plot(x_smooth, y_smooth, label=f'{req_type.name.replace("_", " ").title()} (Smoothed)',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle=self.scenario_linestyles["WFQ"])
                        else:
                            ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                    color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid
                    else:
                        ax.plot(plot_data_times, cumulative_means, label=f'{req_type.name.replace("_", " ").title()}',
                                color=self.request_type_colors.get(req_type), linewidth=2, linestyle='-') # Always solid
        ax.set_title('Analisi Convergenza per Tipo (WFQ)')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Tipo di Richiesta')
        self._save_plot(output_dir, "wfq_convergence_by_type.png", fig)

    def plot_convergence_analysis_by_type(self, output_dir: str, warmup_durations: dict):
        """
        Genera grafici della convergenza per tipo di richiesta per ogni scenario,
        inclusi un grafico di confronto tra gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup_durations (dict): Dizionario con le durate di warm-up per ogni scenario.
        """
        baseline_warmup = warmup_durations.get("baseline", 0.0)
        priority_warmup = warmup_durations.get("priority", 0.0)
        wfq_warmup = warmup_durations.get("wfq", 0.0)

        self._plot_convergence_baseline_by_type(output_dir, warmup_duration=baseline_warmup)
        self._plot_convergence_prio_by_type(output_dir, warmup_duration=priority_warmup)
        self._plot_convergence_wfq_by_type(output_dir, warmup_duration=wfq_warmup)

        print("Generazione grafico di CONFRONTO di convergenza per tipo...")
        fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")

        def plot_smoothed_cumulative_mean(ax_obj: plt.Axes, timestamps: list[float], values: list[float], label: str, color: str, linestyle: str):
            """Helper per plottare la media cumulativa smoothed."""
            if values and timestamps and len(timestamps) > 0:
                cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)
                if len(timestamps) > 1:
                    k_val = min(3, len(timestamps) - 1)
                    if k_val >= 1:
                        # Use a reasonable number of points for smoothing, e.g., 500
                        x_smooth = np.linspace(min(timestamps), max(timestamps), 500)
                        spl = make_interp_spline(timestamps, cumulative_means, k=k_val)
                        y_smooth = spl(x_smooth)
                        ax_obj.plot(x_smooth, y_smooth, color=color, label=label, linestyle=linestyle, linewidth=2.5)
                    else:
                        ax_obj.plot(timestamps, cumulative_means, label=label, color=color, linestyle='-', linewidth=2.5) # Always solid
                else:
                    ax_obj.plot(timestamps, cumulative_means, label=label, color=color, linestyle='-', linewidth=2.5) # Always solid


        for req_type, history in self.metrics.response_times_history.items():
            if history:
                timestamps, values = zip(*sorted(history, key=lambda x: x[0]))
                plot_smoothed_cumulative_mean(ax, list(timestamps), list(values), f'{req_type.name.replace("_", " ").title()} (Baseline)', self.request_type_colors.get(req_type), self.scenario_linestyles["Senza Priorità"])

        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                sorted_timestamps, sorted_values = zip(*sorted(zip(timestamps, response_times), key=lambda x: x[0]))
                plot_smoothed_cumulative_mean(ax, list(sorted_timestamps), list(sorted_values), f'{req_type.name.replace("_", " ").title()} (Priorità)', self.request_type_colors.get(req_type), self.scenario_linestyles["Con Priorità"])

        for req_type in sorted(self.metrics_wfq.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_wfq.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                sorted_timestamps, sorted_values = zip(*sorted(zip(timestamps, response_times), key=lambda x: x[0]))
                plot_smoothed_cumulative_mean(ax, list(sorted_timestamps), list(sorted_values), f'{req_type.name.replace("_", " ").title()} (WFQ)', self.request_type_colors.get(req_type), self.scenario_linestyles["WFQ"])

        # Add only one warmup line, e.g., for baseline, as it's a comparison plot
        ax.axvline(x=baseline_warmup, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({baseline_warmup:.2f}s)')
        ax.set_title('Confronto Convergenza per Tipo di Richiesta')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Scenario e Tipo', bbox_to_anchor=(1.04, 1), loc="upper left")
        self._save_plot(output_dir, "convergence_by_type_comparison.png", fig)

    def _plot_convergence_overall(self, all_responses: list[tuple[float, float]], scenario_name: str, output_dir: str, filename: str, color: str, linestyle: str, warmup_duration: float):
        """
        Genera il grafico della convergenza del tempo di risposta medio cumulativo
        complessivo per un singolo scenario.

        Args:
            all_responses (list[tuple[float, float]]): Lista di tuple (timestamp, tempo_risposta).
            scenario_name (str): Nome dello scenario (es. "Senza Priorità").
            output_dir (str): Directory dove salvare il grafico.
            filename (str): Nome del file del grafico.
            color (str): Colore della linea.
            linestyle (str): Stile della linea.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        if all_responses:
            # Sort by timestamp (x[0])
            sorted_responses = sorted(all_responses, key=lambda x: x[0])
            timestamps, values = zip(*sorted_responses)

            cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)
            if len(timestamps) > 1:
                k_val = min(3, len(timestamps) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(timestamps), max(timestamps), 500)
                    spl = make_interp_spline(timestamps, cumulative_means, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, label='Tempo Risposta Medio Cumulativo (Smoothed)', linestyle=linestyle)
                else:
                    ax.plot(timestamps, cumulative_means, color=color, label='Tempo Risposta Medio Cumulativo', linestyle='-') # Always solid
            else:
                ax.plot(timestamps, cumulative_means, color=color, label='Tempo Risposta Medio Cumulativo', linestyle='-') # Always solid
            ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Analisi Convergenza Tempo Risposta Medio ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend()
        self._save_plot(output_dir, filename, fig)

    def plot_convergence_analysis_overall(self, output_dir: str, warmup_durations: dict):
        """
        Genera grafici della convergenza complessiva del tempo di risposta medio
        per ogni scenario, inclusi un grafico di confronto tra gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup_durations (dict): Dizionario con le durate di warm-up per ogni scenario.
        """
        baseline_warmup = warmup_durations.get("baseline", 0.0)
        priority_warmup = warmup_durations.get("priority", 0.0)
        wfq_warmup = warmup_durations.get("wfq", 0.0)

        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        all_responses_wfq = self.metrics_wfq.get_all_response_times_with_timestamps()

        self._plot_convergence_overall(all_responses_base, "Senza Priorità", output_dir, "baseline_convergence_overall.png", self.scenario_colors["Senza Priorità"], self.scenario_linestyles["Senza Priorità"], warmup_duration=baseline_warmup)
        self._plot_convergence_overall(all_responses_prio, "Con Priorità", output_dir, "prio_convergence_overall.png", self.scenario_colors["Con Priorità"], self.scenario_linestyles["Con Priorità"], warmup_duration=priority_warmup)
        self._plot_convergence_overall(all_responses_wfq, "WFQ", output_dir, "wfq_convergence_overall.png", self.scenario_colors["WFQ"], self.scenario_linestyles["WFQ"], warmup_duration=wfq_warmup)

        print("Generazione grafico di CONFRONTO di convergenza generale...")
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")

        def plot_smoothed_cumulative_mean_overall(ax_obj: plt.Axes, all_responses: list[tuple[float, float]], color: str, label: str, linestyle: str):
            """Helper per plottare la media cumulativa smoothed complessiva."""
            if all_responses:
                # Sort by timestamp (x[0])
                sorted_responses = sorted(all_responses, key=lambda x: x[0])
                timestamps, values = zip(*sorted_responses)
                cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)
                if len(timestamps) > 1:
                    k_val = min(3, len(timestamps) - 1)
                    if k_val >= 1:
                        x_smooth = np.linspace(min(timestamps), max(timestamps), 500)
                        spl = make_interp_spline(timestamps, cumulative_means, k=k_val)
                        y_smooth = spl(x_smooth)
                        ax_obj.plot(x_smooth, y_smooth, color=color, label=label, linestyle=linestyle)
                    else:
                        ax_obj.plot(timestamps, cumulative_means, color=color, label=label, linestyle='-') # Always solid
                else:
                    ax_obj.plot(timestamps, cumulative_means, color=color, label=label, linestyle='-') # Always solid

        plot_smoothed_cumulative_mean_overall(ax, all_responses_base, self.scenario_colors["Senza Priorità"], 'Senza Priorità', self.scenario_linestyles["Senza Priorità"])
        plot_smoothed_cumulative_mean_overall(ax, all_responses_prio, self.scenario_colors["Con Priorità"], 'Con Priorità', self.scenario_linestyles["Con Priorità"])
        plot_smoothed_cumulative_mean_overall(ax, all_responses_wfq, self.scenario_colors["WFQ"], 'WFQ', self.scenario_linestyles["WFQ"])

        if all_responses_base or all_responses_prio or all_responses_wfq:
            # Add only one warmup line for comparison plot
            ax.axvline(x=baseline_warmup, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({baseline_warmup:.2f}s)')
        ax.set_title('Confronto Convergenza del Tempo di Risposta Medio')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Scenario')
        self._save_plot(output_dir, "convergence_overall_comparison.png", fig)

    def _plot_wait_time_trend(self, all_waits: list[tuple[float, float]], scenario_name: str, output_dir: str, filename: str, color: str, linestyle: str, warmup_duration: float):
        """
        Genera il grafico dell'evoluzione del tempo di attesa medio cumulativo
        per un singolo scenario.

        Args:
            all_waits (list[tuple[float, float]]): Lista di tuple (timestamp, tempo_attesa).
            scenario_name (str): Nome dello scenario.
            output_dir (str): Directory dove salvare il grafico.
            filename (str): Nome del file del grafico.
            color (str): Colore della linea.
            linestyle (str): Stile della linea.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        if all_waits:
            # Sort by timestamp (x[0])
            sorted_waits = sorted(all_waits, key=lambda x: x[0])
            times, values = zip(*sorted_waits)
            cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)
            if len(times) > 1:
                k_val = min(3, len(times) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(times), max(times), 500)
                    spl = make_interp_spline(times, cumulative_means, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, label='Tempo Attesa Medio Cumulativo (Smoothed)', linestyle=linestyle)
                else:
                    ax.plot(times, cumulative_means, color=color, label='Tempo Attesa Medio Cumulativo', linestyle='-') # Always solid
            else:
                ax.plot(times, cumulative_means, color=color, label='Tempo Attesa Medio Cumulativo', linestyle='-') # Always solid
            ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Evoluzione del Tempo di Attesa Medio ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend()
        self._save_plot(output_dir, filename, fig)

    def plot_wait_time_trend_analysis(self, output_dir: str, warmup: dict):
        """
        Genera grafici dell'evoluzione del tempo di attesa medio cumulativo
        per ogni scenario, inclusi un grafico di confronto tra gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
        """
        baseline_warmup_duration = warmup.get("baseline", 0.0)
        priority_warmup_duration = warmup.get("priority", 0.0)
        wfq_warmup_duration = warmup.get("wfq", 0.0)

        # Collect all wait times with timestamps for Baseline
        # self.metrics.wait_times_history stores {req_type: [(timestamp, wait_time), ...]}
        all_waits_base = []
        for req_type_waits in self.metrics.wait_times_history.values():
            all_waits_base.extend(req_type_waits)
        all_waits_base = sorted(all_waits_base, key=lambda x:x[0]) # Sort by timestamp

        # Collect all wait times with timestamps for Priority
        all_waits_prio_list = []
        for req_type in self.metrics_prio.wait_times_by_req_type.keys():
            waits = self.metrics_prio.wait_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_prio_list.extend(zip(timestamps, waits))
        all_waits_prio = sorted(all_waits_prio_list, key=lambda x: x[0]) # Sort by timestamp

        # Collect all wait times with timestamps for WFQ
        all_waits_wfq_list = []
        for req_type in self.metrics_wfq.wait_times_by_req_type.keys():
            waits = self.metrics_wfq.wait_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_wfq_list.extend(zip(timestamps, waits))
        all_waits_wfq = sorted(all_waits_wfq_list, key=lambda x: x[0]) # Sort by timestamp


        self._plot_wait_time_trend(all_waits_base, "Senza Priorità", output_dir, "wait_time_trend_baseline.png", self.scenario_colors["Senza Priorità"], self.scenario_linestyles["Senza Priorità"], warmup_duration=baseline_warmup_duration)
        self._plot_wait_time_trend(all_waits_prio, "Con Priorità", output_dir, "wait_time_trend_prio.png", self.scenario_colors["Con Priorità"], self.scenario_linestyles["Con Priorità"], warmup_duration=priority_warmup_duration)
        self._plot_wait_time_trend(all_waits_wfq, "WFQ", output_dir, "wait_time_trend_wfq.png", self.scenario_colors["WFQ"], self.scenario_linestyles["WFQ"], warmup_duration=wfq_warmup_duration)

        print("Generazione grafico di CONFRONTO andamento tempo di attesa...")
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")

        def plot_smoothed_cumulative_mean_wait(ax_obj: plt.Axes, all_waits: list[tuple[float, float]], color: str, label: str, linestyle: str, warmup_duration: float):
            """Helper per plottare la media cumulativa smoothed dei tempi di attesa."""
            if all_waits:
                # Sort by timestamp (x[0])
                sorted_waits = sorted(all_waits, key=lambda x: x[0])
                times, values = zip(*sorted_waits)
                cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)
                if len(times) > 1:
                    k_val = min(3, len(times) - 1)
                    if k_val >= 1:
                        x_smooth = np.linspace(min(times), max(times), 500)
                        spl = make_interp_spline(times, cumulative_means, k=k_val)
                        y_smooth = spl(x_smooth)
                        ax_obj.plot(x_smooth, y_smooth, color=color, label=label, linestyle=linestyle)
                    else:
                        ax_obj.plot(times, cumulative_means, color=color, label=label, linestyle='-') # Always solid
                else:
                    ax_obj.plot(times, cumulative_means, color=color, label=label, linestyle='-') # Always solid

                # Calculate and plot steady-state mean if enough data exists after warmup
                steady_values = [v for t, v in sorted_waits if t >= warmup_duration]
                if steady_values:
                    dark_color = 'gray' # Default fallback
                    if color == self.scenario_colors["Senza Priorità"]: dark_color = '#A60628' # Darker red
                    elif color == self.scenario_colors["Con Priorità"]: dark_color = '#1E5894' # Darker blue
                    elif color == self.scenario_colors["WFQ"]: dark_color = '#2E7C2E' # Darker green
                    ax_obj.axhline(np.mean(steady_values), color=dark_color, linestyle=':', label=f'Media Steady ({label}): {np.mean(steady_values):.2f}')

        plot_smoothed_cumulative_mean_wait(ax, all_waits_base, self.scenario_colors["Senza Priorità"], 'Senza Priorità', self.scenario_linestyles["Senza Priorità"], baseline_warmup_duration)
        plot_smoothed_cumulative_mean_wait(ax, all_waits_prio, self.scenario_colors["Con Priorità"], 'Con Priorità', self.scenario_linestyles["Con Priorità"], priority_warmup_duration)
        plot_smoothed_cumulative_mean_wait(ax, all_waits_wfq, self.scenario_colors["WFQ"], 'WFQ', self.scenario_linestyles["WFQ"], wfq_warmup_duration)

        ax.axvline(x=baseline_warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2, label=f'Fine Warm-up ({baseline_warmup_duration:.2f}s)')

        ax.set_title('Confronto Evoluzione del Tempo di Attesa Medio')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', alpha=0.7)
        ax.legend(title='Scenario')
        self._save_plot(output_dir, "wait_time_trend_comparison.png", fig)

    def _plot_pod_history(self, times: list[float], counts: list[int], scenario_name: str, output_dir: str, filename: str, color: str, linestyle: str, warmup_duration: float):
        """
        Genera il grafico dell'evoluzione del numero di pod attivi per un singolo scenario.

        Args:
            times (list[float]): Lista dei timestamp.
            counts (list[int]): Lista dei conteggi dei pod.
            scenario_name (str): Nome dello scenario.
            output_dir (str): Directory dove salvare il grafico.
            filename (str): Nome del file del grafico.
            color (str): Colore della linea.
            linestyle (str): Stile della linea.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        if times and counts:
            if len(times) > 1:
                k_val = min(3, len(times) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(times), max(times), 500)
                    spl = make_interp_spline(times, counts, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, label='Numero di Pod (Smoothed)', alpha=0.8, linewidth=1.5, linestyle=linestyle)
                else:
                    ax.plot(times, counts, color=color, label='Numero di Pod', alpha=0.8, linewidth=1.5, linestyle='-') # Always solid
            else:
                ax.plot(times, counts, color=color, label='Numero di Pod', alpha=0.8, linewidth=1.5, linestyle='-') # Always solid
            ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2.5, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(f'Evoluzione del Numero di Pod ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend()
        self._save_plot(output_dir, filename, fig)

    def plot_pod_history_analysis(self, output_dir: str, warmup: dict):
        """
        Genera grafici dell'evoluzione del numero di pod attivi per ogni scenario,
        inclusi un grafico di confronto tra gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
        """
        baseline_warmup_duration = warmup.get("baseline", 0.0)
        priority_warmup_duration = warmup.get("priority", 0.0)
        wfq_warmup_duration = warmup.get("wfq", 0.0)

        # Ensure times and pods are extracted correctly
        times_b, pods_b = zip(*self.metrics.pod_count_history) if self.metrics.pod_count_history else ([], [])
        self._plot_pod_history(list(times_b), list(pods_b), "Senza Priorità", output_dir, "pod_history_baseline.png", self.scenario_colors["Senza Priorità"], self.scenario_linestyles["Senza Priorità"], warmup_duration=baseline_warmup_duration)
        self._plot_pod_history(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, "Con Priorità", output_dir, "pod_history_prio.png", self.scenario_colors["Con Priorità"], self.scenario_linestyles["Con Priorità"], warmup_duration=priority_warmup_duration)
        self._plot_pod_history(self.metrics_wfq.timestamps, self.metrics_wfq.pod_counts, "WFQ", output_dir, "pod_history_wfq.png", self.scenario_colors["WFQ"], self.scenario_linestyles["WFQ"], warmup_duration=wfq_warmup_duration)

        print("Generazione grafico di CONFRONTO storico dei Pod...")
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")

        def plot_smoothed_pod_history(ax_obj: plt.Axes, times: list[float], counts: list[int], color: str, label: str, linestyle: str):
            """Helper per plottare la storia dei pod smoothed."""
            if times and counts:
                if len(times) > 1:
                    k_val = min(3, len(times) - 1)
                    if k_val >= 1:
                        x_smooth = np.linspace(min(times), max(times), 500)
                        spl = make_interp_spline(times, counts, k=k_val)
                        y_smooth = spl(x_smooth)
                        ax_obj.plot(x_smooth, y_smooth, color=color, label=label, alpha=0.8, linewidth=1.5, linestyle=linestyle)
                    else:
                        ax_obj.plot(times, counts, color=color, label=label, alpha=0.8, linewidth=1.5, linestyle='-') # Always solid
                else:
                    ax_obj.plot(times, counts, color=color, label=label, alpha=0.8, linewidth=1.5, linestyle='-') # Always solid

        plot_smoothed_pod_history(ax, list(times_b), list(pods_b), self.scenario_colors["Senza Priorità"], 'Senza Priorità', self.scenario_linestyles["Senza Priorità"])
        plot_smoothed_pod_history(ax, self.metrics_prio.timestamps, self.metrics_prio.pod_counts, self.scenario_colors["Con Priorità"], 'Con Priorità', self.scenario_linestyles["Con Priorità"])
        plot_smoothed_pod_history(ax, self.metrics_wfq.timestamps, self.metrics_wfq.pod_counts, self.scenario_colors["WFQ"], 'WFQ', self.scenario_linestyles["WFQ"])

        ax.axvline(x=baseline_warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2.5, label=f'Fine Warm-up ({baseline_warmup_duration:.2f}s)')
        ax.set_title('Confronto Evoluzione del Numero di Pod')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend(title='Scenario')
        self._save_plot(output_dir, "pod_history_comparison.png", fig)

    def _plot_queue_history(self, data: list[tuple[float, int]], scenario_name: str, output_dir: str, filename: str, color: str, linestyle: str, use_log_scale: bool, warmup_duration: float):
        """
        Genera il grafico dell'evoluzione della lunghezza della coda per un singolo scenario.

        Args:
            data (list[tuple[float, int]]): Lista di tuple (timestamp, lunghezza_coda).
            scenario_name (str): Nome dello scenario.
            output_dir (str): Directory dove salvare il grafico.
            filename (str): Nome del file del grafico.
            color (str): Colore della linea.
            linestyle (str): Stile della linea.
            use_log_scale (bool): Se True, usa la scala logaritmica per l'asse Y.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        ylabel = 'Numero Richieste in Coda'

        if data:
            # Sort by timestamp (x[0])
            sorted_data = sorted(data, key=lambda x: x[0])
            times, queue_lengths = zip(*sorted_data)

            if len(times) > 1:
                k_val = min(3, len(times) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(times), max(times), 500)
                    spl = make_interp_spline(times, queue_lengths, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, label='Lunghezza Coda (Smoothed)', alpha=0.7, linewidth=1.5, linestyle=linestyle)
                else:
                    ax.plot(times, queue_lengths, color=color, label='Lunghezza Coda', alpha=0.7, linewidth=1.5, linestyle='-') # Always solid
            else:
                ax.plot(times, queue_lengths, color=color, label='Lunghezza Coda', alpha=0.7, linewidth=1.5, linestyle='-') # Always solid

            # Calculate and plot steady-state mean if enough data exists after warmup
            steady_queue = [q for t, q in sorted_data if t >= warmup_duration]
            if steady_queue:
                dark_color = 'gray' # Default fallback
                if color == self.scenario_colors["Senza Priorità"]: dark_color = '#A60628' # Darker red
                elif color == self.scenario_colors["Con Priorità"]: dark_color = '#1E5894' # Darker blue
                elif color == self.scenario_colors["WFQ"]: dark_color = '#2E7C2E' # Darker green
                ax.axhline(np.mean(steady_queue), color=dark_color, linestyle='--', label=f'Media Steady-State: {np.mean(steady_queue):.2f}')

            ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2.5, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes, fontsize=12)

        if use_log_scale:
            ax.set_yscale('log')
            ylabel += ' (Scala Log)'
            # Set a positive lower bound for log scale, if data allows
            if data and min(q for _, q in data) > 0:
                ax.set_ylim(bottom=max(0.1, min(q for _, q in data) * 0.1))
            else:
                ax.set_ylim(bottom=0.1) # Default to 0.1 if no positive data

        ax.set_title(f'Evoluzione Lunghezza della Coda ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend()
        self._save_plot(output_dir, filename, fig)

    def plot_queue_history_analysis(self, warmup: dict, output_dir: str, use_log_scale: bool = True):
        """
        Genera grafici dell'evoluzione della lunghezza della coda per ogni scenario,
        inclusi un grafico di confronto tra gli scenari.

        Args:
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
            output_dir (str): Directory dove salvare i grafici.
            use_log_scale (bool): Se True, usa la scala logaritmica per l'asse Y.
        """
        baseline_warmup_duration = warmup.get("baseline", 0.0)
        priority_warmup_duration = warmup.get("priority", 0.0)
        wfq_warmup_duration = warmup.get("wfq", 0.0)

        times_b, queue_b = zip(*self.metrics.queue_length_history) if self.metrics.queue_length_history else ([],[])
        self._plot_queue_history(list(zip(times_b, queue_b)), "Senza Priorità", output_dir, f"queue_history_baseline{'_log' if use_log_scale else ''}.png", self.scenario_colors["Senza Priorità"], self.scenario_linestyles["Senza Priorità"], use_log_scale, warmup_duration=baseline_warmup_duration)
        self._plot_queue_history(list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths)), "Con Priorità", output_dir, f"queue_history_prio{'_log' if use_log_scale else ''}.png", self.scenario_colors["Con Priorità"], self.scenario_linestyles["Con Priorità"], use_log_scale, warmup_duration=priority_warmup_duration)
        self._plot_queue_history(list(zip(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths)), "WFQ", output_dir, f"queue_history_wfq{'_log' if use_log_scale else ''}.png", self.scenario_colors["WFQ"], self.scenario_linestyles["WFQ"], use_log_scale, warmup_duration=wfq_warmup_duration)

        print("Generazione grafico di CONFRONTO storico della Coda...")
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        ylabel = 'Numero Richieste in Coda'

        def plot_smoothed_queue_history(ax_obj: plt.Axes, data: list[tuple[float, int]], color: str, label: str, linestyle: str, warmup_duration: float):
            """Helper per plottare la storia della coda smoothed."""
            if data:
                # Sort by timestamp (x[0])
                sorted_data = sorted(data, key=lambda x: x[0])
                times, queue_lengths = zip(*sorted_data)

                if len(times) > 1:
                    k_val = min(3, len(times) - 1)
                    if k_val >= 1:
                        x_smooth = np.linspace(min(times), max(times), 500)
                        spl = make_interp_spline(times, queue_lengths, k=k_val)
                        y_smooth = spl(x_smooth)
                        ax_obj.plot(x_smooth, y_smooth, color=color, label=label, alpha=0.7, linewidth=1.5, linestyle=linestyle)
                    else:
                        ax_obj.plot(times, queue_lengths, color=color, label=label, alpha=0.7, linewidth=1.5, linestyle='-') # Always solid
                else:
                    ax_obj.plot(times, queue_lengths, color=color, label=label, alpha=0.7, linewidth=1.5, linestyle='-') # Always solid

                # Calculate and plot steady-state mean if enough data exists after warmup
                steady_queue = [q for t, q in sorted_data if t >= warmup_duration]
                if steady_queue:
                    dark_color = 'gray'
                    if color == self.scenario_colors["Senza Priorità"]: dark_color = '#A60628'
                    elif color == self.scenario_colors["Con Priorità"]: dark_color = '#1E5894'
                    elif color == self.scenario_colors["WFQ"]: dark_color = '#2E7C2E'
                    ax_obj.axhline(np.mean(steady_queue), color=dark_color, linestyle=':', label=f'Media Steady ({label}): {np.mean(steady_queue):.2f}')

        plot_smoothed_queue_history(ax, list(zip(times_b, queue_b)), self.scenario_colors["Senza Priorità"], 'Senza Priorità', self.scenario_linestyles["Senza Priorità"], baseline_warmup_duration)
        plot_smoothed_queue_history(ax, list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths)), self.scenario_colors["Con Priorità"], 'Con Priorità', self.scenario_linestyles["Con Priorità"], priority_warmup_duration)
        plot_smoothed_queue_history(ax, list(zip(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths)), self.scenario_colors["WFQ"], 'WFQ', self.scenario_linestyles["WFQ"], wfq_warmup_duration)

        if use_log_scale:
            ax.set_yscale('log')
            ylabel += ' (Scala Log)'
            # Set a positive lower bound for log scale, if data allows
            # Combine all queue data to find minimum positive for a consistent y-limit
            # MODIFIED: Corrected to use queue_b directly for baseline lengths
            all_queue_data = list(queue_b) + list(self.metrics_prio.queue_lengths) + list(self.metrics_wfq.queue_lengths)
            min_positive_queue = min([q for q in all_queue_data if q > 0], default=np.nan)
            if not np.isnan(min_positive_queue):
                ax.set_ylim(bottom=max(0.1, min_positive_queue * 0.1))
            else:
                ax.set_ylim(bottom=0.1)


        ax.set_title('Confronto Evoluzione della Lunghezza della Coda')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend(title='Scenario')
        self._save_plot(output_dir, f"queue_history_comparison{'_log' if use_log_scale else ''}.png", fig)

    def _plot_variance_trend(self, all_responses: list[tuple[float, float]], scenario_name: str, output_dir: str, filename: str, color: str, linestyle: str, window_size: int, warmup_duration: float):
        """
        [METODO REVISIONATO]
        Genera un grafico della deviazione standard mobile del tempo di risposta per un singolo scenario.
        Ora utilizza l'intera serie di dati per mostrare il comportamento sia durante il transitorio
        che a regime, con una linea verticale che indica la fine del warm-up.

        Args:
            all_responses (list[tuple[float, float]]): Lista COMPLETA di tuple (timestamp, tempo_risposta).
            scenario_name (str): Nome dello scenario.
            output_dir (str): Directory dove salvare il grafico.
            filename (str): Nome del file del grafico.
            color (str): Colore della linea.
            linestyle (str): Stile della linea.
            window_size (int): Dimensione della finestra per il calcolo della deviazione standard mobile.
            warmup_duration (float): Durata del periodo di warm-up in secondi.
        """
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")

        # NON filtriamo più i dati in anticipo. Usiamo la serie completa.
        if len(all_responses) > window_size:
            # Ordiniamo per sicurezza, anche se dovrebbero già esserlo
            sorted_responses = sorted(all_responses, key=lambda x: x[0])
            times, values = zip(*sorted_responses)

            # Calcoliamo la deviazione standard mobile sull'intera serie
            # Usiamo il metodo più efficiente di pandas.rolling per evitare loop lenti
            series = pd.Series(values)
            moving_std = series.rolling(window=window_size).std(ddof=1) # ddof=1 per varianza campionaria

            # Rimuoviamo i NaN iniziali dove la finestra non era piena
            valid_indices = ~moving_std.isnull()
            plot_times = np.array(times)[valid_indices]
            plot_values = moving_std[valid_indices]

            if len(plot_times) > 1:
                k_val = min(3, len(plot_times) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(plot_times), max(plot_times), 500)
                    spl = make_interp_spline(plot_times, plot_values, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax.plot(x_smooth, y_smooth, color=color, label='Dev. Std. Mobile (Smoothed)', alpha=0.8, linestyle=linestyle)
                else:
                    ax.plot(plot_times, plot_values, color=color, label='Dev. Std. Mobile', alpha=0.8, linestyle='-')
            else:
                ax.plot(plot_times, plot_values, color=color, label='Dev. Std. Mobile', alpha=0.8, linestyle='-')

            # Tracciamo la linea di WARM-UP sull'intero grafico
            ax.axvline(x=warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2.5, label=f'Fine Warm-up ({warmup_duration:.2f}s)')
        else:
            ax.text(0.5, 0.5, f"Dati insufficienti (necessari > {window_size} punti)", ha='center', va='center', transform=ax.transAxes, fontsize=12)

        ax.set_title(f'Stabilizzazione Varianza ({scenario_name}) - Finestra di {window_size}')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend()
        self._save_plot(output_dir, filename, fig)

    def plot_variance_trend_analysis(self, output_dir: str, warmup_durations: dict, window_size: int = 500):
        """
        Genera grafici della deviazione standard mobile del tempo di risposta per ogni scenario,
        inclusi un grafico di confronto tra gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup_durations (dict): Dizionario con le durate di warm-up per ogni scenario.
            window_size (int): Dimensione della finestra per il calcolo della deviazione standard mobile.
        """
        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        all_responses_wfq = self.metrics_wfq.get_all_response_times_with_timestamps()

        baseline_warmup = warmup_durations.get("baseline", 0.0)
        priority_warmup = warmup_durations.get("priority", 0.0)
        wfq_warmup = warmup_durations.get("wfq", 0.0)

        self._plot_variance_trend(all_responses_base, "Senza Priorità", output_dir, "variance_trend_baseline.png",
                                  self.scenario_colors["Senza Priorità"], self.scenario_linestyles["Senza Priorità"], window_size, warmup_duration=baseline_warmup)
        self._plot_variance_trend(all_responses_prio, "Con Priorità", output_dir, "variance_trend_prio.png",
                                  self.scenario_colors["Con Priorità"], self.scenario_linestyles["Con Priorità"], window_size, warmup_duration=priority_warmup)
        self._plot_variance_trend(all_responses_wfq, "WFQ", output_dir, "variance_trend_wfq.png",
                                  self.scenario_colors["WFQ"], self.scenario_linestyles["WFQ"], window_size, warmup_duration=wfq_warmup)

        print("Generazione grafico di CONFRONTO andamento della varianza...")
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")

        for responses, color, label, linestyle, warmup_duration_scenario in [
            (all_responses_base, self.scenario_colors["Senza Priorità"], 'Senza Priorità', self.scenario_linestyles["Senza Priorità"], baseline_warmup),
            (all_responses_prio, self.scenario_colors["Con Priorità"], 'Con Priorità', self.scenario_linestyles["Con Priorità"], priority_warmup),
            (all_responses_wfq, self.scenario_colors["WFQ"], 'WFQ', self.scenario_linestyles["WFQ"], wfq_warmup)
        ]:
            steady_data = [(t, v) for t, v in responses if t >= warmup_duration_scenario]
            if len(steady_data) > window_size:
                times, values = zip(*steady_data)

                moving_std = []
                for i in range(len(values) - window_size + 1):
                    window_values = values[i : i + window_size]
                    if len(window_values) > 1:
                        moving_std.append(np.std(window_values, ddof=1))
                    else:
                        moving_std.append(0)

                if moving_std:
                    plot_times = list(times[window_size - 1:])
                    if len(plot_times) > 1:
                        k_val = min(3, len(plot_times) - 1)
                        if k_val >= 1:
                            x_smooth = np.linspace(min(plot_times), max(plot_times), 500)
                            spl = make_interp_spline(plot_times, moving_std, k=k_val)
                            y_smooth = spl(x_smooth)
                            ax.plot(x_smooth, y_smooth, color=color, label=label, alpha=0.8, linestyle=linestyle)
                        else:
                            ax.plot(plot_times, moving_std, color=color, label=label, alpha=0.8, linestyle='-') # Always solid
                    else:
                        ax.plot(plot_times, moving_std, color=color, label=label, alpha=0.8, linestyle='-') # Always solid


        ax.axvline(x=baseline_warmup, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=2.5, label=f'Fine Warm-up ({baseline_warmup:.2f}s)')
        ax.set_title(f'Confronto Stabilizzazione Varianza (Finestra di {window_size})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', alpha=0.6)
        ax.legend(title='Scenario')
        self._save_plot(output_dir, "variance_trend_comparison.png", fig)


    # -------------- BATCH ---------------
    def _plot_batch_mean_queue_single(self, data: list[tuple[float, int]], scenario_name: str, warmup_duration: float, batch_size_b: int, num_batches_k: int, output_dir: str, color: str):
        """
        # REVISED: This method now includes a fallback mechanism for visualization purposes.
        # It first tries to use the batch parameters (b, k) from the main response time analysis.
        # If the queue length data is insufficient for these parameters (common in short runs),
        # it attempts to recalculate new, plot-specific b and k directly from the available
        # queue data. This allows a plot to be rendered for quick feedback during development,
        # even if the simulation is not long enough for full statistical convergence.

        Args:
            data (list[tuple[float, int]]): Dati grezzi di (timestamp, lunghezza_coda).
            scenario_name (str): Nome dello scenario.
            warmup_duration (float): Durata del periodo di warm-up da scartare.
            batch_size_b (int): Dimensione di ogni batch (numero di osservazioni) DALL'ANALISI PRINCIPALE.
            num_batches_k (int): Numero di batch DALL'ANALISI PRINCIPALE.
            output_dir (str): Directory di output.
            color (str): Colore per la linea dei dati.
        """
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
        ax.set_title(f'Evoluzione Medie per Batch della Coda - {scenario_name}')

        steady_values = [value for timestamp, value in data if timestamp >= warmup_duration]

        b_plot, k_plot = batch_size_b, num_batches_k
        is_valid = steady_values and b_plot > 0 and k_plot >= 2 and len(steady_values) >= b_plot * k_plot

        # --- SEZIONE DI FALLBACK ---
        if not is_valid:
            print(f"  WARNING ({scenario_name}): Parametri batch da analisi primaria (b={b_plot}, k={k_plot}) non utilizzabili per i dati della coda ({len(steady_values)} campioni).")
            print("  INFO: Tento un ricalcolo dei parametri solo per la visualizzazione di questo grafico.")

            # Tentiamo di ricalcolare b e k con un target meno stringente, solo per questo grafico.
            MIN_SAMPLES_FOR_FALLBACK = 30
            if len(steady_values) >= MIN_SAMPLES_FOR_FALLBACK:
                b_fallback, k_fallback, _ = compute_batch_size(steady_values, k_initial_target=20, threshold=0.4)

                if b_fallback is not None and k_fallback is not None and k_fallback >= 2 and len(steady_values) >= b_fallback * k_fallback:
                    print(f"  SUCCESS ({scenario_name}): Fallback riuscito. Uso b={b_fallback}, k={k_fallback} per il grafico.")
                    b_plot, k_plot = b_fallback, k_fallback
                    is_valid = True
                else:
                    print(f"  FAILURE ({scenario_name}): Fallback fallito. Dati della coda insufficienti anche per il ricalcolo.")
            else:
                print(f"  FAILURE ({scenario_name}): Non ci sono abbastanza campioni ({len(steady_values)} < {MIN_SAMPLES_FOR_FALLBACK}) per tentare un fallback.")

        # --- SEZIONE DI PLOTTING ---
        if is_valid:
            total_obs_for_batches = b_plot * k_plot
            batch_means_values = [np.mean(steady_values[i*b_plot : (i+1)*b_plot]) for i in range(k_plot)]
            grand_mean = np.mean(batch_means_values)

            batch_numbers = np.arange(1, k_plot + 1)
            ax.plot(batch_numbers, batch_means_values, marker=self.batch_mean_marker, color=color, linestyle='-', label=f'Media del Batch ({scenario_name})')
            ax.axhline(grand_mean, color='black', linestyle='--', linewidth=1.5, label=f'Media Globale: {grand_mean:.2f}')

            ax.set_xlabel('Numero del Batch')
            ax.set_ylabel('Lunghezza Media della Coda per Batch')
            ax.legend()
            ax.grid(True, which='both', alpha=0.6)
            ax.set_xlim(left=0, right=k_plot + 1)
            ax.set_ylim(bottom=0)
        else:
            # Mostra il messaggio di errore solo se anche il fallback è fallito.
            ax.text(0.5, 0.5, "Dati insufficienti per l'analisi dei batch", ha='center', va='center', transform=ax.transAxes)

        self._save_plot(output_dir, f"queue_batch_means_trend_{scenario_name.lower().replace(' ', '_')}.png", fig)

    def plot_batch_mean_queue_trend_analysis(self, warmup: dict, response_time_results: dict, output_dir: str):
        """
        # REVISED: This orchestration method is updated to call the new, simpler version of
        # _plot_batch_mean_queue_single. The comparison plot has also been simplified to
        # directly overlay the batch mean sequences from each scenario without any smoothing,
        # providing a direct and clear comparison of their stability and behavior across batches.

        Args:
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
            response_time_results (dict): Dizionario con i risultati del Batch Means, usato per
                                          recuperare la dimensione (b) e il numero (k) di batch.
            output_dir (str): Directory dove salvare i grafici.
        """
        print("Generazione grafici di andamento per le medie dei batch della coda...")
        baseline_warmup_duration = warmup.get("baseline", 0.0)
        priority_warmup_duration = warmup.get("priority", 0.0)
        wfq_warmup_duration = warmup.get("wfq", 0.0)

        # Estrai b e k dai risultati dell'analisi.
        res_base = response_time_results.get("baseline", {})
        res_prio = response_time_results.get("priority", {})
        res_wfq = response_time_results.get("wfq", {})

        b_base, k_base = res_base.get("batch_size", 0), res_base.get("num_batches", 0)
        b_prio, k_prio = res_prio.get("batch_size", 0), res_prio.get("num_batches", 0)
        b_wfq, k_wfq = res_wfq.get("batch_size", 0), res_wfq.get("num_batches", 0)

        # Dati della lunghezza della coda
        data_base = self.metrics.queue_length_history if self.metrics.queue_length_history else []
        data_prio = list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths)) if self.metrics_prio.queue_lengths else []
        data_wfq = list(zip(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths)) if self.metrics_wfq.queue_lengths else []

        # Genera i grafici individuali
        self._plot_batch_mean_queue_single(data_base, "Senza Priorità", baseline_warmup_duration, b_base, k_base, output_dir, self.scenario_colors["Senza Priorità"])
        self._plot_batch_mean_queue_single(data_prio, "Con Priorità", priority_warmup_duration, b_prio, k_prio, output_dir, self.scenario_colors["Con Priorità"])
        self._plot_batch_mean_queue_single(data_wfq, "WFQ", wfq_warmup_duration, b_wfq, k_wfq, output_dir, self.scenario_colors["WFQ"])

        # Genera il grafico di confronto
        print("Generazione grafico CONFRONTO andamento delle medie dei batch della coda...")
        fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")

        scenarios = {
            "Senza Priorità": (data_base, baseline_warmup_duration, b_base, k_base, self.scenario_colors["Senza Priorità"]),
            "Con Priorità": (data_prio, priority_warmup_duration, b_prio, k_prio, self.scenario_colors["Con Priorità"]),
            "WFQ": (data_wfq, wfq_warmup_duration, b_wfq, k_wfq, self.scenario_colors["WFQ"])
        }

        max_k = 0
        for name, (data, warmup_d, b, k, color) in scenarios.items():
            steady_values = [val for ts, val in data if ts >= warmup_d]
            if not steady_values or b <= 0 or k < 2 or len(steady_values) < b * k:
                continue

            total_obs = b * k
            batch_means_vals = [np.mean(steady_values[i*b : (i+1)*b]) for i in range(k)]
            batch_nums = np.arange(1, k + 1)
            ax.plot(batch_nums, batch_means_vals, marker=self.batch_mean_marker, linestyle='-', color=color, label=name, markersize=4)
            max_k = max(max_k, k)

        ax.set_title('Confronto Evoluzione delle Medie per Batch della Coda')
        ax.set_xlabel('Numero del Batch')
        ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(title='Scenario')
        ax.grid(True, which='both', alpha=0.6)
        if max_k > 0:
            ax.set_xlim(left=0, right=max_k + 1)
        ax.set_ylim(bottom=0)

        self._save_plot(output_dir, "queue_batch_means_trend_comparison.png", fig)

    def plot_throughput_analysis(self, warmup: dict, output_dir: str):
        """
        Genera un grafico a barre di confronto per il throughput (per tipo di richiesta)
        per tutti gli scenari, con intervalli di confidenza al 95%.

        Args:
            warmup (dict): Dizionario con le durate di warm-up per ogni scenario.
            output_dir (str): Directory dove salvare i grafici.
        """
        print("Generazione grafico di CONFRONTO del throughput per tipo di richiesta...")

        fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
        fig.suptitle("Confronto Throughput (Steady State) per Tipo con IC al 95%")

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        plot_data = []

        baseline_warmup = warmup.get("baseline", 0.0)
        priority_warmup = warmup.get("priority", 0.0)
        wfq_warmup = warmup.get("wfq", 0.0)

        # Helper function to get completion timestamps for a given metric object and request type
        def get_completion_timestamps_for_type(metrics_obj, req_type: RequestType, is_prio_or_wfq: bool) -> list[float]:
            """
            Recupera i timestamp di completamento per un tipo di richiesta specifico.
            """
            if is_prio_or_wfq: # For MetricsWithPriority or WFQ metrics objects
                return sorted(metrics_obj.completion_timestamps_by_req_type.get(req_type, []))
            else: # For base Metrics object
                # For baseline, response_times_history stores (timestamp, value)
                return sorted([ts for ts, _ in metrics_obj.response_times_history.get(req_type, [])])

        # Create temporary analyzers to re-calculate per-type throughput, as `main.py`
        # currently provides only OVERALL throughput CI via `throughput_results`.
        # This allows detailed per-type CI visualization.
        temp_analyzer_baseline = SteadyStateAnalyzer(self.metrics, self.config)
        temp_analyzer_prio = SteadyStateAnalyzer(self.metrics_prio, self.config)
        temp_analyzer_wfq = SteadyStateAnalyzer(self.metrics_wfq, self.config)


        for req_type in all_req_types:
            # Baseline
            timestamps_b = get_completion_timestamps_for_type(self.metrics, req_type, False)
            if results_b := temp_analyzer_baseline.calculate_throughput_ci(timestamps_b, baseline_warmup):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Throughput (req/s)': float(results_b['mean']),
                    'Errore': float(results_b['half_width']),
                    'Scenario': 'Senza Priorità'
                })

            # Priorità
            timestamps_p = get_completion_timestamps_for_type(self.metrics_prio, req_type, True)
            if results_p := temp_analyzer_prio.calculate_throughput_ci(timestamps_p, priority_warmup):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Throughput (req/s)': float(results_p['mean']),
                    'Errore': float(results_p['half_width']),
                    'Scenario': 'Con Priorità'
                })

            # WFQ
            timestamps_wfq = get_completion_timestamps_for_type(self.metrics_wfq, req_type, True)
            if results_wfq := temp_analyzer_wfq.calculate_throughput_ci(timestamps_wfq, wfq_warmup):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Throughput (req/s)': float(results_wfq['mean']),
                    'Errore': float(results_wfq['half_width']),
                    'Scenario': 'WFQ'
                })


        if not plot_data:
            print("Dati insufficienti per il grafico di confronto del throughput.")
            ax.text(0.5, 0.5, "Nessun dato disponibile per il confronto di throughput.", ha='center', va='center', fontsize=12)
            ax.set_title("Confronto Throughput (Nessun Dato)")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            self._save_plot(output_dir, "throughput_comparison.png", fig)
            return

        df = pd.DataFrame(plot_data)
        df["Categoria"] = df["Categoria"].astype("category").cat.reorder_categories(category_names, ordered=True)
        df["Scenario"] = df["Scenario"].astype("category").cat.reorder_categories(list(self.scenario_colors.keys()), ordered=True)

        bars = sns.barplot(
            data=df,
            x='Categoria', y='Throughput (req/s)',
            hue='Scenario',
            palette=list(self.scenario_colors.values()),
            ax=ax
        )

        num_categories = len(category_names)

        bar_width_per_scenario = 0.2
        if ax.containers:
            for container in ax.containers:
                if container.patches:
                    bar_width_per_scenario = container.patches[0].get_width()
                    break

        total_group_width = bar_width_per_scenario * len(self.scenario_colors)
        offset_center = -total_group_width / 2 + bar_width_per_scenario / 2

        for i, scenario_name in enumerate(list(self.scenario_colors.keys())):
            offset = offset_center + i * bar_width_per_scenario
            subset = df[df["Scenario"] == scenario_name].set_index("Categoria").reindex(category_names)

            if subset["Throughput (req/s)"].isnull().all():
                continue

            x_coords_for_error = np.arange(num_categories) + offset
            y_coords = subset["Throughput (req/s)"].fillna(0).values
            errors = subset["Errore"].fillna(0).values

            ax.errorbar(x_coords_for_error, y_coords, yerr=errors, fmt="none", c=self.ci_error_bar_color, capsize=5, elinewidth=1.2)

            for cat_idx, category_name in enumerate(category_names):
                row = subset.loc[category_name]
                if pd.notna(row["Throughput (req/s)"]):
                    mean_val, error_val = row["Throughput (req/s)"], row["Errore"]
                    upper_bound = mean_val + error_val
                    # Throughput often has less precision displayed for CI
                    lower_bound = max(0, mean_val - error_val)
                    ci_text = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    ax.annotate(
                        ci_text,
                        xy=(x_coords_for_error[cat_idx], upper_bound),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="black"
                    )

        # Annotate overall values for each bar
        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0: # Only annotate positive throughputs
                ax.annotate(f'{height:.2f}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='center',
                            fontsize=9, color='black',
                            xytext=(0, 5), textcoords='offset points')


        ax.set_xlabel("Tipo di Richiesta")
        ax.set_ylabel("Throughput Medio (richieste/s)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title='Scenario', loc='upper left')
        ax.grid(True, axis='y', alpha=0.5)

        # Adjust y-limits for annotations
        current_y_lim_top = ax.get_ylim()[1]
        ax.set_ylim(top=current_y_lim_top * 1.25)


        # Delta vs Baseline for Priority and WFQ
        delta_data = []
        for cat_name in category_names:
            base_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Senza Priorità')]
            prio_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Con Priorità')]
            wfq_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'WFQ')]

            # MODIFIED: Correctly access the scalar value from the DataFrame row
            base_throughput = base_row['Throughput (req/s)'].item() if not base_row.empty else 0
            prio_throughput = prio_row['Throughput (req/s)'].item() if not prio_row.empty else 0
            wfq_throughput = wfq_row['Throughput (req/s)'].item() if not wfq_row.empty else 0

            # Delta vs Baseline for Priority
            delta_prio_perc = 0
            if base_throughput > 0:
                delta_prio_perc = ((prio_throughput - base_throughput) / base_throughput) * 100

            # Delta vs Baseline for WFQ
            delta_wfq_perc = 0
            if base_throughput > 0:
                delta_wfq_perc = ((wfq_throughput - base_throughput) / base_throughput) * 100

            delta_data.append({
                'Categoria': cat_name,
                'Delta Prio': delta_prio_perc,
                'Delta WFQ': delta_wfq_perc
            })

        df_deltas = pd.DataFrame(delta_data).set_index('Categoria').reindex(category_names)

        # Positioning for delta annotations
        # Using a fixed y-position relative to the top of the chart for clarity
        y_top_for_deltas = ax.get_ylim()[1] * 0.95 # Get actual top y-limit after bars and CI annotations

        for i, cat_name in enumerate(category_names):
            prio_delta = df_deltas.loc[cat_name]['Delta Prio']
            wfq_delta = df_deltas.loc[cat_name]['Delta WFQ']

            # X position for the group of bars for this category (center of the group)
            # The bars are at positions i-1, i, i+1 for a group of 3.
            # Center of the group is 'i'. Offset by -1.5*bar_width and +1.5*bar_width
            x_pos_group_center = i

            # Annotate Priority Delta
            sign_prio = '+' if prio_delta >= 0 else ''
            ax.text(x_pos_group_center - (bar_width_per_scenario * 1.5), y_top_for_deltas,
                    f'Prio Δ: {sign_prio}{prio_delta:.1f}%',
                    ha='center', va='center',
                    color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor=self.scenario_colors["Con Priorità"], alpha=0.9),
                    clip_on=True) # Ensure text is clipped if outside axis limits

            # Annotate WFQ Delta
            sign_wfq = '+' if wfq_delta >= 0 else ''
            ax.text(x_pos_group_center + (bar_width_per_scenario * 1.5), y_top_for_deltas,
                    f'WFQ Δ: {sign_wfq}{wfq_delta:.1f}%',
                    ha='center', va='center',
                    color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor=self.scenario_colors["WFQ"], alpha=0.9),
                    clip_on=True)

        ax.set_ylim(top=ax.get_ylim()[1] * 1.1) # Ensure enough space for delta annotations

        self._save_plot(output_dir, "throughput_comparison.png", fig)

    def plot_times_by_request_type_grid(self, output_dir: str, warmup_durations: dict):
        """
        Genera una griglia di grafici dettagliati per ogni tipo di richiesta,
        mostrando l'andamento dei tempi di risposta e attesa cumulativi per tutti gli scenari.

        Args:
            output_dir (str): Directory dove salvare i grafici.
            warmup_durations (dict): Dizionario con le durate di warm-up per ogni scenario.
        """
        print("Generazione griglia di confronto per tipo di richiesta...")
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        ncols = 3
        nrows = int(np.ceil(len(all_req_types) / ncols))

        # Adjust figsize dynamically based on number of rows
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6.5, max(4, nrows) * 5.5), sharex=True, sharey=True, layout="constrained")
        axes = axes.flatten() # Flatten for easier iteration

        baseline_warmup_duration = warmup_durations.get("baseline", 0.0)
        priority_warmup_duration = warmup_durations.get("priority", 0.0)
        wfq_warmup_duration = warmup_durations.get("wfq", 0.0)

        def plot_smoothed_line(ax_obj: plt.Axes, times: list[float], values: list[float], color: str, linestyle: str, label: str, plot_raw_if_short: bool = False):
            """Helper per plottare una linea smoothed o raw."""
            if not values or not times or len(times) == 0:
                return

            cumulative_means = np.cumsum(values) / np.arange(1, len(values) + 1)

            if len(times) > 1:
                k_val = min(3, len(times) - 1)
                if k_val >= 1:
                    x_smooth = np.linspace(min(times), max(times), 200) # Reduced points for faster plotting
                    spl = make_interp_spline(times, cumulative_means, k=k_val)
                    y_smooth = spl(x_smooth)
                    ax_obj.plot(x_smooth, y_smooth, color=color, linestyle=linestyle, label=label, linewidth=1.5)
                elif plot_raw_if_short: # Fallback for very few points, if explicitly allowed
                    ax_obj.plot(times, cumulative_means, color=color, linestyle='-', label=label, linewidth=1.5) # Always solid
            elif plot_raw_if_short: # Fallback for single point, if explicitly allowed
                ax_obj.plot(times, cumulative_means, color=color, label=label, linestyle='-', linewidth=1.5) # Always solid


        for i, req_type in enumerate(all_req_types):
            ax = axes[i]

            # --- Baseline (Senza Priorità) ---
            if resp_b_raw := self.metrics.response_times_history.get(req_type, []):
                resp_b = sorted(resp_b_raw, key=lambda x: x[0])
                times, values = zip(*resp_b)
                plot_smoothed_line(ax, list(times), list(values), self.scenario_colors["Senza Priorità"], self.response_line_style, 'Risposta (Baseline)', plot_raw_if_short=True)
            if wait_b_raw := self.metrics.wait_times_history.get(req_type, []):
                wait_b = sorted(wait_b_raw, key=lambda x: x[0])
                times, values = zip(*wait_b)
                plot_smoothed_line(ax, list(times), list(values), self.scenario_colors["Senza Priorità"], self.wait_line_style, 'Attesa (Baseline)', plot_raw_if_short=True)

            # --- Priority (Con Priorità) ---
            times_p = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            values_rp = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            if times_p and values_rp and len(times_p) == len(values_rp):
                times_s, values_s = zip(*sorted(zip(times_p, values_rp), key=lambda x: x[0]))
                plot_smoothed_line(ax, list(times_s), list(values_s), self.scenario_colors["Con Priorità"], self.response_line_style, 'Risposta (Priorità)', plot_raw_if_short=True)
            values_wp = self.metrics_prio.wait_times_by_req_type.get(req_type, [])
            if times_p and values_wp and len(times_p) == len(values_wp):
                times_s, values_s = zip(*sorted(zip(times_p, values_wp), key=lambda x: x[0]))
                plot_smoothed_line(ax, list(times_s), list(values_s), self.scenario_colors["Con Priorità"], self.wait_line_style, 'Attesa (Priorità)', plot_raw_if_short=True)

            # --- WFQ ---
            times_wfq_comp = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, []) # Completion timestamps
            values_rwfq = self.metrics_wfq.response_times_by_req_type.get(req_type, [])
            if times_wfq_comp and values_rwfq and len(times_wfq_comp) == len(values_rwfq):
                times_s, values_s = zip(*sorted(zip(times_wfq_comp, values_rwfq), key=lambda x: x[0]))
                plot_smoothed_line(ax, list(times_s), list(values_s), self.scenario_colors["WFQ"], self.response_line_style, 'Risposta (WFQ)', plot_raw_if_short=True)
            values_wwfq = self.metrics_wfq.wait_times_by_req_type.get(req_type, [])
            if times_wfq_comp and values_wwfq and len(times_wfq_comp) == len(values_wwfq):
                times_s, values_s = zip(*sorted(zip(times_wfq_comp, values_wwfq), key=lambda x: x[0]))
                plot_smoothed_line(ax, list(times_s), list(values_s), self.scenario_colors["WFQ"], self.wait_line_style, 'Attesa (WFQ)', plot_raw_if_short=True)

            # Add vertical line for warmup duration (using baseline as reference for x-axis consistency)
            ax.axvline(x=baseline_warmup_duration, color=self.warmup_line_color, linestyle=self.warmup_line_style, linewidth=1.5, label=f'Warm-up End ({baseline_warmup_duration:.2f}s)')

            ax.set_title(req_type.name.replace('_', ' ').title())
            ax.grid(True, alpha=0.6)
            ax.legend(loc='upper left', fontsize=8) # Smaller legend font for grid plots

        # Hide any unused subplots
        if all_req_types:
            for j in range(len(all_req_types), len(axes)):
                axes[j].set_visible(False)

        fig.supxlabel('Tempo di Simulazione (s)', y=0.02)
        fig.supylabel('Tempo Medio Cumulativo (s)', x=0.02)
        fig.suptitle('Confronto Dettagliato Tempi di Risposta e Attesa per Tipo di Richiesta (Steady State)', y=1.0)
        self._save_plot(output_dir, "times_grid_comparison.png", fig)

    # ==============================================================================
    # NUOVA SEZIONE: PLOTTING CONVERGENZA BATCH MEANS (STILE IMMAGINE)
    # ==============================================================================
    def _plot_batch_means_convergence_single(self, steady_values: list[float], results: dict, scenario_name: str, output_dir: str, metric_name: str):
        """
        [NUOVO METODO]
        Genera un grafico che mostra la convergenza della media cumulativa dei batch e
        del relativo intervallo di confidenza, come nell'immagine di esempio.

        Args:
            steady_values (list[float]): La serie di dati già filtrata dal warm-up.
            results (dict): Il dizionario di risultati prodotto da `steady_state_analysis`.
            scenario_name (str): Nome dello scenario per il titolo.
            output_dir (str): Directory di output.
            metric_name (str): Nome della metrica (es. "Tempo di Risposta").
        """
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        ax.set_title(f'{scenario_name} - Convergenza Stima {metric_name}')

        if not results or len(steady_values) < 2:
            ax.text(0.5, 0.5, "Dati insufficienti per l'analisi", ha='center', va='center', transform=ax.transAxes)
            self._save_plot(output_dir, f"batch_means_convergence_{scenario_name.lower().replace(' ', '_')}.png", fig)
            return

        b = results['batch_size']
        k = results['num_batches']
        confidence = results['confidence_level']

        if not (b > 0 and k >= 2 and len(steady_values) >= b * k):
            ax.text(0.5, 0.5, f"Parametri batch non validi (b={b}, k={k}) per {len(steady_values)} campioni", ha='center', va='center', transform=ax.transAxes)
            self._save_plot(output_dir, f"batch_means_convergence_{scenario_name.lower().replace(' ', '_')}.png", fig)
            return

        batch_means_values = [np.mean(steady_values[i*b : (i+1)*b]) for i in range(k)]

        # Liste per storicizzare le stime cumulative
        cumulative_means_history = []
        ci_lower_history = []
        ci_upper_history = []
        # L'asse x: partiamo da 2 batch fino a k
        batch_indices = np.arange(2, k + 1)

        # Calcoliamo la media e il CI cumulativi
        for i in batch_indices:
            current_batch_means = batch_means_values[:i]
            mean = np.mean(current_batch_means)
            # Varianza campionaria (ddof=1)
            s2 = np.var(current_batch_means, ddof=1)

            # Gradi di libertà per la t-Student
            dof = i - 1
            # Valore critico t
            t_val = t.ppf((1 + confidence) / 2, df=dof)

            # Semi-ampiezza dell'intervallo
            half_width = t_val * np.sqrt(s2 / i)

            cumulative_means_history.append(mean)
            ci_lower_history.append(mean - half_width)
            ci_upper_history.append(mean + half_width)

        # Colori come da richiesta
        plot_color = 'skyblue'
        fill_color = 'skyblue'

        # Plottiamo la linea della media cumulativa
        ax.plot(batch_indices, cumulative_means_history, color=plot_color, linewidth=2,
                label='Media Cumulativa dei Batch')

        # Plottiamo l'area di confidenza
        ax.fill_between(batch_indices, ci_lower_history, ci_upper_history, color=fill_color, alpha=0.3,
                        label=f'Intervallo di Confidenza al {confidence:.0%}')

        ax.set_xlabel('Numero di Batch Inclusi nel Calcolo')
        ax.set_ylabel(f'{metric_name} Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Aggiungi un testo con il risultato finale
        final_mean = cumulative_means_history[-1]
        final_ci_lower = ci_lower_history[-1]
        final_ci_upper = ci_upper_history[-1]
        final_text = (
            f"Stima Finale (con k={k} batch):\n"
            f"  - Media: {final_mean:.4f}\n"
            f"  - IC: [{final_ci_lower:.4f}, {final_ci_upper:.4f}]"
        )
        ax.text(0.98, 0.98, final_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


        self._save_plot(output_dir, f"batch_means_convergence_{scenario_name.lower().replace(' ', '_')}.png", fig)

    def plot_batch_means_convergence_analysis(self, warmup: dict, response_time_results: dict, output_dir: str):
        """
        [METODO ORCHESTRATORE]
        Orchestra la generazione dei grafici di convergenza del Batch Means per i tempi di risposta.

        Args:
            warmup (dict): Dizionario con le durate di warm-up.
            response_time_results (dict): Risultati finali dell'analisi Batch Means.
            output_dir (str): Directory di output.
        """
        print("Generazione grafici di convergenza per le stime Batch Means...")

        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        all_responses_wfq = self.metrics_wfq.get_all_response_times_with_timestamps()

        steady_values_base = [v for t, v in all_responses_base if t >= warmup.get("baseline", 0)]
        steady_values_prio = [v for t, v in all_responses_prio if t >= warmup.get("priority", 0)]
        steady_values_wfq = [v for t, v in all_responses_wfq if t >= warmup.get("wfq", 0)]

        # Genera i grafici individuali per ogni scenario
        self._plot_batch_means_convergence_single(steady_values_base, response_time_results.get("baseline"), "Senza Priorità", output_dir, "Tempo di Risposta")
        self._plot_batch_means_convergence_single(steady_values_prio, response_time_results.get("priority"), "Con Priorità", output_dir, "Tempo di Risposta")
        self._plot_batch_means_convergence_single(steady_values_wfq, response_time_results.get("wfq"), "WFQ", output_dir, "Tempo di Risposta")

        # Genera il grafico di confronto delle sole linee delle medie cumulative
        print("Generazione grafico di CONFRONTO della convergenza delle stime...")
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        ax.set_title('Confronto Convergenza delle Stime del Tempo di Risposta Medio')

        scenarios_data = {
            "Senza Priorità": (steady_values_base, response_time_results.get("baseline"), self.scenario_colors["Senza Priorità"]),
            "Con Priorità": (steady_values_prio, response_time_results.get("priority"), self.scenario_colors["Con Priorità"]),
            "WFQ": (steady_values_wfq, response_time_results.get("wfq"), self.scenario_colors["WFQ"])
        }

        max_k = 0
        for name, (values, results, color) in scenarios_data.items():
            if not results: continue

            b, k = results['batch_size'], results['num_batches']
            if not (b > 0 and k >= 2 and len(values) >= b * k): continue

            batch_means_values = [np.mean(values[i*b : (i+1)*b]) for i in range(k)]
            batch_indices = np.arange(2, k + 1)
            cumulative_means_history = [np.mean(batch_means_values[:i]) for i in batch_indices]

            ax.plot(batch_indices, cumulative_means_history, color=color, linewidth=2, label=name)
            max_k = max(max_k, k)

        ax.set_xlabel('Numero di Batch Inclusi nel Calcolo')
        ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend(title='Scenario')
        if max_k > 0:
            ax.set_xlim(left=0, right=max_k + 1)
        ax.set_ylim(bottom=0)

        self._save_plot(output_dir, "batch_means_convergence_comparison.png", fig)