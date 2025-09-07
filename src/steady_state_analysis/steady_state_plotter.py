# In src/steady_state_analysis/steady_state_plotter.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.acs import batch_means, compute_batch_size
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.config import RequestType
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.utils.welford import Welford

plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'        # Rende bianco lo sfondo dell'intera figura.

plt.rcParams['savefig.facecolor'] = 'white'       # Assicura che il colore di sfondo della figura salvata sia bianco.
plt.rcParams['savefig.transparent'] = False

class SteadyStatePlotter:
    def __init__(self, metrics: Metrics, metrics_prio: MetricsWithPriority, metrics_wfq: MetricsWithPriority,config, use_log_scale_infinite=True):
        self.metrics = metrics
        self.metrics_prio = metrics_prio
        self.metrics_wfq = metrics_wfq
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

    def generate_steady_state_report(self, analyzer_baseline: SteadyStateAnalyzer, analyzer_prio: SteadyStateAnalyzer, analyzer_wfq: SteadyStateAnalyzer, warmup: dict, batches: dict, output_dir="plots/steady_state"):
        print(f"\n--- INIZIO Generazione Report Completo in '{output_dir}' ---")
        os.makedirs(output_dir, exist_ok=True)

        # Extract specific warmup durations for clarity
        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        print("\n--- [SEZIONE 1/4] Analisi Performance a Regime ---")
        self.plot_steady_state_times_by_type(analyzer_baseline, analyzer_prio, analyzer_wfq, warmup, batches, os.path.join(output_dir, "times_analysis")) # MODIFIED
        self.plot_throughput_analysis(analyzer_baseline, analyzer_prio, analyzer_wfq, warmup, batches, os.path.join(output_dir, "throughput_analysis")) # MODIFIED
        self.plot_steady_state_loss(warmup, os.path.join(output_dir, "loss_analysis"))

        print("\n--- [SEZIONE 2/4] Analisi Comportamento del Sistema ---")
        self.plot_pod_history_analysis(os.path.join(output_dir, "pod_history_analysis"), warmup=warmup)
        self.plot_queue_history_analysis(warmup, os.path.join(output_dir, "queue_history_analysis"), use_log_scale=True)
        self.plot_wait_time_trend_analysis(os.path.join(output_dir, "wait_time_trend_analysis"), warmup=warmup)

        print("\n--- [SEZIONE 3/4] Analisi del Transitorio e della Stabilità ---")
        self.plot_convergence_analysis_overall(os.path.join(output_dir, "convergence_overall_analysis"), warmup_durations=warmup)
        self.plot_convergence_analysis_by_type(os.path.join(output_dir, "convergence_by_type_analysis"), warmup_durations=warmup)
        self.plot_variance_trend_analysis(os.path.join(output_dir, "variance_trend_analysis"), warmup_durations=warmup)
        self.plot_batch_mean_queue_trend_analysis(warmup, batches, os.path.join(output_dir, "queue_batch_means_analysis"))

        print("\n--- [SEZIONE 4/4] Analisi Dettagliate per Tipo di Richiesta ---")
        self.plot_times_by_request_type_grid(os.path.join(output_dir, "detailed_grid_analysis"), warmup_durations=warmup)

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

    def _plot_single_scenario_times(self, analyzer, metrics, scenario_name, warmup_duration, output_dir, file_suffix, color_palette):
        is_prio = isinstance(metrics, MetricsWithPriority)
        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace("_", " ").title() for req in all_req_types]

        for metric_name, ax in zip(["response", "wait"], axes):
            plot_data = []

            for req_type in all_req_types:
                if is_prio:
                    values = (
                        metrics.response_times_by_req_type.get(req_type, [])
                        if metric_name == "response"
                        else metrics.wait_times_by_req_type.get(req_type, [])
                    )
                    timestamps = metrics.completion_timestamps_by_req_type.get(req_type, [])
                    if len(values) == len(timestamps):
                        raw_data = sorted(zip(timestamps, values), key=lambda x: x[0])
                    else:
                        raw_data = []
                else:
                    raw_data = (
                        metrics.response_times_history.get(req_type, [])
                        if metric_name == "response"
                        else metrics.wait_times_history.get(req_type, [])
                    )

                if not raw_data:
                    continue

                values = [v for t, v in raw_data if t >= warmup_duration]

                if len(values) < 30:
                    continue

                b, k, _ = compute_batch_size(values, threshold=self.config.BATCH_THRESHOLD)
                if b is None or k is None or b * k == 0:
                    continue

                batch_means_res = batch_means(values, b, k, confidence=self.config.CONFIDENCE_LEVEL)
                mean, ci_interval, half_width = batch_means_res['mean'], batch_means_res['ci'], batch_means_res['half_width']

                plot_data.append({
                    "Categoria": req_type.name.replace("_", " ").title(),
                    "Tempo Medio (s)": mean,
                    "Errore": half_width,
                })

            if not plot_data:
                ax.text(0.5, 0.5, "Nessun dato valido per il plotting.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio (Nessun Dato)", fontsize=16)
                plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
                continue


            df = pd.DataFrame(plot_data)

            sns.barplot(
                data=df,
                x="Categoria",
                y="Tempo Medio (s)",
                order=category_names,
                color=color_palette,
                ax=ax,
            )

            x_positions = np.arange(len(category_names))
            subset = df.set_index("Categoria").reindex(category_names)

            if not subset.empty:
                y_coords = subset["Tempo Medio (s)"].fillna(0)
                errors = subset["Errore"].fillna(0)
                ax.errorbar(
                    x_positions, y_coords, yerr=errors,
                    fmt="none", c="black", capsize=5, elinewidth=1.2
                )

                if ax.containers:
                    ax.bar_label(
                        ax.containers[0], fmt="%.3f", padding=3,
                        fontsize=8, weight="bold", color="black"
                    )

                for k_item, row in subset.iterrows():
                    if pd.notna(row["Tempo Medio (s)"]):
                        try:
                            cat_index = category_names.index(k_item)
                            mean_val, error_val = row["Tempo Medio (s)"], row["Errore"]
                            upper_bound = mean_val + error_val
                            ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"
                            ax.annotate(
                                ci_text,
                                xy=(x_positions[cat_index], upper_bound),
                                xytext=(0, 4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color="black"
                            )
                        except ValueError:
                            continue

            ax.set_title(
                f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio",
                fontsize=16
            )
            ax.set_ylabel("Tempo Medio (s)", fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom=bottom, top=top * 1.03)

        fig.suptitle(f"Tempi Medi (Steady State) - {scenario_name}", fontsize=20, fontweight="bold")
        plt.tight_layout()
        fig.subplots_adjust(top=0.90, bottom=0.20, left=0.07, right=0.98)
        self._save_plot(output_dir, f"times{file_suffix}.png", fig)

    def plot_steady_state_times_by_type(self, analyzer_baseline, analyzer_prio, analyzer_wfq, warmup, batches, output_dir): # MODIFIED: added analyzer_wfq
        print("Generazione grafico di CONFRONTO per tempi per tipo di richiesta...")

        fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace("_", " ").title() for req in all_req_types]

        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        for metric_name, ax in zip(["response", "wait"], axes):
            if self.use_log_scale_infinite:
                ax.set_yscale("log")

            plot_data = []

            for req_type in all_req_types:
                # -------------------------
                # Baseline (senza priorità)
                # -------------------------
                vals_b = (
                    self.metrics.response_times_history.get(req_type, [])
                    if metric_name == "response"
                    else self.metrics.wait_times_history.get(req_type, [])
                )
                values_b = [v for t, v in vals_b if t >= baseline_warmup_duration]

                if len(values_b) >= 2:
                    b_req_type_base, k_req_type_base, _ = compute_batch_size(values_b, threshold=self.config.BATCH_THRESHOLD)
                    if b_req_type_base and k_req_type_base and b_req_type_base * k_req_type_base <= len(values_b):
                        res_b = batch_means(values_b, b_req_type_base, k_req_type_base, confidence=self.config.CONFIDENCE_LEVEL)
                        plot_data.append({
                            "Categoria": req_type.name.replace("_", " ").title(),
                            "Tempo Medio (s)": float(res_b["mean"]),
                            "Errore": float(res_b["half_width"]),
                            "Scenario": "Senza Priorità",
                        })

                # -------------------------
                # Con priorità
                # -------------------------
                vals_p = (
                    self.metrics_prio.response_times_by_req_type.get(req_type, [])
                    if metric_name == "response"
                    else self.metrics_prio.wait_times_by_req_type.get(req_type, [])
                )
                ts_p = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
                if len(vals_p) == len(ts_p):
                    raw_p = sorted(zip(ts_p, vals_p), key=lambda x: x[0])
                    values_p = [v for t, v in raw_p if t >= priority_warmup_duration]

                    if len(values_p) >= 2:
                        b_req_type_prio, k_req_type_prio, _ = compute_batch_size(values_p, threshold=self.config.BATCH_THRESHOLD)
                        if b_req_type_prio and k_req_type_prio and b_req_type_prio * k_req_type_prio <= len(values_p):
                            res_p = batch_means(values_p, b_req_type_prio, k_req_type_prio, confidence=self.config.CONFIDENCE_LEVEL)
                            plot_data.append({
                                "Categoria": req_type.name.replace("_", " ").title(),
                                "Tempo Medio (s)": float(res_p["mean"]),
                                "Errore": float(res_p["half_width"]),
                                "Scenario": "Con Priorità",
                            })

                # -------------------------
                # Con WFQ (Weighted Fair Queuing) # ADDED
                # -------------------------
                vals_wfq = (
                    self.metrics_wfq.response_times_by_req_type.get(req_type, [])
                    if metric_name == "response"
                    else self.metrics_wfq.wait_times_by_req_type.get(req_type, [])
                )
                ts_wfq = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
                if len(vals_wfq) == len(ts_wfq):
                    raw_wfq = sorted(zip(ts_wfq, vals_wfq), key=lambda x: x[0])
                    values_wfq = [v for t, v in raw_wfq if t >= wfq_warmup_duration]

                    if len(values_wfq) >= 2:
                        b_req_type_wfq, k_req_type_wfq, _ = compute_batch_size(values_wfq, threshold=self.config.BATCH_THRESHOLD)
                        if b_req_type_wfq and k_req_type_wfq and b_req_type_wfq * k_req_type_wfq <= len(values_wfq):
                            res_wfq = batch_means(values_wfq, b_req_type_wfq, k_req_type_wfq, confidence=self.config.CONFIDENCE_LEVEL)
                            plot_data.append({
                                "Categoria": req_type.name.replace("_", " ").title(),
                                "Tempo Medio (s)": float(res_wfq["mean"]),
                                "Errore": float(res_wfq["half_width"]),
                                "Scenario": "WFQ",
                            })

            if not plot_data:
                ax.text(0.5, 0.5, "Nessun dato valido per il plotting.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio (Nessun Dato)", fontsize=16)
                plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
                continue


            df = pd.DataFrame(plot_data)
            df["Categoria"] = df["Categoria"].apply(str).astype("category")
            df["Scenario"] = df["Scenario"].apply(str).astype("category")
            df["Tempo Medio (s)"] = df["Tempo Medio (s)"].astype(float)
            df["Errore"] = df["Errore"].astype(float)


            sns.barplot(
                data=df,
                x="Categoria", y="Tempo Medio (s)",
                hue="Scenario",
                order=category_names,
                hue_order=["Senza Priorità", "Con Priorità", "WFQ"], # MODIFIED
                palette=["#ff0000", "#0000ff", "#32CD32"], # MODIFIED: Added a color for WFQ
                ax=ax, dodge=True
            )

            num_categories, width = len(category_names), 0.25 # Adjusted width for 3 bars
            x_positions = np.arange(num_categories)

            for i, scenario in enumerate(["Senza Priorità", "Con Priorità", "WFQ"]): # MODIFIED
                offset = (i - 1) * width
                subset = df[df["Scenario"] == scenario].set_index("Categoria").reindex(category_names)

                if subset["Tempo Medio (s)"].isnull().all():
                    continue

                y_coords = subset["Tempo Medio (s)"].fillna(0)
                errors = subset["Errore"].fillna(0)
                ax.errorbar(x_positions + offset, y_coords, yerr=errors, fmt="none", c="black", capsize=5, elinewidth=1.2)

                for cat_idx, category_name in enumerate(category_names):
                    row = subset.loc[category_name]

                    if pd.notna(row["Tempo Medio (s)"]):
                        try:
                            mean_val, error_val = row["Tempo Medio (s)"], row["Errore"]
                            upper_bound = mean_val + error_val
                            ci_text = f"[{max(0, mean_val - error_val):.3f}, {upper_bound:.3f}]"
                            ax.annotate(
                                ci_text,
                                xy=(x_positions[cat_idx] + offset, upper_bound),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color="black"
                            )
                        except ValueError:
                            continue


            ax.set_title(f"Tempo di {'Risposta' if metric_name == 'response' else 'Attesa'} Medio", fontsize=16)
            ax.set_ylabel("Tempo Medio (s)", fontsize=12)
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            ax.legend(title="Scenario")

            current_bottom, current_top = ax.get_ylim()
            ax.set_ylim(bottom=current_bottom, top=current_top * 1.03)

        fig.suptitle("Confronto Tempi Medi (Steady State) per Tipo con IC al 95%", fontsize=20, fontweight="bold")
        plt.tight_layout()
        fig.subplots_adjust(top=0.90)
        self._save_plot(output_dir, "times_comparison.png", fig)


    def _plot_single_scenario_loss(self, results, scenario_name, color, output_dir, filename):
        fig, ax = plt.subplots(figsize=(8, 6))

        mean_val = float(results['mean'])
        half_width = float(results['half_width'])

        bars = ax.bar(scenario_name, mean_val, yerr=half_width, color=color, capsize=10, alpha=0.8, width=0.4)
        ax.set_title(f'Probabilità di Perdita (Steady State) - {scenario_name}', fontsize=16)
        ax.set_ylabel('Probabilità di Perdita Stimata')
        ax.set_xlabel('Scenario')

        max_y = (mean_val + half_width) * 1.5 if not (np.isnan(mean_val) or np.isnan(half_width)) else 1.0
        ax.set_ylim(bottom=0, top=max_y)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        if bars and not np.isnan(mean_val):
            ax.bar_label(bars, fmt='%.4f', padding=5, fontsize=10, weight='bold')

            patch = bars.patches[0]
            upper_bound = mean_val + half_width
            ci_text = f"IC 95%: [{max(0, mean_val - half_width):.4f}, {upper_bound:.4f}]"
            ax.annotate(ci_text,
                        xy=(patch.get_x() + patch.get_width() / 2, upper_bound),
                        xytext=(0, 12), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)


    def plot_steady_state_loss(self, warmup: dict, output_dir):
        """
        Genera i grafici della probabilità di perdita a regime, sia per scenario Baseline, Priorità, che WFQ,
        includendo il confronto con intervalli di confidenza al 95%.
        """
        print("Generazione grafico di CONFRONTO per probabilità di perdita...")

        baseline_outcomes = self.metrics.get_all_outcomes_as_binary_stream()
        prio_outcomes = self.metrics_prio.get_all_outcomes_as_binary_stream()
        wfq_outcomes = self.metrics_wfq.get_all_outcomes_as_binary_stream() # ADDED

        def compute_loss_ci(outcomes, warmup_period, confidence_level, threshold):
            if not outcomes:
                return None

            first = outcomes[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                steady_values = [float(v) for t, v in outcomes if t >= warmup_period]
            else:
                try:
                    idx = int(warmup_period)
                except Exception:
                    idx = 0
                steady_values = [float(v) for v in outcomes[idx:]]

            if len(steady_values) < 2:
                return None

            b_k_rho = compute_batch_size(steady_values, k_initial_target=getattr(self.config, 'BATCH_K', 64), threshold=threshold)
            if not b_k_rho or b_k_rho[0] is None or b_k_rho[1] is None:
                return None
            b, k = int(b_k_rho[0]), int(b_k_rho[1])

            if b * k > len(steady_values):
                return None

            res = batch_means(steady_values, b, k, confidence=confidence_level)

            try:
                res['mean'] = float(res['mean'])
            except Exception:
                res['mean'] = float(np.asarray(res['mean']).item())
            res['half_width'] = float(res['half_width'])
            res['batch_size'] = int(res.get('batch_size', b))
            res['num_batches'] = int(res.get('num_batches', k))
            return res

        baseline_results = compute_loss_ci(baseline_outcomes, warmup["baseline"], self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD)
        prio_results = compute_loss_ci(prio_outcomes, warmup["priority"], self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD)
        wfq_results = compute_loss_ci(wfq_outcomes, warmup["wfq"], self.config.CONFIDENCE_LEVEL, self.config.BATCH_THRESHOLD) # ADDED

        if baseline_results:
            self._plot_single_scenario_loss(baseline_results, 'Senza Priorità', '#ff0000', output_dir, "loss_probability_baseline.png")
        else:
            print("Warning: Nessun risultato per la probabilità di perdita Baseline.")

        if prio_results:
            self._plot_single_scenario_loss(prio_results, 'Con Priorità', '#0000ff', output_dir, "loss_probability_prio.png")
        else:
            print("Warning: Nessun risultato per la probabilità di perdita con Priorità.")

        if wfq_results: # ADDED
            self._plot_single_scenario_loss(wfq_results, 'WFQ', '#32CD32', output_dir, "loss_probability_wfq.png")
        else:
            print("Warning: Nessun risultato per la probabilità di perdita con WFQ.")


        fig, ax = plt.subplots(figsize=(8, 6))
        plot_data = []
        if baseline_results:
            plot_data.append({'Scenario': 'Senza Priorità', 'Mean': baseline_results['mean'], 'Half_Width': baseline_results['half_width']})
        if prio_results:
            plot_data.append({'Scenario': 'Con Priorità', 'Mean': prio_results['mean'], 'Half_Width': prio_results['half_width']})
        if wfq_results: # ADDED
            plot_data.append({'Scenario': 'WFQ', 'Mean': wfq_results['mean'], 'Half_Width': wfq_results['half_width']})


        if not plot_data:
            ax.text(0.5, 0.5, "Nessun dato disponibile per il confronto di perdita.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Confronto Probabilità di Perdita (Nessun Dato)', fontsize=16)
        else:
            df_loss = pd.DataFrame(plot_data)
            df_loss['Mean'] = df_loss['Mean'].astype(float)
            df_loss['Half_Width'] = df_loss['Half_Width'].astype(float)

            if self.use_log_scale_infinite:
                ax.set_yscale('log')
                y_max_val = df_loss['Mean'].max() + df_loss['Half_Width'].max()
                ax.set_ylim(bottom=max(1e-6, y_max_val * 0.1), top=y_max_val * 2)
            else:
                max_y_val = (df_loss['Mean'] + df_loss['Half_Width']).max() * 1.2
                ax.set_ylim(bottom=0, top=max_y_val)

            bars = ax.bar(df_loss['Scenario'], df_loss['Mean'],
                          yerr=df_loss['Half_Width'],
                          color=['#ff0000', '#0000ff', '#32CD32'][:len(df_loss)], # MODIFIED
                          capsize=10, alpha=0.8, width=0.5)

            ax.set_title('Confronto Probabilità di Perdita (Steady State) con IC al 95%', fontsize=16)
            ax.set_ylabel('Probabilità di Perdita Stimata')
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)

            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=10, weight='bold')

            for i, bar in enumerate(bars):
                mean_val = float(df_loss['Mean'].iloc[i])
                half_width = float(df_loss['Half_Width'].iloc[i])
                upper_bound = mean_val + half_width
                ci_text = f"[{max(0, mean_val - half_width):.4f}, {upper_bound:.4f}]"
                ax.annotate(ci_text,
                            xy=(bar.get_x() + bar.get_width() / 2, upper_bound),
                            xytext=(0, 14), textcoords='offset points',
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self._save_plot(output_dir, "loss_probability_comparison.png", fig)


    def _plot_convergence_baseline_by_type(self, output_dir, warmup_duration):
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type, history in self.metrics.response_times_history.items():
            if history:
                history.sort(key=lambda x: x[0])
                timestamps, values = zip(*history)
                ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), label=f'{req_type.name}', color=self.req_type_colors.get(req_type), linewidth=2)
        ax.set_title('Analisi Convergenza per Tipo (Baseline)'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_duration}s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, "baseline_convergence_by_type.png", fig)

    def _plot_convergence_prio_by_type(self, output_dir, warmup_duration):
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type in sorted(self.metrics_prio.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_prio.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
                sorted_timestamps, sorted_values = zip(*history)
                ax.plot(sorted_timestamps, np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1), label=f'{req_type.name}', color=self.req_type_colors.get(req_type), linewidth=2)
        ax.set_title('Analisi Convergenza per Tipo (Con Priorità)'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_duration}s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, "prio_convergence_by_type.png", fig)

    def _plot_convergence_wfq_by_type(self, output_dir, warmup_duration): # ADDED THIS METHOD
        fig, ax = plt.subplots(figsize=(12, 7))
        for req_type in sorted(self.metrics_wfq.response_times_by_req_type.keys(), key=lambda x: x.name):
            response_times = self.metrics_wfq.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                history = sorted(zip(timestamps, response_times), key=lambda x: x[0])
                sorted_timestamps, sorted_values = zip(*history)
                ax.plot(sorted_timestamps, np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1), label=f'{req_type.name}', color=self.req_type_colors.get(req_type), linewidth=2)
        ax.set_title('Analisi Convergenza per Tipo (WFQ)'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_duration}s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Tipo di Richiesta')
        plt.tight_layout()
        self._save_plot(output_dir, "wfq_convergence_by_type.png", fig)


    def plot_convergence_analysis_by_type(self, output_dir, warmup_durations: dict):
        baseline_warmup = warmup_durations["baseline"]
        priority_warmup = warmup_durations["priority"]
        wfq_warmup = warmup_durations["wfq"] # ADDED

        self._plot_convergence_baseline_by_type(output_dir, warmup_duration=baseline_warmup)
        self._plot_convergence_prio_by_type(output_dir, warmup_duration=priority_warmup)
        self._plot_convergence_wfq_by_type(output_dir, warmup_duration=wfq_warmup) # ADDED

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
        for req_type in sorted(self.metrics_wfq.response_times_by_req_type.keys(), key=lambda x: x.name): # ADDED
            response_times = self.metrics_wfq.response_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if response_times and len(response_times) == len(timestamps):
                sorted_timestamps, sorted_values = zip(*sorted(zip(timestamps, response_times), key=lambda x: x[0]))
                ax.plot(sorted_timestamps, np.cumsum(sorted_values) / np.arange(1, len(sorted_values) + 1), label=f'{req_type.name} (WFQ)', color=self.req_type_colors.get(req_type), linestyle=':', linewidth=2.5) # ADDED, changed linestyle

        ax.axvline(x=baseline_warmup, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({baseline_warmup}s)')
        ax.set_title('Confronto Convergenza per Tipo di Richiesta'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        ax.legend(title='Scenario e Tipo', bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        self._save_plot(output_dir, "convergence_by_type_comparison.png", fig)

    def _plot_convergence_overall(self, all_responses, scenario_name, output_dir, filename, color, warmup_duration):
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_responses:
            timestamps, values = zip(*all_responses)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label='Tempo Risposta Medio Cumulativo')
            ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_duration}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Analisi Convergenza Tempo Risposta Medio ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_convergence_analysis_overall(self, output_dir, warmup_durations: dict):
        baseline_warmup = warmup_durations["baseline"]
        priority_warmup = warmup_durations["priority"]
        wfq_warmup = warmup_durations["wfq"] # ADDED

        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        all_responses_wfq = self.metrics_wfq.get_all_response_times_with_timestamps() # ADDED

        self._plot_convergence_overall(all_responses_base, "Senza Priorità", output_dir, "baseline_convergence_overall.png", 'r', warmup_duration=baseline_warmup)
        self._plot_convergence_overall(all_responses_prio, "Con Priorità", output_dir, "prio_convergence_overall.png", 'b', warmup_duration=priority_warmup)
        self._plot_convergence_overall(all_responses_wfq, "WFQ", output_dir, "wfq_convergence_overall.png", '#32CD32', warmup_duration=wfq_warmup) # ADDED

        print("Generazione grafico di CONFRONTO di convergenza generale...")
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_responses_base:
            timestamps, values = zip(*all_responses_base)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color='r', label='Senza Priorità')
        if all_responses_prio:
            timestamps, values = zip(*all_responses_prio)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color='b', label='Con Priorità')
        if all_responses_wfq: # ADDED
            timestamps, values = zip(*all_responses_wfq)
            ax.plot(timestamps, np.cumsum(values) / np.arange(1, len(values) + 1), color='#32CD32', label='WFQ') # ADDED
        if all_responses_base or all_responses_prio or all_responses_wfq: # MODIFIED
            ax.axvline(x=baseline_warmup, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({baseline_warmup}s)')
        ax.set_title('Confronto Convergenza del Tempo di Risposta Medio'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Risposta Medio (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "convergence_overall_comparison.png", fig)

    def _plot_wait_time_trend(self, all_waits, scenario_name, output_dir, filename, color, warmup_duration):
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_waits:
            times, values = zip(*all_waits)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color=color, label='Tempo Attesa Medio Cumulativo')
            ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({warmup_duration}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Evoluzione del Tempo di Attesa Medio ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_wait_time_trend_analysis(self, output_dir, warmup: dict):
        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        all_waits_base = sorted([item for sublist in self.metrics.wait_times_history.values() for item in sublist], key=lambda x:x[0])

        all_waits_prio_list = []
        for req_type in self.metrics_prio.wait_times_by_req_type.keys():
            waits = self.metrics_prio.wait_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_prio_list.extend(zip(timestamps, waits))
        all_waits_prio = sorted(all_waits_prio_list, key=lambda x: x[0])

        # ADDED WFQ DATA COLLECTION
        all_waits_wfq_list = []
        for req_type in self.metrics_wfq.wait_times_by_req_type.keys():
            waits = self.metrics_wfq.wait_times_by_req_type.get(req_type, [])
            timestamps = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if len(waits) == len(timestamps):
                all_waits_wfq_list.extend(zip(timestamps, waits))
        all_waits_wfq = sorted(all_waits_wfq_list, key=lambda x: x[0])


        self._plot_wait_time_trend(all_waits_base, "Senza Priorità", output_dir, "wait_time_trend_baseline.png", 'r', warmup_duration=baseline_warmup_duration)
        self._plot_wait_time_trend(all_waits_prio, "Con Priorità", output_dir, "wait_time_trend_prio.png", 'b', warmup_duration=priority_warmup_duration)
        self._plot_wait_time_trend(all_waits_wfq, "WFQ", output_dir, "wait_time_trend_wfq.png", '#32CD32', warmup_duration=wfq_warmup_duration) # ADDED

        print("Generazione grafico di CONFRONTO andamento tempo di attesa...")
        fig, ax = plt.subplots(figsize=(12, 7))
        if all_waits_base:
            times, values = zip(*all_waits_base)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color='r', label='Senza Priorità')
        if all_waits_prio:
            times, values = zip(*all_waits_prio)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color='b', label='Con Priorità')
        if all_waits_wfq: # ADDED
            times, values = zip(*all_waits_wfq)
            ax.plot(times, np.cumsum(values) / np.arange(1, len(values) + 1), color='#32CD32', label='WFQ') # ADDED


        ax.axvline(x=baseline_warmup_duration, color='k', linestyle=':', linewidth=2, label=f'Fine Warm-up ({baseline_warmup_duration}s)')

        ax.set_title('Confronto Evoluzione del Tempo di Attesa Medio')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Tempo di Attesa Medio Cumulativo (s)')
        ax.grid(True, which='both', linestyle='--', alpha=0.7); ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "wait_time_trend_comparison.png", fig)

    def _plot_pod_history(self, times, counts, scenario_name, output_dir, filename, color, warmup_duration):
        fig, ax = plt.subplots(figsize=(14, 7))
        if times and counts:
            ax.plot(times, counts, color=color, label='Numero di Pod', alpha=0.8, linewidth=1.5)
            ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup_duration}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Evoluzione del Numero di Pod ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_pod_history_analysis(self, output_dir, warmup: dict):
        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        times_b, pods_b = zip(*self.metrics.pod_count_history) if self.metrics.pod_count_history else ([], [])
        self._plot_pod_history(times_b, pods_b, "Senza Priorità", output_dir, "pod_history_baseline.png", 'r', warmup_duration=baseline_warmup_duration)
        self._plot_pod_history(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, "Con Priorità", output_dir, "pod_history_prio.png", 'b', warmup_duration=priority_warmup_duration)
        self._plot_pod_history(self.metrics_wfq.timestamps, self.metrics_wfq.pod_counts, "WFQ", output_dir, "pod_history_wfq.png", '#32CD32', warmup_duration=wfq_warmup_duration) # ADDED

        print("Generazione grafico di CONFRONTO storico dei Pod...")
        fig, ax = plt.subplots(figsize=(14, 7))
        if times_b and pods_b: ax.plot(times_b, pods_b, color='r', label='Senza Priorità', alpha=0.8, linewidth=1.5)
        if self.metrics_prio.timestamps and self.metrics_prio.pod_counts: ax.plot(self.metrics_prio.timestamps, self.metrics_prio.pod_counts, color='b', label='Con Priorità', alpha=0.8, linewidth=1.5)
        if self.metrics_wfq.timestamps and self.metrics_wfq.pod_counts: ax.plot(self.metrics_wfq.timestamps, self.metrics_wfq.pod_counts, color='#32CD32', label='WFQ', alpha=0.8, linewidth=1.5) # ADDED

        ax.axvline(x=baseline_warmup_duration, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({baseline_warmup_duration}s)')
        ax.set_title('Confronto Evoluzione del Numero di Pod'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Numero di Pod Attivi')
        ax.set_ylim(bottom=0, top=self.config.MAX_PODS + 1); ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, "pod_history_comparison.png", fig)

    def _plot_queue_history(self, times, queue_lengths, scenario_name, output_dir, filename, color, use_log_scale, warmup_duration):
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'

        dark_color_map = {'r': 'darkred', 'b': 'darkblue', '#32CD32': 'darkgreen'} # MODIFIED

        if times and queue_lengths:
            ax.plot(times, queue_lengths, color=color, label='Lunghezza Coda', alpha=0.7, linewidth=1.5)
            steady_queue = [q for t, q in zip(times, queue_lengths) if t >= warmup_duration]

            dark_color = dark_color_map.get(color, color)
            if steady_queue:
                ax.axhline(np.mean(steady_queue), color=dark_color, linestyle='--', label=f'Media Steady-State: {np.mean(steady_queue):.2f}')

            ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup_duration}s)')
        else:
            ax.text(0.5, 0.5, "Nessun dato disponibile", ha='center', va='center', transform=ax.transAxes)

        if use_log_scale:
            ax.set_yscale('log'); ylabel += ' (Scala Log)'; ax.set_ylim(bottom=0.1)

        ax.set_title(f'Evoluzione Lunghezza della Coda ({scenario_name})')
        ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)

    def plot_queue_history_analysis(self, warmup: dict, output_dir, use_log_scale=True):
        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        times_b, queue_b = zip(*self.metrics.queue_length_history) if self.metrics.queue_length_history else ([],[])
        self._plot_queue_history(times_b, queue_b, "Senza Priorità", output_dir, f"queue_history_baseline{'_log' if use_log_scale else ''}.png", 'r', use_log_scale, warmup_duration=baseline_warmup_duration)
        self._plot_queue_history(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, "Con Priorità", output_dir, f"queue_history_prio{'_log' if use_log_scale else ''}.png", 'b', use_log_scale, warmup_duration=priority_warmup_duration)
        self._plot_queue_history(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths, "WFQ", output_dir, f"queue_history_wfq{'_log' if use_log_scale else ''}.png", '#32CD32', use_log_scale, warmup_duration=wfq_warmup_duration) # ADDED

        print("Generazione grafico di CONFRONTO storico della Coda...")
        fig, ax = plt.subplots(figsize=(14, 7))
        ylabel = 'Numero Richieste in Coda'
        if times_b and queue_b:
            ax.plot(times_b, queue_b, color='r', label='Senza Priorità', alpha=0.7, linewidth=1.5)
            if steady_queue_b := [q for t, q in zip(times_b, queue_b) if t >= baseline_warmup_duration ]:
                ax.axhline(np.mean(steady_queue_b), color='darkred', linestyle=':', label=f'Media Steady (Baseline): {np.mean(steady_queue_b):.2f}')
        if self.metrics_prio.timestamps and self.metrics_prio.queue_lengths:
            ax.plot(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths, color='b', label='Con Priorità', alpha=0.7, linewidth=1.5)
            if steady_queue_p := [q for t, q in zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths) if t >= priority_warmup_duration]:
                ax.axhline(np.mean(steady_queue_p), color='darkblue', linestyle=':', label=f'Media Steady (Priorità): {np.mean(steady_queue_p):.2f}')
        if self.metrics_wfq.timestamps and self.metrics_wfq.queue_lengths: # ADDED
            ax.plot(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths, color='#32CD32', label='WFQ', alpha=0.7, linewidth=1.5) # ADDED
            if steady_queue_wfq := [q for t, q in zip(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths) if t >= wfq_warmup_duration]: # ADDED
                ax.axhline(np.mean(steady_queue_wfq), color='darkgreen', linestyle=':', label=f'Media Steady (WFQ): {np.mean(steady_queue_wfq):.2f}') # ADDED

        if use_log_scale:
            ax.set_yscale('log'); ylabel += ' (Scala Log)'; ax.set_ylim(bottom=0.1)
        ax.set_title('Confronto Evoluzione della Lunghezza della Coda'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel(ylabel)
        ax.grid(True, which='both', linestyle='--', alpha=0.6); ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, f"queue_history_comparison{'_log' if use_log_scale else ''}.png", fig)


    def _plot_variance_trend(self, all_responses, scenario_name, output_dir, filename, color, window_size, warmup_duration):
        """
        Grafico della deviazione standard mobile del tempo di risposta per uno scenario.
        Usa la classe Welford per il calcolo della varianza.
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        steady_data = [(t, v) for t, v in all_responses if t >= warmup_duration]

        if len(steady_data) > window_size:
            times, values = zip(*steady_data)
            welford_window = Welford()
            moving_std = []
            window_elements = []

            for i, val in enumerate(values):
                window_elements.append(val)
                welford_window.add(val)

                if len(window_elements) > window_size:
                    window_elements.pop(0)
                    welford_window = Welford()
                    welford_window.add_all(window_elements)


                if i >= window_size - 1:
                    if welford_window.count > 1:
                        moving_std.append(np.sqrt(welford_window.var_s))
                    else:
                        moving_std.append(0)

            if moving_std:
                ax.plot(times[window_size-1:], moving_std, color=color, label='Dev. Std. Mobile', alpha=0.8)
            ax.axvline(x=warmup_duration, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({warmup_duration}s)')
        else:
            ax.text(0.5, 0.5, f"Dati insufficienti (necessari > {window_size} dopo warmup)", ha='center', va='center', transform=ax.transAxes)

        ax.set_title(f'Stabilizzazione Varianza ({scenario_name}) - Finestra di {window_size}')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend()
        plt.tight_layout()
        self._save_plot(output_dir, filename, fig)


    def plot_variance_trend_analysis(self, output_dir, warmup_durations: dict, window_size=500):
        """
        Grafici della deviazione standard mobile e confronto tra baseline e priorità.
        """
        all_responses_base = self.metrics.get_all_response_times_with_timestamps()
        all_responses_prio = self.metrics_prio.get_all_response_times_with_timestamps()
        all_responses_wfq = self.metrics_wfq.get_all_response_times_with_timestamps() # ADDED

        baseline_warmup = warmup_durations["baseline"]
        priority_warmup = warmup_durations["priority"]
        wfq_warmup = warmup_durations["wfq"] # ADDED

        self._plot_variance_trend(all_responses_base, "Senza Priorità", output_dir, "variance_trend_baseline.png", 'r', window_size, warmup_duration=baseline_warmup)
        self._plot_variance_trend(all_responses_prio, "Con Priorità", output_dir, "variance_trend_prio.png", 'b', window_size, warmup_duration=priority_warmup)
        self._plot_variance_trend(all_responses_wfq, "WFQ", output_dir, "variance_trend_wfq.png", '#32CD32', window_size, warmup_duration=wfq_warmup) # ADDED

        print("Generazione grafico di CONFRONTO andamento della varianza...")
        fig, ax = plt.subplots(figsize=(14, 7))

        for responses, color, label, warmup_duration in [
            (all_responses_base, 'r', 'Senza Priorità', baseline_warmup),
            (all_responses_prio, 'b', 'Con Priorità', priority_warmup),
            (all_responses_wfq, '#32CD32', 'WFQ', wfq_warmup) # ADDED
        ]:
            steady_data = [(t, v) for t, v in responses if t >= warmup_duration]
            if len(steady_data) > window_size:
                times, values = zip(*steady_data)
                welford_window = Welford()
                moving_std = []
                window_elements = []

                for i, val in enumerate(values):
                    window_elements.append(val)
                    welford_window.add(val)

                    if len(window_elements) > window_size:
                        window_elements.pop(0)
                        welford_window = Welford()
                        welford_window.add_all(window_elements)


                    if i >= window_size - 1:
                        if welford_window.count > 1:
                            moving_std.append(np.sqrt(welford_window.var_s))
                        else:
                            moving_std.append(0)

                if moving_std:
                    ax.plot(times[window_size-1:], moving_std, color=color, label=label, alpha=0.8)

        ax.axvline(x=baseline_warmup, color='k', linestyle=':', linewidth=2.5, label=f'Fine Warm-up ({baseline_warmup}s)')
        ax.set_title(f'Confronto Stabilizzazione Varianza (Finestra di {window_size})')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Deviazione Standard Mobile del Tempo di Risposta')
        ax.set_ylim(bottom=0)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.legend(title='Scenario')
        plt.tight_layout()
        self._save_plot(output_dir, "variance_trend_comparison.png", fig)


    def _plot_batch_mean_queue_single(self, data, scenario_name, warmup_duration, num_batches_k, output_dir, color):
        fig, ax = plt.subplots(figsize=(14, 7))
        if not data or not (steady_data := [(t, v) for t, v in data if t >= warmup_duration]):
            ax.text(0.5, 0.5, "Nessun dato disponibile/in steady-state", ha='center', va='center', transform=ax.transAxes)
        else:
            total_duration = steady_data[-1][0] - warmup_duration
            if total_duration > 0 and num_batches_k > 0:
                batch_duration = total_duration / num_batches_k
                batch_means_values, batch_timestamps = [], []

                for i in range(num_batches_k):
                    batch_start_time = warmup_duration + i * batch_duration
                    batch_end_time = warmup_duration + (i + 1) * batch_duration
                    values_in_batch = [v for t, v in steady_data if batch_start_time <= t < batch_end_time]

                    if values_in_batch:
                        batch_means_values.append(np.mean(values_in_batch))
                        batch_timestamps.append(batch_start_time + (batch_duration / 2))

                if batch_timestamps: ax.plot(batch_timestamps, batch_means_values, marker='o', linestyle='-', color=color, label="Media per Batch")
            else:
                ax.text(0.5, 0.5, "Durata steady-state insufficiente o numero di batch non valido.", ha='center', va='center', transform=ax.transAxes)


        ax.set_title(f'Evoluzione Medie per Batch della Coda ({scenario_name})'); ax.set_xlabel('Tempo di Simulazione (s)'); ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(); ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(left=0); ax.set_ylim(bottom=0)
        plt.tight_layout()
        self._save_plot(output_dir, f"queue_batch_means_trend_{scenario_name.lower().replace(' ', '_')}.png", fig)

    def plot_batch_mean_queue_trend_analysis(self, warmup: dict, batches: dict, output_dir):
        """
        Plotta le medie per batch della coda per ciascuno scenario (baseline e priorità)
        e il grafico di confronto.
        """
        baseline_warmup_duration = warmup["baseline"]
        priority_warmup_duration = warmup["priority"]
        wfq_warmup_duration = warmup["wfq"] # ADDED

        k_base_optimal = batches["baseline"][3] if batches["baseline"] and len(batches["baseline"]) > 3 else 0
        k_prio_optimal = batches["priority"][3] if batches["priority"] and len(batches["priority"]) > 3 else 0
        k_wfq_optimal = batches["wfq"][3] if batches["wfq"] and len(batches["wfq"]) > 3 else 0 # ADDED


        self._plot_batch_mean_queue_single(
            self.metrics.queue_length_history,
            "Senza Priorità",
            baseline_warmup_duration,
            k_base_optimal,
            output_dir,
            'r'
        )

        data_prio = list(zip(self.metrics_prio.timestamps, self.metrics_prio.queue_lengths)) if self.metrics_prio.queue_lengths else []
        self._plot_batch_mean_queue_single(
            data_prio,
            "Con Priorità",
            priority_warmup_duration,
            k_prio_optimal,
            output_dir,
            'b'
        )

        data_wfq = list(zip(self.metrics_wfq.timestamps, self.metrics_wfq.queue_lengths)) if self.metrics_wfq.queue_lengths else [] # ADDED
        self._plot_batch_mean_queue_single(
            data_wfq,
            "WFQ",
            wfq_warmup_duration,
            k_wfq_optimal,
            output_dir,
            '#32CD32' # ADDED
        )


        print("Generazione grafico CONFRONTO trend delle medie dei batch della coda...")

        fig, ax = plt.subplots(figsize=(14, 7))
        scenarios = {
            "Senza Priorità": (self.metrics.queue_length_history, baseline_warmup_duration, k_base_optimal, 'r'),
            "Con Priorità": (data_prio, priority_warmup_duration, k_prio_optimal, 'b'),
            "WFQ": (data_wfq, wfq_warmup_duration, k_wfq_optimal, '#32CD32') # ADDED
        }

        for scenario_name, (data, scenario_warmup, num_batches_k, color) in scenarios.items():
            if not data:
                continue

            steady_data = [(t, v) for t, v in data if t >= scenario_warmup]
            if not steady_data:
                continue

            total_duration = steady_data[-1][0] - scenario_warmup
            if total_duration <= 0 or num_batches_k <= 0:
                continue

            batch_duration = total_duration / num_batches_k
            batch_means_values, batch_timestamps = [], []

            for i in range(num_batches_k):
                batch_start_time = scenario_warmup + i * batch_duration
                batch_end_time = scenario_warmup + (i + 1) * batch_duration
                values_in_batch = [v for t, v in steady_data if batch_start_time <= t < batch_end_time]

                if values_in_batch:
                    batch_means_values.append(np.mean(values_in_batch))
                    batch_timestamps.append(batch_start_time + batch_duration / 2)

            if batch_timestamps:
                ax.plot(batch_timestamps, batch_means_values, marker='o', linestyle='-', color=color, label=scenario_name)

        ax.set_title('Confronto Evoluzione delle Medie per Batch della Coda')
        ax.set_xlabel('Tempo di Simulazione (s)')
        ax.set_ylabel('Lunghezza Media della Coda per Batch')
        ax.legend(title='Scenario')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        self._save_plot(output_dir, "queue_batch_means_trend_comparison.png", fig)


    def _plot_throughput_single_scenario(self, analyzer, metrics, scenario_name, warmup_durations_dict, output_dir, color):
        """
        Grafico del throughput per singolo scenario.
        - analyzer: oggetto che calcola il throughput con batch means
        - metrics: oggetto Metrics o MetricsWithPriority
        - warmup_durations_dict: dict {"baseline": valore, "priority": valore, "wfq": valore}
        """
        is_prio_or_wfq = isinstance(metrics, MetricsWithPriority) # Generalize check

        fig, ax = plt.subplots(figsize=(16, 9))

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        plot_data = []

        for req_type in all_req_types:
            timestamps = []
            warmup_value = 0

            if scenario_name == "Senza Priorità":
                timestamps = sorted([ts for ts, _ in metrics.response_times_history.get(req_type, [])])
                warmup_value = warmup_durations_dict.get("baseline")
            elif scenario_name == "Con Priorità":
                timestamps = sorted(metrics.completion_timestamps_by_req_type.get(req_type, []))
                warmup_value = warmup_durations_dict.get("priority")
            elif scenario_name == "WFQ": # ADDED
                timestamps = sorted(metrics.completion_timestamps_by_req_type.get(req_type, []))
                warmup_value = warmup_durations_dict.get("wfq") # ADDED


            if not timestamps:
                continue

            results = analyzer.calculate_throughput_ci(timestamps, warmup_value, confidence_level=self.config.CONFIDENCE_LEVEL, threshold=self.config.BATCH_THRESHOLD)
            if results:
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Conteggio': results['total_count']
                })

        if not plot_data:
            print(f"Dati insufficienti per il grafico del throughput ({scenario_name}).")
            ax.text(0.5, 0.5, "Nessun dato disponibile per il plotting.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"Richieste Servite con Successo per Tipo ({scenario_name}) (Nessun Dato)", fontsize=18)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            self._save_plot(output_dir, f"throughput_{scenario_name.lower().replace(' ', '_')}.png", fig)
            return

        df = pd.DataFrame(plot_data)
        sns.barplot(data=df, x='Categoria', y='Conteggio', order=category_names, color=color, ax=ax)

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                fontsize=11, color='black',
                xytext=(0, 5), textcoords='offset points'
            )

        ax.set_title(f"Richieste Servite con Successo per Tipo ({scenario_name})", fontsize=18)
        ax.set_xlabel("Tipo di Richiesta", fontsize=14)
        ax.set_ylabel("Numero di Richieste Servite", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

        plt.tight_layout()
        self._save_plot(output_dir, f"throughput_{scenario_name.lower().replace(' ', '_')}.png", fig)



    def plot_throughput_analysis(self, analyzer_baseline, analyzer_prio, analyzer_wfq, warmup: dict, batches: dict, output_dir): # MODIFIED: added analyzer_wfq
        self._plot_throughput_single_scenario(analyzer_baseline, self.metrics, "Senza Priorità", warmup, output_dir, '#ff0000')
        self._plot_throughput_single_scenario(analyzer_prio, self.metrics_prio, "Con Priorità", warmup, output_dir, '#0000ff')
        self._plot_throughput_single_scenario(analyzer_wfq, self.metrics_wfq, "WFQ", warmup, output_dir, '#32CD32') # ADDED


        print("Generazione grafico di CONFRONTO delle richieste soddisfatte (throughput)...")

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.suptitle("Confronto Richieste Servite per Tipo - Steady State", fontsize=24, fontweight='bold')

        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        category_names = [req.name.replace('_', ' ').title() for req in all_req_types]

        plot_data = []

        baseline_warmup = warmup["baseline"]
        priority_warmup = warmup["priority"]
        wfq_warmup = warmup["wfq"] # ADDED

        for req_type in all_req_types:
            # Baseline
            timestamps_b = sorted([ts for ts, _ in self.metrics.response_times_history.get(req_type, [])])
            if results_b := analyzer_baseline.calculate_throughput_ci(timestamps_b, baseline_warmup, confidence_level=self.config.CONFIDENCE_LEVEL, threshold=self.config.BATCH_THRESHOLD):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Conteggio': results_b['total_count'],
                    'Scenario': 'Senza Priorità'
                })

            # Priorità
            timestamps_p = sorted(self.metrics_prio.completion_timestamps_by_req_type.get(req_type, []))
            if results_p := analyzer_prio.calculate_throughput_ci(timestamps_p, priority_warmup, confidence_level=self.config.CONFIDENCE_LEVEL, threshold=self.config.BATCH_THRESHOLD):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Conteggio': results_p['total_count'],
                    'Scenario': 'Con Priorità'
                })

            # WFQ # ADDED
            timestamps_wfq = sorted(self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, []))
            if results_wfq := analyzer_wfq.calculate_throughput_ci(timestamps_wfq, wfq_warmup, confidence_level=self.config.CONFIDENCE_LEVEL, threshold=self.config.BATCH_THRESHOLD):
                plot_data.append({
                    'Categoria': req_type.name.replace('_', ' ').title(),
                    'Conteggio': results_wfq['total_count'],
                    'Scenario': 'WFQ'
                })


        if not plot_data:
            print("Dati insufficienti per il grafico di confronto del throughput.")
            ax.text(0.5, 0.5, "Nessun dato disponibile per il confronto di throughput.", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Confronto Richieste Servite per Tipo (Nessun Dato)", fontsize=18)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            plt.tight_layout()
            self._save_plot(output_dir, "throughput_comparison.png", fig)
            return

        df = pd.DataFrame(plot_data)

        sns.barplot(
            data=df,
            x='Categoria', y='Conteggio',
            hue='Scenario',
            order=category_names,
            hue_order=['Senza Priorità', 'Con Priorità', 'WFQ'], # MODIFIED
            palette=['#ff0000', '#0000ff', '#32CD32'], # MODIFIED
            ax=ax
        )

        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=11, color='black',
                        xytext=(0, 5), textcoords='offset points')

        y_top = ax.get_ylim()[1]
        # Calculate max y to place annotations well
        max_height = df['Conteggio'].max() if not df.empty else 0
        y_annotation_pos = max_height * 1.1 if max_height > 0 else 1.0 # Adjust vertically based on max bar height

        for i, cat_name in enumerate(category_names):
            base_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Senza Priorità')]
            prio_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'Con Priorità')]
            wfq_row = df[(df['Categoria'] == cat_name) & (df['Scenario'] == 'WFQ')] # ADDED

            base_count = base_row.iloc[0]['Conteggio'] if not base_row.empty else 0
            prio_count = prio_row.iloc[0]['Conteggio'] if not prio_row.empty else 0
            wfq_count = wfq_row.iloc[0]['Conteggio'] if not wfq_row.empty else 0 # ADDED

            # Delta vs Baseline for Priority
            delta_prio_perc, sign_prio, color_prio = (0, '', 'gray')
            if base_count > 0:
                delta_prio_perc = ((prio_count - base_count) / base_count) * 100
                sign_prio, color_prio = ('+', 'green') if delta_prio_perc >= 0 else ('', 'red')

            # Delta vs Baseline for WFQ # ADDED
            delta_wfq_perc, sign_wfq, color_wfq = (0, '', 'gray')
            if base_count > 0:
                delta_wfq_perc = ((wfq_count - base_count) / base_count) * 100
                sign_wfq, color_wfq = ('+', 'green') if delta_wfq_perc >= 0 else ('', 'red')

            # Display deltas next to each other
            ax.text(i - 0.2, y_annotation_pos * 1.05, f'Prio Δ: {sign_prio}{delta_prio_perc:.1f}%', # Adjusted y-position slightly
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor=color_prio, alpha=0.9))
            ax.text(i + 0.2, y_annotation_pos * 1.05, f'WFQ Δ: {sign_wfq}{delta_wfq_perc:.1f}%', # Adjusted y-position slightly
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor=color_wfq, alpha=0.9))


        ax.set_title("Richieste Servite con Successo per Tipo", fontsize=18)
        ax.set_xlabel("Tipo di Richiesta", fontsize=14)
        ax.set_ylabel("Numero di Richieste Servite", fontsize=14)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(title='Scenario', loc='upper left')
        ax.set_ylim(top=y_annotation_pos * 1.2) # Adjusted ylim to accommodate annotations better
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        self._save_plot(output_dir, "throughput_comparison.png", fig)


    def plot_times_by_request_type_grid(self, output_dir, warmup_durations: dict):
        print("Generazione griglia di confronto per tipo di richiesta...")
        all_req_types = sorted(list(self.metrics.requests_generated_data.keys()), key=lambda x: x.name)
        ncols, i = 3, -1
        nrows = int(np.ceil(len(all_req_types) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5), sharex=True, sharey=True)
        axes = axes.flatten()

        baseline_warmup_duration = warmup_durations["baseline"]
        priority_warmup_duration = warmup_durations["priority"]
        wfq_warmup_duration = warmup_durations["wfq"] # ADDED

        for i, req_type in enumerate(all_req_types):
            ax = axes[i]
            # Baseline
            if resp_b := sorted(self.metrics.response_times_history.get(req_type, []), key=lambda x: x[0]):
                times, values = zip(*resp_b)
                ax.plot(times, np.cumsum(values) / np.arange(1, len(values)+1), color='salmon', linestyle='--', label='Risposta (Baseline)')
            if wait_b := sorted(self.metrics.wait_times_history.get(req_type, []), key=lambda x: x[0]):
                times, values = zip(*wait_b)
                ax.plot(times, np.cumsum(values) / np.arange(1, len(values)+1), color='red', label='Attesa (Baseline)')

            # Priority
            times_p = self.metrics_prio.completion_timestamps_by_req_type.get(req_type, [])
            if (values_rp := self.metrics_prio.response_times_by_req_type.get(req_type, [])) and len(times_p) == len(values_rp):
                times_s, values_s = zip(*sorted(zip(times_p, values_rp), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='lightblue', linestyle='--', label='Risposta (Priorità)')
            if (values_wp := self.metrics_prio.wait_times_by_req_type.get(req_type, [])) and len(times_p) == len(values_wp):
                times_s, values_s = zip(*sorted(zip(times_p, values_wp), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='blue', label='Attesa (Priorità)')

            # WFQ # ADDED
            times_wfq = self.metrics_wfq.completion_timestamps_by_req_type.get(req_type, [])
            if (values_rwfq := self.metrics_wfq.response_times_by_req_type.get(req_type, [])) and len(times_wfq) == len(values_rwfq):
                times_s, values_s = zip(*sorted(zip(times_wfq, values_rwfq), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='#90EE90', linestyle=':', label='Risposta (WFQ)') # LightGreen
            if (values_wwfq := self.metrics_wfq.wait_times_by_req_type.get(req_type, [])) and len(times_wfq) == len(values_wwfq):
                times_s, values_s = zip(*sorted(zip(times_wfq, values_wwfq), key=lambda x: x[0]))
                ax.plot(times_s, np.cumsum(values_s) / np.arange(1, len(values_s)+1), color='#32CD32', label='Attesa (WFQ)') # LimeGreen

            # Add vertical line for warmup duration
            ax.axvline(x=baseline_warmup_duration, color='gray', linestyle=':', linewidth=1.5, label=f'Warm-up End ({baseline_warmup_duration}s)')

            ax.set_title(req_type.name.replace('_', ' ').title())
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
        if i != -1:
            for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        fig.supxlabel('Tempo di Simulazione (s)', y=0.02)
        fig.supylabel('Tempo Medio Cumulativo (s)', x=0.02)
        plt.tight_layout(rect=(0.03, 0.03, 1, 0.95))
        self._save_plot(output_dir, "times_grid_comparison.png", fig)