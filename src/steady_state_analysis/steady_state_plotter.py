import os
import matplotlib.pyplot as plt
import numpy as np

from src.steady_state_analysis.steady_state_plots import  CumulativeResponsePlotter


class SteadyStatePlotter:
    """
    [CLASSE RISTRUTTURATA E SEMPLIFICATA v2.0]
    Classe responsabile della generazione di un unico tipo di grafico:
    l'analisi visuale del tempo di risposta medio per batch con intervallo di confidenza,
    con stile personalizzato.
    """
    def __init__(self, config):
        """
        Inizializza il plotter.
        Args:
            config: Oggetto di configurazione della simulazione.
        """
        self.config = config
        # REQ 2: Colori aggiornati a rosa e azzurro
        self.scenario_colors = {
            "Senza Priorità": '#FF69B4', # Rosa acceso (HotPink)
            "WFQ": '#87CEEB',          # Azzurro cielo (SkyBlue)
            # Manteniamo questo per non generare errori se dovesse apparire
            "Con Priorità": '#A9A9A9'   # Grigio scuro per lo scenario escluso
        }

    def _save_plot(self, output_dir: str, filename: str, fig: plt.Figure):
        """
        Salva un grafico nella directory specificata e chiude la figura.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico salvato in: {save_path}")

    def plot_batch_means_analysis(self, steady_values: list[float], results: dict, scenario_name: str, output_dir: str):
        """
        [METODO REVISIONATO v2.1]
        Genera il grafico di analisi visuale del metodo Batch Means per il tempo di risposta
        con le personalizzazioni grafiche richieste, inclusa la nuova posizione della legenda.
        """
        fig, ax = plt.subplots(figsize=(18, 10), layout="constrained")

        if not results or len(steady_values) < 2:
            ax.text(0.5, 0.5, "Dati insufficienti o analisi fallita.\nImpossibile generare il grafico.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=16, color='red')
            ax.set_title(f'Analisi Tempo di Risposta per Batch - {scenario_name} [DATI MANCANTI]', fontsize=24)
            self._save_plot(output_dir, f"response_time_ci_{scenario_name.lower().replace(' ', '_')}.png", fig)
            return

        b, k, grand_mean = results.get('batch_size'), results.get('num_batches'), results.get('mean')
        ci_lower, ci_upper = results.get('ci', (np.nan, np.nan))
        color = self.scenario_colors.get(scenario_name, 'gray')

        if not all(v is not None for v in [b, k, grand_mean]) or not (b > 0 and k >= 2 and len(steady_values) >= b * k):
            ax.text(0.5, 0.5, f"Parametri batch non validi (b={b}, k={k}) per {len(steady_values)} campioni.",
                    ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
            ax.set_title(f'Analisi Tempo di Risposta per Batch - {scenario_name} [PARAMETRI NON VALIDI]', fontsize=24)
            self._save_plot(output_dir, f"response_time_ci_{scenario_name.lower().replace(' ', '_')}.png", fig)
            return

        batch_means_values = [np.mean(steady_values[i*b : (i+1)*b]) for i in range(k)]
        batch_numbers = np.arange(1, k + 1)

        ax.plot(batch_numbers, batch_means_values, marker='o', color=color, linestyle='-',
                label=f'Media del Batch ({scenario_name})', alpha=0.8, markersize=6)
        ax.axhline(grand_mean, color='black', linestyle='--', linewidth=2.5,
                   label=f'Media Globale: {grand_mean:.4f}s')
        ax.axhspan(ci_lower, ci_upper, color='gray', alpha=0.25,
                   label=f'IC al {results["confidence_level"]:.0%}: [{ci_lower:.4f}, {ci_upper:.4f}]s')



        title_fontsize = 24
        label_fontsize = 22
        tick_fontsize = 20
        legend_fontsize = 18

        ax.set_title(f'Analisi Tempo di Risposta per Batch - {scenario_name}', fontsize=title_fontsize, weight='bold')
        ax.set_xlabel('Numero del Batch', fontsize=label_fontsize)
        ax.set_ylabel('Tempo di Risposta Medio per Batch (s)', fontsize=label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # --- MODIFICA CHIAVE: Posizione Legenda ---
        # Usa 'loc' per posizionarla internamente. 'lower center' è una buona scelta.
        ax.legend(loc='lower center', fancybox=True, shadow=True, ncol=3, fontsize=legend_fontsize)

        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.set_xlim(left=0, right=k + 1)

        min_val = min(batch_means_values)
        max_val = max(batch_means_values)
        padding = (max_val - min_val) * 0.30
        ax.set_ylim(bottom=max(0, min_val - padding), top=max_val + padding)

        ljung_box_pvalue_str = f"{results['ljung_box_pvalue']:.4f}" if results.get('ljung_box_pvalue') is not None else "N/A"
        stats_text = (
            f"Stima Puntuale (Media): {grand_mean:.4f} s\n"
            f"Intervallo di Confidenza: [{ci_lower:.4f}, {ci_upper:.4f}] s\n"
            f"Semi-ampiezza: {results['half_width']:.4f} s\n"
            f"Batch Size (b): {b}, Num. Batch (k): {k}\n"
            f"Ljung-Box p-value: {ljung_box_pvalue_str} "
            f"({'OK' if results.get('independence_ok', False) else 'NO'})"
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.2))

        self._save_plot(output_dir, f"response_time_ci_{scenario_name.lower().replace(' ', '_')}.png", fig)
    def generate_final_report(self, all_steady_values: dict, all_results: dict, output_dir: str):
        """
        [ORCHESTRATORE REVISIONATO]
        Genera i grafici finali per gli scenari richiesti.
        """
        print("\n--- [FASE FINALE] Generazione Report Grafici Batch Means ---")

        # REQ 1: Scenari da confrontare sono solo Baseline e WFQ
        scenarios = ["Senza Priorità", "WFQ"]

        for scenario in scenarios:
            if scenario in all_results and scenario in all_steady_values:
                print(f"  - Generazione grafico per lo scenario: {scenario}")
                self.plot_batch_means_analysis(
                    steady_values=all_steady_values.get(scenario, []),
                    results=all_results.get(scenario, {}),
                    scenario_name=scenario,
                    output_dir=output_dir
                )
                print(f"  - Generazione grafici cumulativi per lo scenario: {scenario}")
                # Estrai i dati necessari
                results = all_results.get(scenario, {})
                steady_values = all_steady_values.get(scenario, [])

                b = results.get('batch_size')
                k = results.get('num_batches')
                grand_mean = results.get('mean')
                system_label = scenario

                # Calcola batch_mean_values da passare alla classe CumulativeResponsePlotter
                batch_mean_values = []
                if b is not None and k is not None and b > 0 and k > 0 and len(steady_values) >= b * k:
                    batch_mean_values = [np.mean(steady_values[i*b : (i+1)*b]) for i in range(k)]
                else:
                    print(f"    - ATTENZIONE: Dati batch insufficienti o non validi per i grafici cumulativi dello scenario '{scenario}'.")

                # Istanzia e chiama i metodi di plotting cumulativo SOLO SE ci sono dati validi per i batch
                if batch_mean_values:
                    cumulative_plotter = CumulativeResponsePlotter(output_dir=output_dir)

                    cumulative_plotter.plot_by_batch(
                        batch_mean_values=batch_mean_values,
                        overall_mean=grand_mean,
                        system_label=system_label
                    )

                    cumulative_plotter.plot_by_jobs(
                        batch_mean_values=batch_mean_values,
                        batch_size=b,
                        overall_mean=grand_mean,
                        system_label=system_label
                    )
                else:
                    print(f"    - ATTENZIONE: Saltati i grafici cumulativi per lo scenario '{scenario}' a causa di dati insufficienti o non validi.")


            else:
                print(f"  - ATTENZIONE: Dati o risultati mancanti per lo scenario '{scenario}'. Grafico non generato.")

        print("\n--- Report grafici generati con successo. ---")





