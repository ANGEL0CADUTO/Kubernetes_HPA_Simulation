import os
import matplotlib.pyplot as plt

from src import config
from src.simulation.simulator import Simulator
from src.simulation.simulator_wfq import SimulatorWFQ
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG as RNGManager
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority


def run_steady_state_experiment(rng_manager: RNGManager):
    """
    [FUNZIONE PRINCIPALE FINALE]
    Orchestra l'esperimento a stato stazionario per gli scenari Baseline e WFQ.
    1. Esegue le simulazioni.
    2. Centralizza l'analisi statistica (warm-up, campionamento, Batch Means).
    3. Chiama il plotter per generare i grafici finali.
    """
    print("\n--- AVVIO ESPERIMENTO STEADY-STATE ---")
    output_dir = "plots/steady_state"
    os.makedirs(output_dir, exist_ok=True)

    # --- Parametri della simulazione ---
    simulation_duration = config.STEADY_SIMULATION_TIME
    steady_lambda_fn = lambda t: 85

    print(f"--- Configurazione: Durata={simulation_duration}s, Tasso di Arrivo={steady_lambda_fn(0)} req/s ---")

    steady_streams_dict, steady_rep_seed = rng_manager.get_replication_streams()
    print(f"--- La run Steady-State utilizzerà il SEED master: {steady_rep_seed} ---")

    # ==========================================================================
    # 1. ESECUZIONE DELLE SIMULAZIONI
    # ==========================================================================
    print("\n--- [FASE 1/3] Esecuzione delle Simulazioni ---")

    # --- Scenario Baseline ("Senza Priorità") ---
    print("  - Esecuzione Scenario Baseline...")
    simulator_baseline = Simulator(config, Metrics,
                                   arrival_rng=steady_streams_dict['arrivals'],
                                   choice_rng=steady_streams_dict['choice'],
                                   service_rng=steady_streams_dict['service'],
                                   lambda_function=steady_lambda_fn)
    simulator_baseline.run(simulation_duration=simulation_duration)
    metrics_baseline = simulator_baseline.metrics_agg

    # --- Scenario WFQ ---
    print("  - Esecuzione Scenario WFQ...")
    simulator_wfq = SimulatorWFQ(config, MetricsWithPriority,
                                 arrival_rng=steady_streams_dict['arrivals'],
                                 choice_rng=steady_streams_dict['choice'],
                                 service_rng=steady_streams_dict['service'],
                                 lambda_function=steady_lambda_fn)
    simulator_wfq.run(simulation_duration=simulation_duration)
    metrics_wfq = simulator_wfq.metrics_agg

    # ==========================================================================
    # 2. ANALISI STATISTICA CENTRALIZZATA
    # ==========================================================================
    print("\n--- [FASE 2/3] Analisi Statistica Centralizzata ---")

    all_steady_values = {}
    all_results = {}

    # REQ 1: Scenari da analizzare sono solo Baseline e WFQ
    scenarios_to_analyze = {
        "Senza Priorità": metrics_baseline,
        "WFQ": metrics_wfq
    }

    for name, metrics_obj in scenarios_to_analyze.items():
        print(f"\n--- Analisi per lo scenario: {name} ---")
        analyzer = SteadyStateAnalyzer(metrics_obj, config)

        print("  - Estrazione e campionamento dei dati (1 ogni 2)...")
        full_data_sampled = analyzer.extract_full_response_data()
        values_only_sampled = [v for _, v in full_data_sampled]
        print(f"    - Numero di campioni dopo thinning: {len(values_only_sampled)}")

        print("  - Stima del periodo di warm-up...")
        warmup_time = config.WARMUP
        print(f"    - Warm-up stimato a: {warmup_time:.2f} secondi")

        steady_values = [v for t, v in full_data_sampled if t >= warmup_time]
        all_steady_values[name] = steady_values
        print(f"    - Numero di campioni a regime: {len(steady_values)}")

        print("  - Calcolo Intervallo di Confidenza con Batch Means...")
        results = analyzer.steady_state_analysis(steady_values)
        all_results[name] = results

        analyzer.print_ci_results(results, f"{name} - Overall Response Time")

    # ==========================================================================
    # 3. GENERAZIONE DEI REPORT GRAFICI
    # ==========================================================================
    print("\n--- [FASE 3/3] Generazione Report Grafici Finali ---")

    plt.style.use('./style/plot_style.mplstyle')
    print("  - Stile 'plot_style.mplstyle' caricato.")

    steady_plotter = SteadyStatePlotter(config)

    steady_plotter.generate_final_report(
        all_steady_values=all_steady_values,
        all_results=all_results,
        output_dir=output_dir
    )

    print("\n--- Esperimento Steady-State Completato ---")


if __name__ == "__main__":


    rng_manager_for_steady_state = RNGManager(master_seed=config.LEHMER_SEED)
    run_steady_state_experiment(rng_manager_for_steady_state)


    print("\n--- Processo di Simulazione e Analisi Concluso ---")