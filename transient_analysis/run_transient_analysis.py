
import os

from src import config
from src.analysis.new_plotter import newPlotter
from src.simulation.simulator import Simulator
from src.simulation.simulator_wfq import SimulatorWFQ
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG as RNGManager  # Usiamo un alias per chiarezza
from src.utils.metrics import Metrics # Importa la classe Metrics
from src.utils.metrics_with_priority import MetricsWithPriority # Importa la classe MetricsWithPriority
from src.utils.acs import batch_means, compute_batch_size
import matplotlib.pyplot as plt

def main():
    """
    Funzione principale che orchestra l'analisi del transitorio
    per i modelli Baseline (FIFO) e DWFQ.
    """
    print("--- Inizio Analisi del Transitorio (FIFO vs. DWFQ) ---")

    NUM_REPLICATIONS = 5
    SIM_DURATION = 10000 # Durata lunga per garantire il raggiungimento dello stato stazionario

    arrival_scenarios = {
        "steady_load_170": lambda t: 170,
    }

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    all_results = {scenario_name: {} for scenario_name in arrival_scenarios}

    # --- CICLO ESTERNO: REPLICHE ---
    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        replication_streams, rep_seed = rng_manager.get_replication_streams()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {rep_seed} ---")

        # --- CICLO INTERNO: SCENARI ---
        for scenario_name, lambda_fn in arrival_scenarios.items():
            print(f"\n--- ESECUZIONE SCENARIO: {scenario_name.upper()} ---")

            # --- ESECUZIONE BASELINE (FIFO) ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO BASELINE (FIFO) ---")
            simulator_base = Simulator(
                config, Metrics,
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_base.run(simulation_duration=SIM_DURATION)
            metrics_base = simulator_base.metrics_agg


            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO DWFQ ---")
            simulator_wfq = SimulatorWFQ(
                config, MetricsWithPriority,
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_wfq.run(simulation_duration=SIM_DURATION)
            metrics_wfq = simulator_wfq.metrics_agg

            all_results[scenario_name][i] = {
                'baseline': metrics_base,
                'priority': metrics_wfq,
                'seed': rep_seed
            }


    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE E SCENARI {'='*30}")
    print("\n--- Generazione di tutti i report di simulazione completata ---")

    return all_results, NUM_REPLICATIONS


if __name__ == "__main__":
    all_results, num_replications = main()

    if all_results:
        print("\n--- Generazione Grafico Analisi del Transitorio (FIFO vs. DWFQ) ---")

        transient_plotter = newPlotter(None, None, config)

        # Questa chiamata ora funzionerà per entrambi i modelli.
        # Il primo subplot mostrerà il Baseline, il secondo mostrerà il DWFQ.
        transient_plotter.plot_replication_traces_per_scenario(
            all_results,
            output_dir='output/transient_analysis'
        )

    print("\n--- Processo di Simulazione e Analisi Completato ---")