# File: main.py (VERSİONE CORRETTA E AGGIORNATA)

import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src import config
from src.simulation.simulator import Simulator
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter import Plotter
from analysis.validation_plotter import ValidationPlotter
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter

# MODIFICATO: L'import ora carica la NUOVA classe che gestisce gli stream
from src.utils.lehmer_rng import LehmerRNG as RNGManager # Usiamo un alias per chiarezza

def main():
    """
    Funzione principale che orchestra l'intero processo di simulazione.
    """
    print("--- Inizio Progetto di Simulazione E-commerce (Versione con Rigore Metodologico) ---")

    NUM_REPLICATIONS = 2

    arrival_scenarios = {
        "tasso_70": lambda t: 85,
        #"tasso_85": lambda t: 170,
        # "tasso_100": lambda t: 255, # Temporaneamente escluso per test rapidi
    }

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    all_results = {scenario_name: {} for scenario_name in arrival_scenarios}

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")
        for scenario_name, lambda_fn in arrival_scenarios.items():
            print(f"\n--- ESECUZIONE SCENARIO: {scenario_name.upper()} ---")
            replication_streams, rep_seed = rng_manager.get_replication_streams()

            # --- ESECUZIONE BASELINE ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO BASELINE (FIFO) ---")
            metrics_base = Metrics(config_module=config)
            simulator_base = Simulator(
                config_module=config, metrics=metrics_base,
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_base.run(simulation_duration=config.SIMULATION_TIME)

            # --- ESECUZIONE MIGLIORATA ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO MIGLIORATO (PRIORITY) ---")
            metrics_prio = MetricsWithPriority(config)
            simulator_prio = SimulatorWithPriority(
                config_module=config, metrics=metrics_prio,
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_prio.run(simulation_duration=config.SIMULATION_TIME)

            all_results[scenario_name][i] = {
                'baseline': metrics_base,
                'priority': metrics_prio,
                'seed': rep_seed
            }
            # --- Generazione report dettagliati per QUESTA specifica replica ---
            print(f"\n--- Generazione report per {scenario_name}, Replica {i+1} ---")
            output_folder = f"output/plots_{scenario_name}/replica_{i+1}"

            single_run_plotter = Plotter(metrics_base, metrics_prio, config)
            single_run_plotter.generate_comprehensive_report(
                output_dir=output_folder,
                run_prefix=f"{scenario_name}_repl{i+1}"
            )

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE E SCENARI {'='*30}")

    # La funzione main ora restituisce semplicemente i dati raccolti
    return all_results, NUM_REPLICATIONS



    # # --- ANALISI FINALE DEI RISULTATI ---
    # # La generazione dei report per singola replica è utile per il debug, ma possiamo commentarla
    # # per concentrarci sui risultati aggregati.
    # # print("\n--- Generazione report di esempio (basati sull'ultima replica) ---")
    # # last_replica_index = NUM_REPLICATIONS - 1
    # # for scenario_name in arrival_scenarios:
    # #     output_folder = f"output/plots_{scenario_name}"
    # #     os.makedirs(output_folder, exist_ok=True)
    # #     last_run_metrics = all_results[scenario_name][last_replica_index]
    # #     plotter = Plotter(last_run_metrics['baseline'], last_run_metrics['priority'], config)
    # #     plotter.generate_comprehensive_report(output_dir=output_folder, run_prefix=f"{scenario_name}_repl{last_replica_index+1}")
    #
    #
    #
    #
    # # --- STEADY-STATE (se abilitato) ---
    # if config.STEADY_ENABLED:
    #     print("\n--- Inizio Simulazione Steady-State ---")
    #     run_steady_state_experiment(rng_manager)
    #     print("--- Fine Simulazione Steady-State ---")
    #
    # print("\n--- Processo di Simulazione e Analisi Completato ---")
    # return all_results, NUM_REPLICATIONS # <-- Questa riga è necessaria


def run_steady_state_experiment(rng_manager: RNGManager):
    """
    Esegue la simulazione a orizzonte infinito per l'analisi di regime permanente.
    (Questa funzione rimane invariata)
    """
    print("\n--- AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")
    output_dir = "plots/steady_state"
    steady_lambda_fn = lambda t: 85
    steady_streams = rng_manager.get_replication_streams()

    print("\n--- Esecuzione Scenario Baseline (Steady-State) ---")
    metrics_baseline = Metrics(config_module=config)
    simulator_baseline = Simulator(
        config_module=config, metrics=metrics_baseline,
        arrival_rng=steady_streams['arrivals'], choice_rng=steady_streams['choice'],
        service_rng=steady_streams['service'], lambda_function=steady_lambda_fn
    )
    simulator_baseline.run(simulation_duration=config.STEADY_SIMULATION_TIME)

    print("\n--- Esecuzione Scenario con Priorità (Steady-State) ---")
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(
        config_module=config, metrics=metrics_prio,
        arrival_rng=steady_streams['arrivals'], choice_rng=steady_streams['choice'],
        service_rng=steady_streams['service'], lambda_function=steady_lambda_fn
    )
    simulator_prio.run(simulation_duration=config.STEADY_SIMULATION_TIME)

    print("\n--- Generazione Report Steady-State ---")
    analyzer_baseline = SteadyStateAnalyzer(metrics_baseline, config)
    analyzer_prio = SteadyStateAnalyzer(metrics_prio, config)
    steady_plotter = SteadyStatePlotter(metrics_baseline, metrics_prio, config)
    steady_plotter.generate_steady_state_report(
        analyzer_baseline=analyzer_baseline, analyzer_prio=analyzer_prio,
        warmup=config.WARM_UP_TO_STEADY, batches=config.NUM_BATCHES,
        output_dir=output_dir
    )
    print("\n--- Fine dell'analisi Steady-State ---")



if __name__ == "__main__":

    # 1. Esegui tutte le simulazioni e ottieni i risultati
    all_results, num_replications = main()

    # 2. Se le simulazioni hanno prodotto risultati, procedi con l'analisi aggregata
    if all_results:
        # Inizializziamo un Plotter. I dati passati all'init sono irrilevanti
        # per il metodo di plotting aggregato, che riceve tutto ciò di cui ha bisogno.
        final_plotter = Plotter(None, None, config)

        # Chiamiamo il metodo corretto per generare i grafici delle tracce
        final_plotter.plot_replication_traces_per_scenario(all_results, num_replications)

    # 3. Esegui l'analisi steady-state se è abilitata nel config
    #    (Questa parte è separata e non è stata toccata)
    if config.STEADY_ENABLED:
        # Dovremmo ridefinire qui le funzioni o importarle correttamente
        # Per ora, si assume che non sia l'obiettivo principale
        print("\n--- Esecuzione Steady-State non implementata in questo script pulito ---")

    print("\n--- Processo di Simulazione e Analisi Completato ---")