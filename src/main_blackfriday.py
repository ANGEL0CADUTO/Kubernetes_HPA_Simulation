# In main_blackfriday.py

import numpy as np
import os
from src import config
from src.simulation.simulator_blackfriday import SimulatorBlackFridayBaseline, SimulatorBlackFriday as SimulatorBlackFridayDWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter_blackfriday import PlotterBlackFriday
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_blackfriday_analysis():
    print("="*80)
    print("ANALISI SCENARIO 'BLACK FRIDAY' ")
    print("="*80)

    NUM_REPLICATIONS = 5
    config.SIMULATION_TIME = 2000

    t_notte_fine = config.SIMULATION_TIME * 0.15
    t_mattina_fine = config.SIMULATION_TIME * 0.30
    t_pranzo_fine = config.SIMULATION_TIME * 0.45
    t_picco_fine = config.SIMULATION_TIME * 0.55
    t_pomeriggio_fine = config.SIMULATION_TIME * 0.70
    t_sera_fine = config.SIMULATION_TIME * 0.85

    CARICO_1 = 255
    CARICO_2 = 255
    CARICO_3 = 255
    CARICO_4 = 255

    def lambda_black_friday(t: float) -> float:
        if t < t_notte_fine:
            return CARICO_1          # 20
        if t < t_mattina_fine:
            return CARICO_2          # 40
        if t < t_pranzo_fine:
            return CARICO_3          # 85
        if t < t_picco_fine:
            return CARICO_4          # 170
        if t < t_pomeriggio_fine:
            return CARICO_3          # 85
        if t < t_sera_fine:
            return CARICO_2          # 40
        return CARICO_1              # 20

    print(f"\nConfigurazione Esperimento 'Black Friday':")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s, Repliche: {NUM_REPLICATIONS}")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    all_results = {}

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        replication_streams, rep_seed = rng_manager.get_replication_streams()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {rep_seed} per entrambi i sistemi ---")

        # 1. Baseline (FIFO)
        simulator_base = SimulatorBlackFridayBaseline(config, Metrics, replication_streams['arrivals'], replication_streams['choice'], replication_streams['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 1: Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # 2. Dynamic Weighted Fair Queuing (DWFQ)
        simulator_wfq = SimulatorBlackFridayDWFQ(config, MetricsWithPriority, replication_streams['arrivals'], replication_streams['choice'], replication_streams['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 2: Dynamic Weighted Fair Queuing (DWFQ) ---")

        simulator_wfq.run(config.SIMULATION_TIME)


        all_results[i] = {
            'baseline': simulator_base.metrics_agg,
            'wfq': simulator_wfq.metrics_agg,
            'seed': rep_seed
        }

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

    if all_results:
        # Inizializziamo il plotter con i dati della prima replica, solo per soddisfare il costruttore
        plotter = PlotterBlackFriday(
            metrics_base_agg=all_results[0]['baseline'],
            metrics_prio_agg=None,
            metrics_wfq_agg=all_results[0]['wfq'],
            metrics_per_worker_base=None, # Non ci servono per il nuovo grafico
            metrics_per_worker_prio=None,
            metrics_per_worker_wfq=None,
            config_module=config
        )

        # ANALISI DEL TRANSITORIO #
        print("\n--- Generazione grafici di confronto repliche per scenario Black Friday ---")
        plotter.plot_blackfriday_replication_traces(
            all_results,
            lambda_func=lambda_black_friday,
            output_dir="output/black_friday_analysis/aggregated_traces"
        )
        plotter.plot_confidence_interval_trace(
            all_results,
            lambda_black_friday,
            output_dir="output/black_friday_analysis/aggregated_traces/IC"
        )


if __name__ == "__main__":
    main_blackfriday_analysis()