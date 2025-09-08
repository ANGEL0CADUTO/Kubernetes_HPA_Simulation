import numpy as np
import os
from src import config
# MODIFICA: Importa i nuovi simulatori dedicati
from src.simulation.simulator_blackfriday import SimulatorBlackFridayBaseline, SimulatorBlackFriday as SimulatorBlackFridayDWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter_blackfriday import PlotterBlackFriday
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_blackfriday_analysis():
    print("="*80)
    print("ANALISI SCENARIO 'BLACK FRIDAY' (ARCHITETTURA A SILOS)")
    print("="*80)

    NUM_REPLICATIONS = 1
    config.SIMULATION_TIME = 30000

    # Definizione del profilo di carico variabile
    t_notte_fine = config.SIMULATION_TIME * 0.15
    t_mattina_fine = config.SIMULATION_TIME * 0.30
    t_picco1_fine = config.SIMULATION_TIME * 0.40
    t_pomeriggio_fine = config.SIMULATION_TIME * 0.70
    t_picco2_fine = config.SIMULATION_TIME * 0.80
    CARICO_NOTTE = 10; CARICO_MATTINA = 85; CARICO_PICCO_1 = 300
    CARICO_POMERIGGIO = 170; CARICO_PICCO_2 = 320

    def lambda_black_friday(t: float) -> float:
        if t < t_notte_fine: return CARICO_NOTTE
        if t < t_mattina_fine: return CARICO_MATTINA
        if t < t_picco1_fine: return CARICO_PICCO_1
        if t < t_pomeriggio_fine: return CARICO_POMERIGGIO
        if t < t_picco2_fine: return CARICO_PICCO_2
        return CARICO_NOTTE

    print(f"\nConfigurazione Esperimento 'Black Friday':")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s, Repliche: {NUM_REPLICATIONS}")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        master_seed_replica = rng_manager._master_rng._next_seed()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {master_seed_replica} per entrambi i sistemi ---")

        # 1. Baseline (FIFO)
        rng_base = RNGManager(master_seed=master_seed_replica)
        streams_base, _ = rng_base.get_replication_streams()
        simulator_base = SimulatorBlackFridayBaseline(config, Metrics, streams_base['arrivals'], streams_base['choice'], streams_base['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 1: Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # 2. Dynamic Weighted Fair Queuing (DWFQ)
        rng_wfq = RNGManager(master_seed=master_seed_replica)
        streams_wfq, _ = rng_wfq.get_replication_streams()
        simulator_wfq = SimulatorBlackFridayDWFQ(config, MetricsWithPriority, streams_wfq['arrivals'], streams_wfq['choice'], streams_wfq['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 2: Dynamic Weighted Fair Queuing (DWFQ) ---")
        simulator_wfq.run(config.SIMULATION_TIME)

        # PLOTTING
        output_folder = f"output/black_friday_analysis/replica_{i+1}"
        plotter = PlotterBlackFriday(
            metrics_base_agg=simulator_base.metrics_agg,
            metrics_prio_agg=None, # Non eseguiamo Priorità Strette qui
            metrics_wfq_agg=simulator_wfq.metrics_agg,
            metrics_per_worker_base=simulator_base.metrics_per_worker,
            metrics_per_worker_prio=None,
            metrics_per_worker_wfq=simulator_wfq.metrics_per_worker,
            config_module=config
        )

        print(f"\n--- Generazione grafici per la replica {i+1} ---")
        plotter.generate_final_dashboards(
            output_dir=output_folder,
            run_prefix=f"bf_repl_{i+1}",
            lambda_func=lambda_black_friday
        )

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

if __name__ == "__main__":
    main_blackfriday_analysis()