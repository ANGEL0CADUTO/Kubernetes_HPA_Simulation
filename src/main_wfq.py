# File: main_wfq.py

import numpy as np
import os
from src import config
from src.simulation.simulator import Simulator
from src.simulation.simulator_with_priority import SimulatorWithPriority as SimulatorSPS # Usiamo il prioritario come SPS
from src.simulation.simulator_wfq import SimulatorWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter_wfq import PlotterWFQ
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_wfq_analysis():
    print("="*80)
    print("ANALISI FINALE DI CONFRONTO: FIFO vs. PRIORITA' STRETTE vs. WFQ")
    print("="*80)

    NUM_REPLICATIONS = 3 # Esegui una sola replica per ora per analizzare i grafici
    CARICO_BASE = 6; CARICO_PICCO = 170; INIZIO_PICCO = 200
    DURATA_PICCO = 300; FINE_PICCO = INIZIO_PICCO + DURATA_PICCO

    def lambda_con_picco(t: float) -> float:
        if INIZIO_PICCO <= t < FINE_PICCO: return CARICO_PICCO
        return CARICO_BASE

    config.SIMULATION_TIME = 700

    print(f"\nConfigurazione Esperimento:")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s, Repliche: {NUM_REPLICATIONS}")
    print(f" - Carico: Base={CARICO_BASE} req/s, Picco={CARICO_PICCO} req/s")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")
        master_seed_replica = rng_manager._master_rng._next_seed()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {master_seed_replica} per tutti i sistemi ---")

        # 1. Baseline (FIFO)
        rng_base = RNGManager(master_seed=master_seed_replica); streams_base, _ = rng_base.get_replication_streams()
        metrics_base = Metrics(config)
        simulator_base = Simulator(config, metrics_base, streams_base['arrivals'], streams_base['choice'], streams_base['service'], lambda_con_picco)
        print("\n--- Esecuzione Modello 1: Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # 2. Priorità Strette (SPS)
        rng_prio = RNGManager(master_seed=master_seed_replica); streams_prio, _ = rng_prio.get_replication_streams()
        metrics_prio = MetricsWithPriority(config)
        simulator_prio = SimulatorSPS(config, metrics_prio, streams_prio['arrivals'], streams_prio['choice'], streams_prio['service'], lambda_con_picco)
        print("\n--- Esecuzione Modello 2: Priorità Strette ---")
        simulator_prio.run(config.SIMULATION_TIME)

        # 3. Weighted Fair Queuing (WFQ)
        rng_wfq = RNGManager(master_seed=master_seed_replica); streams_wfq, _ = rng_wfq.get_replication_streams()
        metrics_wfq = MetricsWithPriority(config)
        simulator_wfq = SimulatorWFQ(config, metrics_wfq, streams_wfq['arrivals'], streams_wfq['choice'], streams_wfq['service'], lambda_con_picco)
        print("\n--- Esecuzione Modello 3: Weighted Fair Queuing (WFQ) ---")
        simulator_wfq.run(config.SIMULATION_TIME)

        output_folder = f"output/WFQ_analysis/replica_{i+1}"
        plotter = PlotterWFQ(metrics_base, metrics_prio, metrics_wfq, config)
        plotter.generate_final_dashboards(output_dir=output_folder, run_prefix=f"final_repl_{i+1}", peak_start=INIZIO_PICCO, peak_end=FINE_PICCO, base_load=CARICO_BASE, peak_load=CARICO_PICCO)

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

if __name__ == "__main__":
    main_wfq_analysis()