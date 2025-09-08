import numpy as np
import os
from src import config
from src.simulation.simulator import Simulator
# ALIAS CORRETTO: Usiamo SimulatorPriority per chiarezza.
from src.simulation.simulator_with_priority import SimulatorWithPriority as SimulatorPriority
from src.simulation.simulator_wfq import SimulatorWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter_wfq import PlotterWFQ
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_wfq_analysis():
    print("="*80)
    print("ANALISI FINALE DI CONFRONTO (ARCHITETTURA A SILOS): FIFO vs. PRIORITA' vs. WFQ")
    print("="*80)

    NUM_REPLICATIONS = 5 # Manteniamo 1 per un'analisi più rapida dei grafici
    CARICO_BASE = 6
    CARICO_PICCO = 300
    INIZIO_PICCO = 200
    DURATA_PICCO = 300
    FINE_PICCO = INIZIO_PICCO + DURATA_PICCO

    def lambda_con_picco(t: float) -> float:
        if INIZIO_PICCO <= t < FINE_PICCO: return CARICO_PICCO
        return CARICO_BASE

    config.SIMULATION_TIME = 900

    print(f"\nConfigurazione Esperimento:")
    print(f" - Architettura: {config.NUM_WORKERS} Worker Node a 'silos'")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s, Repliche: {NUM_REPLICATIONS}")
    print(f" - Carico: Base={CARICO_BASE} req/s, Picco={CARICO_PICCO} req/s")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        master_seed_replica = rng_manager._master_rng._next_seed()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {master_seed_replica} per tutti i sistemi ---")

        # 1. Baseline (FIFO)
        rng_base = RNGManager(master_seed=master_seed_replica)
        streams_base, _ = rng_base.get_replication_streams()
        simulator_base = Simulator(config, Metrics, streams_base['arrivals'], streams_base['choice'], streams_base['service'], lambda_con_picco)
        print("\n--- Esecuzione Modello 1: Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # # 2. Priorità Strette
        # rng_prio = RNGManager(master_seed=master_seed_replica)
        # streams_prio, _ = rng_prio.get_replication_streams()
        # simulator_prio = SimulatorPriority(config, MetricsWithPriority, streams_prio['arrivals'], streams_prio['choice'], streams_prio['service'], lambda_con_picco)
        # print("\n--- Esecuzione Modello 2: Priorità Strette ---")
        # simulator_prio.run(config.SIMULATION_TIME)

        # 3. Weighted Fair Queuing (WFQ)
        rng_wfq = RNGManager(master_seed=master_seed_replica)
        streams_wfq, _ = rng_wfq.get_replication_streams()
        simulator_wfq = SimulatorWFQ(config, MetricsWithPriority, streams_wfq['arrivals'], streams_wfq['choice'], streams_wfq['service'], lambda_con_picco)
        print("\n--- Esecuzione Modello 3: Weighted Fair Queuing (WFQ) ---")
        simulator_wfq.run(config.SIMULATION_TIME)

        # PLOTTING DEI RISULTATI
        output_folder = f"output/WFQ_analysis/replica_{i+1}"

        # Passiamo al plotter sia le metriche aggregate che quelle per-worker.
        plotter = PlotterWFQ(
            # Metriche aggregate per i grafici a livello di cluster
            metrics_base_agg=simulator_base.metrics_agg,
            # metrics_prio_agg=simulator_prio.metrics_agg,
            metrics_wfq_agg=simulator_wfq.metrics_agg,

            # Metriche per-worker per l'analisi degli hotspot
            metrics_per_worker_base=simulator_base.metrics_per_worker,
            # metrics_per_worker_prio=simulator_prio.metrics_per_worker,
            metrics_per_worker_wfq=simulator_wfq.metrics_per_worker,

            config_module=config
        )

        print(f"\n--- Generazione grafici per la replica {i+1} ---")
        plotter.generate_final_dashboards(
            output_dir=output_folder,
            run_prefix=f"final_repl_{i+1}",
            peak_start=INIZIO_PICCO,
            peak_end=FINE_PICCO,
            base_load=CARICO_BASE,
            peak_load=CARICO_PICCO
        )

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

if __name__ == "__main__":
    main_wfq_analysis()