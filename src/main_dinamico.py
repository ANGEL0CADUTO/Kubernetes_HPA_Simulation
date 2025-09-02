import numpy as np
import os
from src import config
from src.simulation.simulator import Simulator
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter import Plotter
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_dynamic_analysis():
    print("="*80)
    print("FASE 2: ANALISI DEL SISTEMA CON CARICO DINAMICO (PICCO DI TRAFFICO)")
    print("="*80)

    NUM_REPLICATIONS = 1

    # --- FIX SULL'ESPERIMENTO: Carico base sensato ---
    CARICO_BASE = 6
    CARICO_PICCO = 85
    INIZIO_PICCO = 200
    DURATA_PICCO = 300
    FINE_PICCO = INIZIO_PICCO + DURATA_PICCO

    def lambda_con_picco(t: float) -> float:
        if INIZIO_PICCO <= t < FINE_PICCO: return CARICO_PICCO
        return CARICO_BASE

    config.SIMULATION_TIME = 700

    print(f"\nConfigurazione Esperimento Corretta:")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s")
    print(f" - Carico Base: {CARICO_BASE} req/s")
    print(f" - Picco di Carico: {CARICO_PICCO} req/s (dal {INIZIO_PICCO}s al {FINE_PICCO}s)")
    print(f" - Numero Repliche: {NUM_REPLICATIONS}")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        # --- FIX FONDAMENTALE SUI SEED ---
        # Per applicare correttamente i Common Random Numbers (CRN), dobbiamo
        # assicurarci che entrambi i simulatori partano con lo stesso stato del generatore.
        # Il modo più semplice è creare un nuovo set di stream per ogni simulatore,
        # ma partendo dallo stesso seed di replica.

        # 1. Generiamo il seed master per QUESTA replica
        master_seed_replica = rng_manager._master_rng._next_seed()

        # 2. Creiamo i generatori per la simulazione BASELINE
        rng_manager_base = RNGManager(master_seed=master_seed_replica)
        streams_base, seed_base = rng_manager_base.get_replication_streams()

        # 3. Creiamo i generatori per la simulazione a PRIORITA'
        #    Ripartendo dallo STESSO seed di replica, otterremo la stessa identica sequenza.
        rng_manager_prio = RNGManager(master_seed=master_seed_replica)
        streams_prio, seed_prio = rng_manager_prio.get_replication_streams()

        print(f"--- Replica {i+1} utilizzerà il SEED master: {seed_base} per entrambi i sistemi ---")

        # --- ESECUZIONE BASELINE ---
        metrics_base = Metrics(config)
        simulator_base = Simulator(config, metrics_base, streams_base['arrivals'],
                                   streams_base['choice'], streams_base['service'],
                                   lambda_con_picco)
        print("\n--- Esecuzione Modello Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # --- ESECUZIONE MIGLIORATA ---
        metrics_prio = MetricsWithPriority(config)
        simulator_prio = SimulatorWithPriority(config, metrics_prio, streams_prio['arrivals'],
                                               streams_prio['choice'], streams_prio['service'],
                                               lambda_con_picco)
        print("\n--- Esecuzione Modello Migliorato (Priorità) ---")
        simulator_prio.run(config.SIMULATION_TIME)

        output_folder = f"output/dynamic_analysis/replica_{i+1}"
        plotter = Plotter(metrics_base, metrics_prio, config)
        plotter.plot_dynamic_load_dashboard(output_dir=output_folder,
                                            run_prefix=f"dynamic_repl_{i+1}",
                                            peak_start=INIZIO_PICCO,
                                            peak_end=FINE_PICCO)

        plotter.plot_dynamic_analysis_dashboards(output_dir=output_folder,
                                                 run_prefix=f"dynamic_repl_{i+1}",
                                                 peak_start=INIZIO_PICCO,
                                                 peak_end=FINE_PICCO,
                                                 base_load=CARICO_BASE,
                                                 peak_load=CARICO_PICCO)

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

if __name__ == "__main__":
    main_dynamic_analysis()