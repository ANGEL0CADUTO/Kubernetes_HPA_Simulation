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
    print("ANALISI SCENARIO 'BLACK FRIDAY' (ARCHITETTURA A SILOS)")
    print("="*80)

    # <-- MODIFICA: Imposta il numero di repliche che vuoi visualizzare -->
    NUM_REPLICATIONS = 5
    config.SIMULATION_TIME = 500

    # ... (la definizione di lambda_black_friday rimane la stessa) ...
    t_notte_fine = config.SIMULATION_TIME * 0.15
    t_mattina_fine = config.SIMULATION_TIME * 0.30
    t_picco1_fine = config.SIMULATION_TIME * 0.40
    t_pomeriggio_fine = config.SIMULATION_TIME * 0.70
    t_picco2_fine = config.SIMULATION_TIME * 0.80
    CARICO_NOTTE = 10; CARICO_MATTINA = 30; CARICO_PICCO_1 = 170
    CARICO_POMERIGGIO = 85; CARICO_PICCO_2 = 300
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

    # <-- MODIFICA: Dizionario per raccogliere tutti i risultati -->
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

        # <-- MODIFICA: Salva i risultati di questa replica -->
        all_results[i] = {
            'baseline': simulator_base.metrics_agg,
            'wfq': simulator_wfq.metrics_agg,
            'seed': rep_seed
        }

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

    # <-- MODIFICA: Chiama il plotter una sola volta alla fine, con tutti i risultati -->
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