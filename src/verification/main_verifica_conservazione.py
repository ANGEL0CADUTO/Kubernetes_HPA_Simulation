import os
import sys

# Aggiungi il percorso radice del progetto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import config
from src.simulation.simulator_blackfriday import SimulatorBlackFridayBaseline, SimulatorBlackFriday as SimulatorBlackFridayDWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def main_verifica_conservazione():
    print("\n" + "="*80)
    print("FASE 2: VERIFICA LEGGE DI CONSERVAZIONE (FIFO vs. DWFQ)")
    print("="*80)

    # --- 1. CONFIGURAZIONE ESPERIMENTO ---
    print("\n--- [1/3] Configurazione del sistema M/G/c ---")

    NUM_PODS_FISSO = 8
    LAMBDA = 150.0 # Carico elevato per stressare lo scheduler
    SIM_DURATION = 900

    # Usiamo la configurazione originale con tempi di servizio lognormali
    config.HPA_ENABLED = False
    config.NUM_WORKERS = 1
    config.INITIAL_PODS_PER_WORKER = NUM_PODS_FISSO
    config.MIN_PODS = NUM_PODS_FISSO
    config.MAX_PODS = NUM_PODS_FISSO

    # --- 2. ESECUZIONE SIMULAZIONI CON CRN ---
    print("\n--- [2/3] Esecuzione simulazioni con Common Random Numbers ---")
    master_seed = config.LEHMER_SEED
    print(f"Usando stream comuni generati dal master seed: {master_seed}")

    # Esecuzione Baseline (FIFO)
    print("Esecuzione del sistema Baseline (FIFO)...")
    rng_base = RNGManager(master_seed=master_seed)
    streams_base, _ = rng_base.get_replication_streams()
    sim_base = SimulatorBlackFridayBaseline(
        config, Metrics, streams_base['arrivals'], streams_base['choice'],
        streams_base['service'], lambda t: LAMBDA
    )
    sim_base.run(SIM_DURATION)
    e_t_baseline = sim_base.metrics_agg.global_response_times_welford.mean

    # Esecuzione DWFQ
    print("Esecuzione del sistema DWFQ...")
    rng_wfq = RNGManager(master_seed=master_seed)
    streams_wfq, _ = rng_wfq.get_replication_streams()
    sim_wfq = SimulatorBlackFridayDWFQ(
        config, MetricsWithPriority, streams_wfq['arrivals'], streams_wfq['choice'],
        streams_wfq['service'], lambda t: LAMBDA
    )
    sim_wfq.run(SIM_DURATION)
    e_t_wfq = sim_wfq.metrics_agg.global_welford_response.mean

    # --- 3. CONFRONTO RISULTATI ---
    print("\n--- [3/3] Confronto dei tempi di risposta medi globali ---")

    diff_abs = abs(e_t_baseline - e_t_wfq)
    error_perc = (diff_abs / e_t_baseline) * 100 if e_t_baseline > 0 else 0

    print("\n" + "="*60)
    print("RISULTATI DELLA VERIFICA DI CONSERVAZIONE")
    print("-" * 60)
    print(f"{'Sistema':<20} | {'E[T] Globale Similato':<25}")
    print(f"{'Baseline (FIFO)':<20} | {e_t_baseline:<25.4f}")
    print(f"{'DWFQ':<20} | {e_t_wfq:<25.4f}")
    print("-" * 60)
    print(f"Differenza Percentuale: {error_perc:.2f}%")
    print("="*60)

    if error_perc < 5.0:
        print("\nRISULTATO: VERIFICA SUPERATA. La Legge di Conservazione è rispettata.")
        print("L'implementazione DWFQ è considerata corretta (work-conserving).")
    else:
        print("\nRISULTATO: VERIFICA FALLITA. C'è una discrepanza, l'implementazione DWFQ potrebbe avere un bug.")

if __name__ == "__main__":
    main_verifica_conservazione()