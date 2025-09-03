

import numpy as np
import math
from src import config
from src.simulation.simulator import Simulator
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.simulation.simulator_sps import SimulatorSPS
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from src.utils.lehmer_rng import LehmerRNG as RNGManager
from src.config import Priority, RequestType

def run_baseline_sim(c: int, lam: float, mu: float, sim_duration: int, streams: dict) -> float:
    """Esegue la simulazione baseline e restituisce E[T] globale."""
    config.HPA_ENABLED = False; config.NUM_WORKERS = 1; config.INITIAL_PODS = c
    config.MIN_PODS = c; config.MAX_PODS = c
    for rt in config.SERVICE_TIME_CONFIG:
        config.SERVICE_TIME_CONFIG[rt] = {"dist": "exponential", "params": {"scale": 1.0 / mu}}
    metrics = Metrics(config_module=config)
    simulator = Simulator(config, metrics, streams['arrivals'], streams['choice'],
                          streams['service'], lambda t: lam)
    simulator.run(simulation_duration=sim_duration)
    return metrics.global_response_times_welford.mean.item()

def run_sps_sim(c: int, lam: float, mu: float, sim_duration: int, streams: dict) -> float:
    """Esegue la simulazione SPS e restituisce E[T] globale."""
    config.HPA_ENABLED = False; config.NUM_WORKERS = 1; config.INITIAL_PODS = c
    config.MIN_PODS = c; config.MAX_PODS = c
    for rt in config.SERVICE_TIME_CONFIG:
        config.SERVICE_TIME_CONFIG[rt] = {"dist": "exponential", "params": {"scale": 1.0 / mu}}
    metrics = MetricsWithPriority(config_module=config)
    simulator = SimulatorSPS(config, metrics, streams['arrivals'], streams['choice'],
                             streams['service'], lambda t: lam, timeouts_enabled=False)
    simulator.run(simulation_duration=sim_duration)
    return metrics.global_welford_response.mean.item()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERIFICA DELLA LEGGE DI CONSERVAZIONE: BASELINE (FIFO) vs. SPS")
    print("="*80)

    # --- 1. CONFIGURAZIONE DELL'ESPERIMENTO ---
    print("\n--- [1/3] Configurazione del Caso di Test (M/M/24, rho=0.885) ---")
    NUM_PODS = 24
    MU = 4.0
    LAMBDA = 85.0
    SIM_DURATION = 50000 # Simulazione lunga per risultati stabili

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    common_streams, seed = rng_manager.get_replication_streams()
    print(f"Usando stream comuni generati dal seed: {seed}")

    # --- 2. ESECUZIONE DELLE SIMULAZIONI ---
    print("\n--- [2/3] Esecuzione delle due simulazioni ---")
    print("Esecuzione del sistema Baseline (FIFO)...")
    e_t_baseline = run_baseline_sim(NUM_PODS, LAMBDA, MU, SIM_DURATION, common_streams)

    print("Esecuzione del sistema SPS...")
    # Dobbiamo creare un nuovo set di stream identico per il SPS
    rng_manager_sps = RNGManager(master_seed=config.LEHMER_SEED)
    streams_sps, _ = rng_manager_sps.get_replication_streams()
    e_t_sps = run_sps_sim(NUM_PODS, LAMBDA, MU, SIM_DURATION, streams_sps)

    # --- 3. CONFRONTO DEI RISULTATI ---
    print("\n--- [3/3] Analisi dei Risultati ---")

    diff = abs(e_t_baseline - e_t_sps)
    error_perc = (diff / e_t_baseline) * 100 if e_t_baseline > 0 else 0

    print("\n" + "="*60)
    print("RISULTATI DELLA VERIFICA DI CONSERVAZIONE")
    print("-" * 60)
    print(f"{'Sistema':<20} | {'E[T] Globale Similato':<25}")
    print("-" * 60)
    print(f"{'Baseline (FIFO)':<20} | {e_t_baseline:<25.4f}")
    print(f"{'SPS':<20} | {e_t_sps:<25.4f}")
    print("-" * 60)
    print(f"Differenza Assoluta: {diff:.4f}")
    print(f"Differenza Percentuale: {error_perc:.2f}%")
    print("="*60)

    print("\n--- Conclusione della Verifica ---")
    if error_perc < 5.0: # Usiamo una tolleranza più generosa per il debug
        print("VERIFICA SUPERATA: La Legge di Conservazione è rispettata.")
    else:
        print("VERIFICA FALLITA: C'è un bug fondamentale nel simulatore SPS.")
        print("Il tempo di risposta medio globale è significativamente diverso, il che è teoricamente impossibile.")
    print("="*80 + "\n")