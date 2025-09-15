# File: main_verifica.py (VERSIONE CORRETTA)

import numpy as np
import math
from src import config
from src.simulation.simulator import Simulator
from src.utils.metrics import Metrics
from src.utils.lehmer_rng import LehmerRNG as RNGManager

def calculate_mmc_metrics(lam: float, mu: float, c: int) -> dict:
    """
    Calcola le metriche di performance teoriche per una coda M/M/c.
    """
    rho = lam / (c * mu)
    if rho >= 1.0:
        return {'E_W': float('inf'), 'E_T': float('inf'), 'rho': rho}

    p0_sum_part = sum([(c * rho)**n / math.factorial(n) for n in range(c)])
    p0_frac_part = (c * rho)**c / (math.factorial(c) * (1 - rho))
    P0 = 1 / (p0_sum_part + p0_frac_part)
    E_W = (p0_frac_part * P0) / (c * mu * (1 - rho))
    E_S = 1 / mu
    E_T = E_W + E_S
    return {'E_W': E_W, 'E_T': E_T, 'rho': rho}


def main_verification():
    """
    Esegue una simulazione di verifica confrontando il simulatore in un caso
    semplificato (M/M/c) con i risultati analitici.
    """
    print("\n" + "="*30 + " AVVIO SCRIPT DI VERIFICA " + "="*30)
    print("Obiettivo: Verificare il modello Rete di Code (Baseline) contro la teoria M/M/c.")

    print("\n--- [1/4] Configurazione del Caso Semplificato ---")


    # Con lambda=85 e c=24, la capacità totale (c*mu) deve essere > 85.
    # c * mu > 85  =>  24 * mu > 85  => mu > 3.54
    # Scegliamo un mu = 4.0, che corrisponde a E[S] = 1/4 = 0.25s.

    NUM_PODS_FISSO = 24
    LAMBDA = 85.0
    E_S_TEORICO = 0.25  # <-- MODIFICA PRINCIPALE: Da 0.4 a 0.25 per rendere il sistema stabile
    MU_EXPONENTIAL = 1.0 / E_S_TEORICO
    SIM_DURATION = 20000

    rho_atteso = LAMBDA / (NUM_PODS_FISSO * MU_EXPONENTIAL)
    print(f"Modello target: M/M/c con c={NUM_PODS_FISSO}, lambda={LAMBDA}, mu={MU_EXPONENTIAL:.2f}")
    print(f"Utilizzazione attesa (rho): {rho_atteso:.4f} (deve essere < 1)")

    config.HPA_ENABLED = False
    config.NUM_WORKERS = 1
    config.INITIAL_PODS = NUM_PODS_FISSO
    config.MIN_PODS = NUM_PODS_FISSO
    config.MAX_PODS = NUM_PODS_FISSO

    for req_type in config.SERVICE_TIME_CONFIG:
        config.SERVICE_TIME_CONFIG[req_type] = {
            "dist": "exponential",
            "params": {"scale": E_S_TEORICO}
        }

    print(f"\n--- [2/4] Esecuzione Simulazione (durata: {SIM_DURATION}s) ---")
    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    sim_streams, _ = rng_manager.get_replication_streams()

    metrics = Metrics(config_module=config)
    simulator = Simulator(
        config_module=config, metrics=metrics,
        arrival_rng=sim_streams['arrivals'], choice_rng=sim_streams['choice'],
        service_rng=sim_streams['service'], lambda_function=lambda t: LAMBDA
    )
    simulator.run(simulation_duration=SIM_DURATION)
    print("Simulazione completata.")

    print("\n--- [3/4] Calcolo Risultati Teorici e da Simulazione ---")

    teoria = calculate_mmc_metrics(lam=LAMBDA, mu=MU_EXPONENTIAL, c=NUM_PODS_FISSO)

    sim_E_T = metrics.global_response_times_welford.mean.item()
    sim_E_W = metrics.global_wait_times_welford.mean.item()

    error_E_T = abs(sim_E_T - teoria['E_T']) / teoria['E_T'] * 100
    error_E_W = abs(sim_E_W - teoria['E_W']) / teoria['E_W'] * 100

    print("\n" + "="*30 + " RISULTATI DELLA VERIFICA " + "="*30)
    print(f"{'Metrica':<25} | {'Valore Teorico':<20} | {'Valore Simulato':<20} | {'Errore %':<10}")
    print("-" * 80)
    print(f"{'Utilizzazione (rho)':<25} | {teoria['rho']:<20.4f} | {'N/A':<20} | {'N/A':<10}")
    print(f"{'Tempo Risposta Medio (E[T])':<25} | {teoria['E_T']:<20.4f} | {sim_E_T:<20.4f} | {error_E_T:<9.2f}%")
    print(f"{'Tempo Attesa Medio (E[W])':<25} | {teoria['E_W']:<20.4f} | {sim_E_W:<20.4f} | {error_E_W:<9.2f}%")
    print("-" * 80)

    print("\n--- [4/4] Conclusione della Verifica ---")
    if error_E_T < 5.0 and error_E_W < 5.0:
        print("VERIFICA SUPERATA: L'errore tra modello analitico e simulazione è inferiore al 5%.")
        print("L'implementazione del simulatore di base è considerata corretta.")
    else:
        print("VERIFICA FALLITA: L'errore tra modello analitico e simulazione è superiore al 5%.")
        print("Controllare la logica del simulatore o i parametri di verifica.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main_verification()