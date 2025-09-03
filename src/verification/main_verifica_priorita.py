# File: main_verifica_priorita.py (LA VERIFICA FINALE A DUE STADI)

import numpy as np
import math
from src import config
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.metrics_with_priority import MetricsWithPriority
from src.utils.lehmer_rng import LehmerRNG as RNGManager
from src.config import Priority, RequestType

# ==============================================================================
# SEZIONE TEORICA: SOLO LA FORMULA M/M/1 CHE SAPPIAMO ESSERE CORRETTA
# ==============================================================================
def calculate_mm1_priority_metrics(lambdas: dict, mu: float) -> dict:
    priorities = sorted(lambdas.keys(), key=lambda p: p.value)
    E_S = 1.0 / mu; E_S2 = 2 * (E_S**2)
    W0 = 0.5 * sum(lam * E_S2 for lam in lambdas.values())
    rhos = {p: l / mu for p, l in lambdas.items()}
    sigmas = {}; current_sigma = 0.0
    for p in priorities:
        current_sigma += rhos.get(p, 0.0); sigmas[p] = current_sigma
    if sum(rhos.values()) >= 1.0: return {'E_W_per_prio': {p: float('inf') for p in priorities}}
    E_W_per_prio = {}
    for i, p in enumerate(priorities):
        sigma_k = sigmas[p]; sigma_k_minus_1 = sigmas[priorities[i-1]] if i > 0 else 0
        E_W_per_prio[p] = W0 / ((1 - sigma_k_minus_1) * (1 - sigma_k))
    return {'E_W_per_prio': E_W_per_prio}

# ==============================================================================
# FUNZIONE HELPER PER ESEGUIRE LE SIMULAZIONI DI TEST
# ==============================================================================
def run_test_scenario(num_pods: int, total_lambda: float, lambdas_dist: dict, sim_duration: int) -> dict:
    MU = 4.0
    config.HPA_ENABLED = False; config.NUM_WORKERS = 1; config.INITIAL_PODS = num_pods
    config.MIN_PODS = num_pods; config.MAX_PODS = num_pods
    for rt in config.SERVICE_TIME_CONFIG:
        config.SERVICE_TIME_CONFIG[rt] = {"dist": "exponential", "params": {"scale": 1.0 / MU}}

    prio_to_req = {p: [] for p in Priority};
    for rt, p in config.REQUEST_TYPE_TO_PRIORITY.items(): prio_to_req[p].append(rt)
    for p, rts in prio_to_req.items():
        if not rts: continue
        lambda_prio = total_lambda * lambdas_dist[p]
        prob = (lambda_prio / total_lambda) / len(rts) if total_lambda > 0 else 0
        for rt in rts: config.TRAFFIC_PROFILE[rt] = prob

    rng = RNGManager(master_seed=config.LEHMER_SEED)
    streams, _ = rng.get_replication_streams()
    metrics = MetricsWithPriority(config)
    sim = SimulatorWithPriority(config, metrics, streams['arrivals'], streams['choice'],
                                streams['service'], lambda t: total_lambda, timeouts_enabled=False)
    sim.run(sim_duration)

    stats = metrics.get_welford_statistics()
    sim_E_W = {}
    for p in Priority:
        means, counts = [],[]
        for rt in prio_to_req.get(p,[]):
            s = stats['by_req_type'].get(rt)
            if s and s['count'] > 0:
                means.append(s['wait_time']['mean'].item()); counts.append(s['count'])
        if counts: sim_E_W[p] = np.average(means, weights=counts)
        else: sim_E_W[p] = 0
    return sim_E_W, MU

# ==============================================================================
# ORCHESTRATORE PRINCIPALE DELLA VERIFICA
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("VERIFICA FORMALE DELLO SCHEDULING A PRIORITA' IN DUE STADI")
    print("="*80)

    # --- STADIO 1: TEST DI CORRETTEZZA ANALITICA (M/M/1) ---
    print("\n--- STADIO 1: Test di Correttezza Analitica su M/M/1 (rho=0.95) ---")
    c1, rho1 = 1, 0.95
    dist1 = {Priority.HIGH: 0.6, Priority.MEDIUM: 0.3, Priority.LOW: 0.1}
    sim_E_W1, mu1 = run_test_scenario(c1, rho1 * c1 * 4.0, dist1, 100000)
    lambdas1 = {p: rho1 * c1 * 4.0 * dist for p, dist in dist1.items()}
    teoria1 = calculate_mm1_priority_metrics(lambdas1, mu1)

    print(f"\n{'Priorità':<10} | {'E[W] Teorico':<15} | {'E[W] Similato':<15} | {'Errore %':<10}")
    print("-" * 60)
    verifica_stadio1 = True
    for p in Priority:
        teorico = teoria1['E_W_per_prio'].get(p, 0.0); simulato = sim_E_W1.get(p, 0.0)
        errore = abs(simulato - teorico) / teorico * 100 if teorico > 0 else 0
        if errore > 5: verifica_stadio1 = False
        print(f"{p.name:<10} | {teorico:<15.4f} | {simulato:<15.4f} | {errore:<9.2f}%")
    print("-" * 60)

    # --- STADIO 2: TEST DI CORRETTEZZA ORDINALE (M/M/c) ---
    print("\n--- STADIO 2: Test di Correttezza Ordinale su M/M/c ad Alta Contesa (rho=0.95) ---")
    c2, rho2 = 2, 0.95
    dist2 = {Priority.HIGH: 0.6, Priority.MEDIUM: 0.3, Priority.LOW: 0.1}
    sim_E_W2, _ = run_test_scenario(c2, rho2 * c2 * 4.0, dist2, 50000)

    print(f"\n{'Priorità':<10} | {'E[W] Similato':<15}")
    print("-" * 30)
    sorted_priorities = sorted(sim_E_W2.items(), key=lambda item: item[0].value)
    for p, wait_time in sorted_priorities:
        print(f"{p.name:<10} | {wait_time:<15.4f}")
    print("-" * 30)

    high_wait = sim_E_W2.get(Priority.HIGH, -1)
    medium_wait = sim_E_W2.get(Priority.MEDIUM, -1)
    low_wait = sim_E_W2.get(Priority.LOW, -1)
    verifica_stadio2 = (high_wait < medium_wait) and (medium_wait < low_wait)

    # --- CONCLUSIONE FINALE ---
    print("\n" + "="*80)
    print("CONCLUSIONE FINALE DELLA VERIFICA")
    print(f"Stadio 1 (Correttezza Analitica M/M/1): {'SUPERATO' if verifica_stadio1 else 'FALLITO'}")
    print(f"Stadio 2 (Correttezza Ordinale M/M/c):  {'SUPERATO' if verifica_stadio2 else 'FALLITO'}")
    if verifica_stadio1 and verifica_stadio2:
        print("\nRISULTATO: L'implementazione dello scheduling a priorità è considerata VERIFICATA e CORRETTA.")
    else:
        print("\nRISULTATO: L'implementazione presenta ancora problemi.")