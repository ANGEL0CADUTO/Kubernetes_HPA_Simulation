import math
import os
import sys
import simpy
from types import ModuleType
from enum import Enum # <-- IMPORTAZIONE AGGIUNTA



# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class RequestType(Enum):
    NAVIGATION = "NAVIGATION"
    SEARCH = "SEARCH"
    OTHER = "OTHER"


class config:
    RequestType = RequestType
    LEHMER_SEED = 12345
    SERVICE_TIME_CONFIG = {
        RequestType.NAVIGATION: {"dist": "exponential", "params": {"scale": 0.05}}
    }

class Request:
    def __init__(self, request_id, req_type, arrival_time, timeout, service_time):
        self.request_id = request_id
        self.req_type = req_type.value # Usiamo .value se passiamo un membro della Enum
        self.arrival_time = arrival_time
        self.timeout = timeout
        self.service_time = service_time

class Welford:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self._M2 = 0.0
    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        delta2 = new_value - self.mean
        self._M2 += delta * delta2

class Metrics:
    def __init__(self, config_module):
        self.config = config_module
        self.global_response_times_welford = Welford()
        self.global_wait_times_welford = Welford()
    def record_request_generation(self, req_type): pass
    def record_request_metrics(self, completion_time, req_type, response_time, wait_time):
        self.global_response_times_welford.update(response_time)
        self.global_wait_times_welford.update(wait_time)

class LehmerRNG:
    def __init__(self, seed): self.seed = seed
    def exponential(self, scale): return -scale * math.log(self._rand())
    def _rand(self):
        self.seed = (self.seed * 48271) % 2147483647
        return self.seed / 2147483647

class RNGManager:
    def __init__(self, master_seed): self.master_seed = master_seed
    def get_replication_streams(self):
        return {'arrivals': LehmerRNG(self.master_seed), 'service': LehmerRNG(self.master_seed + 1)}, None

class PodService:
    def __init__(self, service_rng, config_module):
        self.service_rng = service_rng
        self.config = config_module
    def get_service_time(self, req_type):
        conf = self.config.SERVICE_TIME_CONFIG[req_type]
        return self.service_rng.exponential(conf['params']['scale'])

# --- Fine Sezione Stub/Mock ---


# ==============================================================================
# CLASSE DI SIMULAZIONE DEDICATA E ISOLATA (INVARIATA)
# ==============================================================================
class MMcVerificationSimulator:
    def __init__(self, config_module, metrics_instance, arrival_rng, service_rng, lambda_rate, num_pods):
        self.config = config_module
        self.metrics = metrics_instance
        self.env = simpy.Environment()
        self.arrival_rng = arrival_rng
        self.service_rng = service_rng
        self.lambda_rate = lambda_rate
        self.num_pods = num_pods
        self.service = PodService(service_rng, config_module)
        self.request_queue = simpy.Store(self.env)

    def request_generator(self):
        req_id_counter = 0
        while True:
            time_to_next = self.arrival_rng.exponential(1.0 / self.lambda_rate)
            yield self.env.timeout(time_to_next)
            chosen_type = config.RequestType.NAVIGATION # Scegliamo un tipo fisso
            service_time = self.service.get_service_time(chosen_type)
            req_id_counter += 1
            new_request = Request(
                request_id=req_id_counter, req_type=chosen_type,
                arrival_time=self.env.now, timeout=float('inf'),
                service_time=service_time
            )
            self.metrics.record_request_generation(chosen_type)
            self.request_queue.put(new_request)

    def pod_worker(self, pod_id):
        while True:
            request = yield self.request_queue.get()
            wait_time = self.env.now - request.arrival_time
            yield self.env.timeout(request.service_time)
            completion_time = self.env.now
            response_time = completion_time - request.arrival_time
            self.metrics.record_request_metrics(completion_time, request.req_type, response_time, wait_time)

    def run(self, simulation_duration: float):
        self.env.process(self.request_generator())
        for i in range(self.num_pods):
            self.env.process(self.pod_worker(i))
        self.env.run(until=simulation_duration)



def calculate_mmc_metrics(lam: float, mu: float, c: int) -> dict:
    if c * mu <= lam:
        return {k: float('inf') for k in ['rho', 'P0', 'Pq', 'Lq', 'E_Wq', 'E_T']}
    rho = lam / (c * mu)
    a = lam / mu
    p0_sum_part = sum([(a**n) / math.factorial(n) for n in range(c)])
    p0_frac_part = (a**c) / (math.factorial(c) * (1 - rho))
    P0 = 1 / (p0_sum_part + p0_frac_part)
    Pq = ((a**c) / (math.factorial(c) * (1 - rho))) * P0
    Lq = Pq * rho / (1 - rho)
    E_Wq = Lq / lam
    E_T = E_Wq + (1 / mu)
    return {'lam': lam, 'mu': mu, 'c': c, 'a': a, 'rho': rho, 'P0': P0, 'Pq': Pq, 'Lq': Lq, 'E_Wq': E_Wq, 'E_T': E_T}


def main_verifica_analitica():
    print("\n" + "="*80)
    print("                 FASE DI VERIFICA ANALITICA DEL MODELLO M/M/c")
    print("="*80)
    print("\n--- [FASE 1: PARAMETRI DI INPUT DEL SISTEMA] ---\n")

    NUM_PODS_FISSO = 8
    LAMBDA = 150.0
    MEAN_SERVICE_TIME = 0.05
    MU = 1.0 / MEAN_SERVICE_TIME
    SIM_DURATION = 50000

    print(f"  - Tasso di arrivo (λ):         {LAMBDA:.2f} req/s")
    print(f"  - Tempo di servizio medio (E[S]): {MEAN_SERVICE_TIME:.4f} s")
    print(f"  - Tasso di servizio (μ):         1 / {MEAN_SERVICE_TIME:.4f} = {MU:.2f} req/s per pod")
    print(f"  - Numero di Pod/Server (c):      {NUM_PODS_FISSO}")

    rho_atteso = LAMBDA / (NUM_PODS_FISSO * MU)
    print(f"\nUtilizzazione attesa (ρ = λ / (c * μ)): {rho_atteso:.4f}")
    if rho_atteso >= 1:
        print("\nERRORE: Il sistema configurato è instabile (ρ >= 1). Interruzione.")
        return

    print("\n--- [FASE 2: CALCOLI ANALITICI (TEORIA DELLE CODE)] ---\n")
    teoria = calculate_mmc_metrics(lam=LAMBDA, mu=MU, c=NUM_PODS_FISSO)
    print("1. Carico offerto al sistema (a):")
    print(f"   a = λ * E[S] = {teoria['lam']:.2f} * {1/teoria['mu']:.4f} = {teoria['a']:.6f} Erlang\n")
    print("2. Utilizzazione del sistema (ρ):")
    print(f"   ρ = a / c = {teoria['a']:.6f} / {teoria['c']} = {teoria['rho']:.6f}\n")
    print("3. Probabilità che il sistema sia vuoto (P0):")
    print(f"   P0 = [ Σ_{{n=0}}^{{c-1}} (a^n / n!) + (a^c / c!) * (1 / (1 - ρ)) ]^-1 = {teoria['P0']:.6f}\n")
    print("4. Probabilità di attesa in coda (Pq - Formula di Erlang C):")
    print(f"   Pq = (a^c / (c! * (1 - ρ))) * P0 = {teoria['Pq']:.6f}\n")
    print("5. Numero medio di richieste in coda (Lq o Nq):")
    print(f"   Lq = (Pq * ρ) / (1 - ρ) = ({teoria['Pq']:.6f} * {teoria['rho']:.6f}) / (1 - {teoria['rho']:.6f}) = {teoria['Lq']:.6f} richieste\n")
    print("6. Tempo di attesa medio in coda (E[Wq] o E[Tq]):")
    print(f"   E[Wq] = Lq / λ = {teoria['Lq']:.6f} / {teoria['lam']:.2f} = {teoria['E_Wq']:.6f} sec\n")
    print("7. Tempo di risposta medio totale (E[T] o E[Tr]):")
    print(f"   E[T] = E[Wq] + E[S] = {teoria['E_Wq']:.6f} + {1/teoria['mu']:.4f} = {teoria['E_T']:.6f} sec\n")

    print("\n" + "-"*50)
    print(f"--- [FASE 3: ESECUZIONE SIMULAZIONE (Durata: {SIM_DURATION}s)] ---")
    print("-" * 50 + "\n")

    local_config = ModuleType("local_config")
    for key, value in config.__dict__.items():
        if not key.startswith("__"): setattr(local_config, key, value)

    local_config.SERVICE_TIME_CONFIG = {}
    for req_type in config.RequestType: # Ora questo ciclo funziona
        local_config.SERVICE_TIME_CONFIG[req_type] = {"dist": "exponential", "params": {"scale": MEAN_SERVICE_TIME}}

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    sim_streams, _ = rng_manager.get_replication_streams()
    metrics_instance = Metrics(local_config)
    simulator = MMcVerificationSimulator(
        local_config, metrics_instance, sim_streams['arrivals'],
        sim_streams['service'], LAMBDA, NUM_PODS_FISSO
    )
    simulator.run(simulation_duration=SIM_DURATION)
    print("Simulazione completata.\n")

    print("\n--- [FASE 4: CONFRONTO RISULTATI TEORICI E SIMULATI] ---\n")
    sim_E_T = metrics_instance.global_response_times_welford.mean
    sim_E_W = metrics_instance.global_wait_times_welford.mean
    sim_E_S = sim_E_T - sim_E_W
    sim_rho = LAMBDA * sim_E_S / NUM_PODS_FISSO
    sim_Lq = LAMBDA * sim_E_W

    error_perc = abs(sim_E_T - teoria['E_T']) / teoria['E_T'] * 100

    print("="*80)
    print("                       TABELLA DI VERIFICA FINALE")
    print("="*80)
    print(f"{'Metrica':<28} | {'Valore Teorico (Analitico)':<30} | {'Valore Similato (Empirico)':<30}")
    print("-" * 80)
    print(f"{'Utilizzazione (ρ)':<28} | {teoria['rho']:<30.6f} | {sim_rho:<30.6f}")
    print(f"{'Numero Richieste in Coda (Nq)':<28} | {teoria['Lq']:<30.6f} | {sim_Lq:<30.6f}")
    print(f"{'Tempo Attesa in Coda (E[Tq])':<28} | {teoria['E_Wq']:<30.6f} sec | {sim_E_W:<30.6f} sec")
    print(f"{'Tempo Risposta Totale (E[T])':<28} | {teoria['E_T']:<30.6f} sec | {sim_E_T:<30.6f} sec")
    print(f"{'Tempo Servizio (E[S])':<28} | {1/MU:<30.6f} sec | {sim_E_S:<30.6f} sec")
    print("=" * 80)
    print(f"\n=> Errore Percentuale sul Tempo di Risposta Medio E[T]: {error_perc:.2f}%")

    if error_perc < 5.0:
        print("\nRISULTATO: VERIFICA SUPERATA. I risultati empirici sono consistenti con il modello analitico.")
    else:
        print("\nRISULTATO: VERIFICA FALLITA. Discrepanza significativa tra teoria e simulazione.")
    print("="*80)


if __name__ == "__main__":
    main_verifica_analitica()