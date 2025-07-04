import simpy
# --- MODIFICA CHIAVE: Import corretti da simpy.resources.store ---
from simpy.resources.store import PriorityStore, PriorityItem
from collections import defaultdict

from src.config import Priority
from src.model.request import PriorityRequest
from src.controller.hpa import HPA
from src.service.service import PodService

class SimulatorWithPriority:
    """
    Simulatore per lo scenario migliorato con una coda di priorità unica e robusta.
    Questo modello elimina il dispatcher e usa simpy.PriorityStore con PriorityItem
    per una gestione delle priorità efficiente e corretta.
    """

    class _Pod:
        def __init__(self, pod_id, process):
            self.id = pod_id
            self.process = process

    def __init__(self, config_module, metrics, rng):
        self.config = config_module
        self.metrics = metrics
        self.rng = rng
        self.env = simpy.Environment()
        self.service = PodService(rng, config_module)

        # Si usa una singola PriorityStore.
        self.request_queue = PriorityStore(self.env)

        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def request_generator(self):
        """Genera richieste e le mette nella PriorityStore usando PriorityItem."""
        req_id_counter = 0
        req_types = list(self.config.TRAFFIC_PROFILE.keys())
        req_probs = list(self.config.TRAFFIC_PROFILE.values())
        while True:
            time_to_next = self.rng.exponential(1.0 / self.config.TOTAL_ARRIVAL_RATE)
            yield self.env.timeout(time_to_next)
            req_id_counter += 1
            chosen_type = self.rng.choice(req_types, p=req_probs)
            assigned_priority = self.config.REQUEST_TYPE_TO_PRIORITY[chosen_type]
            service_time = self.service.get_service_time(chosen_type)
            type_timeout = self.config.REQUEST_TIMEOUTS[chosen_type]

            new_request = PriorityRequest(
                request_id=req_id_counter, req_type=chosen_type,
                arrival_time=self.env.now, priority=assigned_priority,
                service_time=service_time, timeout = type_timeout)

            self.metrics.record_request_generation(self.env.now, assigned_priority)
            print(f"{self.env.now:.2f} [Generator]: Richiesta {new_request.request_id} ({new_request.req_type.name} -> Priorità: {new_request.priority.name}) generata.")

            # --- MODIFICA CHIAVE: Inserimento usando PriorityItem ---
            # Si avvolge la richiesta in un PriorityItem.
            # Il primo argomento è la priorità (valore numerico), il secondo è l'oggetto.
            priority_value = assigned_priority.value
            yield self.request_queue.put(PriorityItem(priority_value, new_request))
            # --------------------------------------------------------

    def pod_worker(self, pod_id):
        """Processo che simula un Pod. Consuma PriorityItem dalla PriorityStore."""
        print(f"{self.env.now:.2f} [Pod {pod_id}]: Avviato e in attesa di lavoro.")
        try:
            while True:
                # --- MODIFICA CHIAVE: Get dalla PriorityStore ---
                # Il .get() ora restituisce un oggetto PriorityItem.
                priority_item = yield self.request_queue.get()
                # Estraiamo l'oggetto richiesta dall'attributo .item
                request = priority_item.item
                # -----------------------------------------------

                arrival_in_service = self.env.now
                wait_time = arrival_in_service - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento rich. {request.request_id} "
                      f"(Priorità: {request.priority.name}). Attesa: {wait_time:.4f}s")

                yield self.env.timeout(request.service_time)

                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento rich. {request.request_id}. "
                      f"Tempo di risposta: {response_time:.4f}s")
                self.metrics.record_request_metrics(completion_time, request, response_time, wait_time)
        except simpy.Interrupt:
            print(f"{self.env.now:.2f} [Pod {pod_id}]: Ricevuto segnale di stop, terminazione.")

    def metrics_recorder(self):
        """Registra le metriche di sistema a intervalli regolari."""
        while True:
            # --- MODIFICA CHIAVE: Itera sui PriorityItem nella coda ---
            queue_lengths_per_prio = defaultdict(int)
            # self.request_queue.items ora contiene PriorityItem
            for p_item in self.request_queue.items:
                req = p_item.item  # Estrai la richiesta effettiva
                queue_lengths_per_prio[req.priority] += 1

            total_queue_len = len(self.request_queue.items)
            pod_count = len(self.active_pods)

            self.metrics.record_system_metrics(self.env.now, pod_count, total_queue_len, queue_lengths_per_prio)
            # -------------------------------------------------------------
            yield self.env.timeout(1)

    def get_busy_pods_count(self):
        """Calcola il numero di pod attualmente occupati. Logica invariata."""
        num_pods_waiting_for_work = len(self.request_queue.get_queue)
        num_active_pods = len(self.active_pods)
        num_busy_pods = num_active_pods - num_pods_waiting_for_work
        return max(0, num_busy_pods)

    def scale_to(self, desired_replicas):
        """Gestisce lo scaling dei pod. Logica invariata."""
        current_replicas = len(self.active_pods)
        if desired_replicas > current_replicas:
            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator]: Aggiungo {num_to_add} Pods...")
            for _ in range(num_to_add):
                if self.available_pod_ids: pod_id = self.available_pod_ids.pop()
                else: pod_id = self.next_pod_id; self.next_pod_id += 1
                process = self.env.process(self.pod_worker(pod_id))
                self.active_pods.append(self._Pod(pod_id, process))
        elif desired_replicas < current_replicas:
            num_to_remove = current_replicas - desired_replicas
            print(f"{self.env.now:.2f} [Simulator]: Rimuovo {num_to_remove} Pods...")
            pods_to_remove = self.active_pods[-num_to_remove:]
            for pod in pods_to_remove:
                if pod.process.is_alive and not pod.process.triggered:
                    pod.process.interrupt()
                self.available_pod_ids.add(pod.id)
            self.active_pods = self.active_pods[:-num_to_remove]

    def run(self):
        """Avvia la simulazione."""
        print("--- Avvio Simulatore con PriorityStore ---")
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())

        self.scale_to(self.config.INITIAL_PODS)

        if self.config.HPA_ENABLED:
            HPA(self.env, self)

        self.env.run(until=self.config.SIMULATION_TIME)
        print("--- Simulazione con priorità Terminata ---")