


from collections import defaultdict
import simpy
from simpy.resources.store import PriorityStore, PriorityItem

from src.controller.hpa import HPA
from src.model.request import PriorityRequest
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler
# NUOVO: Importiamo il WorkerNode
from src.model.worker_node import WorkerNode

class SimulatorWithPriority:
    # MODIFICATO: Il Pod ora deve sapere a quale worker è assegnato
    class _Pod:
        def __init__(self, pod_id, process, worker_node_id):
            self.id = pod_id
            self.process = process
            self.worker_node_id = worker_node_id

    def __init__(self, config_module, metrics, arrival_rng, choice_rng, service_rng, lambda_function):
        self.config = config_module
        self.metrics = metrics
        self.env = simpy.Environment()
        self.arrival_rng = arrival_rng
        self.choice_rng = choice_rng
        self.lambda_function = lambda_function
        self.service = PodService(service_rng, config_module)
        self.traffic_profiler = DynamicTrafficProfiler(metrics, config_module)

        # RIMOSSO: La coda singola a priorità non esiste più
        # self.request_queue = PriorityStore(self.env)

        # NUOVO: Coda di ingresso per il dispatcher
        self.dispatcher_inbox = simpy.Store(self.env)

        # NUOVO: Creiamo i Worker Node. LA LORO CODA ORA E' UNA PRIORITYSTORE.
        self.worker_nodes = []
        for i in range(self.config.NUM_WORKERS):
            # Creiamo un worker e sovrascriviamo la sua coda con una PriorityStore
            worker = WorkerNode(self.env, i)
            worker.queue = PriorityStore(self.env)
            self.worker_nodes.append(worker)

        self.next_worker_idx_dispatch = 0
        self.next_worker_idx_scale = 0

        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def dispatcher_process(self):
        """
        NUOVO PROCESSO: Simula il dispatcher.
        Smista le richieste (PriorityItem) nelle code a priorità dei Worker Node.
        """
        print(f"0.00 [Dispatcher-Prio]: Avviato.")
        while True:
            # L'item che arriva è già un PriorityItem
            priority_item = yield self.dispatcher_inbox.get()
            request = priority_item.item

            target_worker = self.worker_nodes[self.next_worker_idx_dispatch]
            # Mettiamo il PriorityItem nella coda a priorità del worker
            target_worker.queue.put(priority_item)

            print(f"{self.env.now:.2f} [Dispatcher-Prio]: Richiesta {request.request_id} smistata al Worker {target_worker.id}.")

            self.next_worker_idx_dispatch = (self.next_worker_idx_dispatch + 1) % self.config.NUM_WORKERS

    def request_generator(self):
        req_id_counter = 0
        while True:
            # ... (logica di generazione identica alla versione precedente)
            current_arrival_rate = self.lambda_function(self.env.now)
            if current_arrival_rate <= 0: yield self.env.timeout(1); continue
            time_to_next = self.arrival_rng.exponential(1.0 / current_arrival_rate)
            yield self.env.timeout(time_to_next)
            req_types, req_probs = self.traffic_profiler.get_current_probabilities()
            chosen_type = self.choice_rng.choice(req_types, p=req_probs)
            service_time = self.service.get_service_time(chosen_type)
            assigned_priority = self.config.REQUEST_TYPE_TO_PRIORITY[chosen_type]
            type_timeout = self.config.REQUEST_TIMEOUTS[chosen_type]
            req_id_counter += 1
            new_request = PriorityRequest(
                request_id=req_id_counter, req_type=chosen_type,
                arrival_time=self.env.now, priority=assigned_priority,
                service_time=service_time, timeout=type_timeout
            )
            self.metrics.record_request_generation(self.env.now, assigned_priority, chosen_type)
            print(f"{self.env.now:.2f} [Generator-Prio]: Richiesta {new_request.request_id} (Prio: {new_request.priority.name}) generata.")
            self.env.process(self.timeout_watcher(new_request))

            # MODIFICATO: Mettiamo il PriorityItem nell'inbox del dispatcher
            self.dispatcher_inbox.put(PriorityItem(assigned_priority.value, new_request))

    def pod_worker(self, pod_id, worker_node: WorkerNode):
        print(f"{self.env.now:.2f} [Pod {pod_id} @ Worker {worker_node.id}]: Avviato.")
        while True:
            try:
                # MODIFICATO: Il pod estrae dalla CODA LOCALE A PRIORITA' del suo worker
                priority_item = yield worker_node.queue.get()
                request = priority_item.item

                # Marca la richiesta come "servita" (utile per tracciarne lo stato)
                request.is_serviced = True
                if request.timed_out:
                    print(f"{self.env.now:.2f} [Pod {pod_id}]: Scartata richiesta {request.request_id} (già scaduta).")
                    continue
                wait_time = self.env.now - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento rich. {request.request_id} (Prio: {request.priority.name}). Attesa: {wait_time:.4f}s")
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento rich. {request.request_id}. Tempo di risposta: {response_time:.4f}s")
                self.metrics.record_request_metrics(completion_time, request, response_time, wait_time)

            except simpy.Interrupt:
                break
        print(f"{self.env.now:.2f} [Pod {pod_id} @ Worker {worker_node.id}]: Terminato.")

    def metrics_recorder(self):
        while True:
            # MODIFICATO: La lunghezza totale e per priorità è la somma di tutte le code dei worker
            queue_lengths_per_prio = defaultdict(int)
            total_queue_len = 0
            for w in self.worker_nodes:
                total_queue_len += len(w.queue.items)
                for p_item in w.queue.items:
                    req = p_item.item
                    queue_lengths_per_prio[req.priority] += 1

            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(self.env.now, pod_count, total_queue_len, queue_lengths_per_prio)
            yield self.env.timeout(1)

    # La logica di scaling è identica al simulatore baseline
    def scale_to(self, desired_replicas):
        current_replicas = len(self.active_pods)
        if desired_replicas > current_replicas:
            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator-Prio]: Aggiungo {num_to_add} Pods...")
            for _ in range(num_to_add):
                target_worker = self.worker_nodes[self.next_worker_idx_scale]
                self.next_worker_idx_scale = (self.next_worker_idx_scale + 1) % self.config.NUM_WORKERS
                if self.available_pod_ids: pod_id = self.available_pod_ids.pop()
                else: pod_id = self.next_pod_id; self.next_pod_id += 1
                process = self.env.process(self.pod_worker(pod_id, target_worker))
                self.active_pods.append(self._Pod(pod_id, process, target_worker.id))
        elif desired_replicas < current_replicas:
            num_to_remove = current_replicas - desired_replicas
            print(f"{self.env.now:.2f} [Simulator-Prio]: Rimuovo {num_to_remove} Pods...")
            pods_to_remove = self.active_pods[-num_to_remove:]
            for pod in pods_to_remove:
                if pod.process.is_alive and not pod.process.triggered: pod.process.interrupt()
                self.available_pod_ids.add(pod.id)
            self.active_pods = self.active_pods[:-num_to_remove]

    # Nessuna modifica qui
    def timeout_watcher(self, request: PriorityRequest):
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True
            self.metrics.record_timeout(request, self.env.now)
            print(f"{self.env.now:.2f} [Watcher]: Richiesta {request.request_id} TIMED OUT.")

    def run(self,simulation_duration: float):
        print("--- Avvio Simulatore (Priority - Rete di Code) ---")
        self.env.process(self.request_generator())
        # NUOVO: Avviamo il dispatcher
        self.env.process(self.dispatcher_process())
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)