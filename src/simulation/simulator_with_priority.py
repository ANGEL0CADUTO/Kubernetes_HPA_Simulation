# src/simulation/simulator_with_priority.py

import simpy
import random

from src.config import Priority
from src.model.request import PriorityRequest
from src.controller.hpa import HPA
from src.service.service import PodService


class SimulatorWithPriority:
    """
    Simulatore per lo scenario migliorato con code di priorità per ogni worker.
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

        # --- PRIORITY QUEUES ---
        self.high_priority_queue = simpy.Store(self.env)
        self.medium_priority_queue = simpy.Store(self.env)
        self.low_priority_queue = simpy.Store(self.env)

        # --- PODS ---
        self.active_pods = []
        self.idle_pod_ids = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def request_generator(self):
        """
        Genera richieste e le passa al distributore.
        Questa parte è simile alla precedente implementazione con priorità.
        """
        req_id_counter = 0

        ordered_priorities = sorted(self.config.SERVICE_CLASSES_CONFIG.keys())
        traffic_shares = [self.config.SERVICE_CLASSES_CONFIG[p]["traffic_share"] for p in ordered_priorities]

        while True:

            # --- INTER ARRIVAL TIMES ---
            time_to_next = self.rng.exponential(1.0 / self.config.TOTAL_ARRIVAL_RATE)
            yield self.env.timeout(time_to_next)

            # --- CREATE NEW REQUEST ---
            req_id_counter += 1
            req_types = list(self.config.TRAFFIC_PROFILE.keys())
            req_probs = list(self.config.TRAFFIC_PROFILE.values())

            chosen_type = self.rng.choice(req_types, p=req_probs)

            #todo fix la priorità non va generata ma dedotta dal tipo di richiesta. accordarsi sulle richieste perché non mi ricordo adesso
            chosen_priority = random.choices(ordered_priorities, weights=traffic_shares, k=1)[0]

            class_config = self.config.SERVICE_CLASSES_CONFIG[chosen_priority]
            avg_service_time = class_config["avg_service_time_ms"] / 1000.0
            service_time = self.rng.exponential(avg_service_time)

            new_request = PriorityRequest(
                request_id=req_id_counter,
                req_type=chosen_type,
                arrival_time=self.env.now,
                priority=chosen_priority,
                service_time=service_time
            )
            self.metrics.record_request_generation(self.env.now)

            print(f"{self.env.now:.2f} [Generator]: Richiesta {new_request.request_id} (Priorità: {new_request.priority.name}) generata.")

            # --- PLACE NEW REQUEST IN CORRECT PRIORITY QUEUE ---
            if new_request.priority == Priority.HIGH:
                yield self.high_priority_queue.put(new_request)
            elif new_request.priority == Priority.MEDIUM:
                yield self.medium_priority_queue.put(new_request)
            else: # Priority.BASSA
                yield self.low_priority_queue.put(new_request)

    def pod_worker(self, pod_id):
        """
        Processo che simula un Pod. Pesca richieste dalle code abstract priority
        """
        print(f"{self.env.now:.2f} [Pod {pod_id}]: Avviato e pronto a ricevere lavoro.")

        try:
            while True:
                request_to_process = None
                is_idle = False

                # 1. Controlla le code in ordine di priorità
                if len(self.high_priority_queue.items) > 0:
                    request_to_process = yield self.high_priority_queue.get()
                elif len(self.medium_priority_queue.items) > 0:
                    request_to_process = yield self.medium_priority_queue.get()
                elif len(self.low_priority_queue.items) > 0:
                    request_to_process = yield self.low_priority_queue.get()

                # 2. Se tutte le code sono vuote si mette in attesa, appena arriva una richiesta "si sveglia"
                if request_to_process is None:
                    is_idle = True
                    self.idle_pod_ids.append(pod_id)

                    # `Spiegazione: any_of` qui è perfetto per l'attesa. Quando il pod si sveglia,
                    # il loop `while` ripartirà e la catena di if/elif sopra garantirà
                    # che la richiesta a priorità più alta venga servita.
                    result = yield self.env.any_of([
                        self.high_priority_queue.get(),
                        self.medium_priority_queue.get(),
                        self.low_priority_queue.get()
                    ])
                    request_to_process = list(result.values())[0]

                # Prima del processamento indichiamo che il pod non è più idle
                if is_idle:
                    self.idle_pod_ids.remove(pod_id)

                # 3. Processamento della richiesta
                arrival_in_service = self.env.now
                wait_time = arrival_in_service - request_to_process.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento rich. {request_to_process.request_id} "
                      f"(Priorità: {request_to_process.priority.name}). Attesa: {wait_time:.4f}s")

                yield self.env.timeout(request_to_process.service_time)

                completion_time = self.env.now
                response_time = completion_time - request_to_process.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento rich. {request_to_process.request_id}. "
                      f"Tempo di risposta: {response_time:.4f}s")

                self.metrics.record_request_metrics(completion_time, request_to_process, response_time, wait_time)

        # SCALE DOWN: Interruzione e rimozione del pod dalla lista idle in caso di spegnimento del pod
        except simpy.Interrupt:
            if pod_id in self.idle_pod_ids:
                self.idle_pod_ids.remove(pod_id)

            print(f"{self.env.now:.2f} [Pod {pod_id}]: Ricevuto segnale di stop, terminazione.")

    def metrics_recorder(self):

        while True:
            # Calcola le lunghezze delle code esplicitamente
            queue_lengths_per_prio = {
                Priority.HIGH: len(self.high_priority_queue.items),
                Priority.MEDIUM: len(self.medium_priority_queue.items),
                Priority.LOW: len(self.low_priority_queue.items)
            }
            total_queue_len = sum(queue_lengths_per_prio.values())

            pod_count = len(self.active_pods)

            self.metrics.record_system_metrics(
                self.env.now, pod_count, total_queue_len
            )
            yield self.env.timeout(1)

    def get_busy_pods_count(self):
        """
        Calcola il numero esatto di Pod che stanno attualmente processando una richiesta.
        """

        return len(self.active_pods) - len(self.idle_pod_ids)

    def scale_to(self, desired_replicas):

        current_replicas = len(self.active_pods)

        # --- SCALE UP ---
        if desired_replicas > current_replicas:

            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator]: Aggiungo {num_to_add} Pods...")

            for _ in range(num_to_add):
                pod_id = self.available_pod_ids.pop() if self.available_pod_ids else self.next_pod_id
                if not self.available_pod_ids:
                    self.next_pod_id += 1

                process = self.env.process(self.pod_worker(pod_id))
                self.active_pods.append(self._Pod(pod_id, process))

        # --- SCALE DOWN
        elif desired_replicas < current_replicas:

            num_to_remove = current_replicas - desired_replicas
            print(f"{self.env.now:.2f} [Simulator]: Rimuovo {num_to_remove} Pods...")

            pods_to_remove = self.active_pods[-num_to_remove:]

            for pod in pods_to_remove:
                if pod.process.is_alive:
                    pod.process.interrupt()
                self.available_pod_ids.add(pod.id)
            self.active_pods = self.active_pods[:-num_to_remove]

    def run(self):
        print("--- Avvio Simulatore con Code di Priorità per Worker ---")
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED:
            HPA(self.env, self)
        self.env.run(until=self.config.SIMULATION_TIME)
        print("--- Simulazione Terminata ---")