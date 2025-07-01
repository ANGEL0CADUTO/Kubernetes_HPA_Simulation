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

        # --- LOGICA CHIAVE: CODE PER WORKER ---
        # NON usiamo più una coda centrale. La logica di dispatching
        # sarà nel request_distributor.
        self.worker_queues = {
            worker_id: {
                prio: simpy.Store(self.env) for prio in Priority
            } for worker_id in range(self.config.NUM_WORKERS)
        }

        # Un contatore per il Round Robin tra i worker
        self.next_worker_idx = 0
        # ------------------------------------

        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def request_generator(self):
        """
        Genera richieste e le passa al distributore.
        Questa parte è simile alla precedente implementazione con priorità.
        """
        req_id_counter = 0
        service_classes = list(self.config.SERVICE_CLASSES_CONFIG.keys())
        traffic_shares = [self.config.SERVICE_CLASSES_CONFIG[p]["traffic_share"] for p in service_classes]

        while True:
            time_to_next = self.rng.exponential(1.0 / self.config.TOTAL_ARRIVAL_RATE)
            yield self.env.timeout(time_to_next)

            req_id_counter += 1
            req_types = list(self.config.TRAFFIC_PROFILE.keys())
            req_probs = list(self.config.TRAFFIC_PROFILE.values())

            chosen_type = self.rng.choice(req_types, p=req_probs)
            chosen_priority = random.choices(service_classes, weights=traffic_shares, k=1)[0]
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

            # Invece di una coda centrale, chiamiamo un processo distributore
            self.env.process(self.request_distributor(new_request))

    def request_distributor(self, request: PriorityRequest):
        """
        Processo che riceve una richiesta e la instrada alla coda del worker
        corretto usando una politica Round Robin.
        """
        # Scegli il prossimo worker in modo Round Robin
        worker_id = self.next_worker_idx
        self.next_worker_idx = (self.next_worker_idx + 1) % self.config.NUM_WORKERS

        # Prendi la coda giusta per quel worker e quella priorità
        target_queue = self.worker_queues[worker_id][request.priority]

        print(f"{self.env.now:.2f} [Distributor]: Richiesta {request.request_id} instradata al Worker {worker_id}, Coda {request.priority.name}.")

        # Metti la richiesta nella coda specifica del worker
        # Qui potresti aggiungere la logica per la coda piena, se necessario
        yield target_queue.put(request)

    def pod_worker(self, pod_id, worker_id):
        """
        Processo che simula un Pod. Ora è legato a un worker specifico
        e pesca richieste solo dalle code di quel worker.
        """
        print(f"{self.env.now:.2f} [Pod {pod_id} on Worker {worker_id}]: Avviato.")

        # Lista delle code di questo worker, in ordine di priorità
        # (dal valore di Enum più basso al più alto)
        ordered_queues = [
            self.worker_queues[worker_id][prio]
            for prio in sorted(Priority)
        ]

        while True:
            try:
                # --- LOGICA CHIAVE: SCHEDULING A PRIORITÀ STRETTA ---
                # Prova a prendere una richiesta dalla coda con la priorità più alta.
                # `simpy.any_of` è perfetto per questo: attende che almeno uno
                # degli eventi (get da una delle code) si verifichi.
                get_events = [q.get() for q in ordered_queues]
                result = yield self.env.any_of(get_events)

                # `result` è un dizionario che mappa l'evento alla sua risposta.
                # Poiché `any_of` si sblocca al primo evento completato,
                # ci sarà solo una chiave.
                request = list(result.values())[0]
                # --------------------------------------------------------

                arrival_in_service = self.env.now
                wait_time = arrival_in_service - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento rich. {request.request_id} (Priorità: {request.priority.name}). Attesa: {wait_time:.4f}s")

                yield self.env.timeout(request.service_time)

                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento rich. {request.request_id}. Tempo di risposta: {response_time:.4f}s")

                self.metrics.record_request_metrics(completion_time, request, response_time, wait_time)

            except simpy.Interrupt:
                break

        print(f"{self.env.now:.2f} [Pod {pod_id}]: Rilevato segnale di stop, terminazione.")

    def metrics_recorder(self):
        # La logica è la stessa, ma ora la lunghezza della coda è la somma di tutte le code
        while True:
            total_queue_len = 0
            for worker_id in range(self.config.NUM_WORKERS):
                for prio in Priority:
                    total_queue_len += len(self.worker_queues[worker_id][prio].items)

            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(
                self.env.now, pod_count, total_queue_len
            )
            yield self.env.timeout(1)

    def get_busy_pods_count(self):
        # Questa metrica diventa più complessa da calcolare con precisione
        # con le code distribuite. Una buona approssimazione rimane basata
        # sui processi attivi. Per HPA, questa metrica è meno importante
        # se l'HPA si basa sulla lunghezza della coda o sulla CPU.
        # Manteniamo la logica semplice per ora.
        num_active_pods = len(self.active_pods)
        # Una stima: se la coda totale non è vuota, probabilmente tutti i pod sono occupati.
        total_queue_len = sum(len(q.items) for w_queues in self.worker_queues.values() for q in w_queues.values())
        if total_queue_len > 0:
            return num_active_pods
        else:
            # Calcolo più preciso se la coda è vuota
            # (non implementato qui per semplicità, ma potresti tracciare lo stato di ogni Pod)
            return 0 # Stima conservativa

    def scale_to(self, desired_replicas):
        # La logica di scaling deve ora assegnare i Pod ai Worker.
        # Useremo una strategia Round Robin anche qui.
        current_replicas = len(self.active_pods)

        if desired_replicas > current_replicas:
            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator]: Aggiungo {num_to_add} Pods...")
            for _ in range(num_to_add):
                pod_id = self.available_pod_ids.pop() if self.available_pod_ids else self.next_pod_id
                if not self.available_pod_ids: self.next_pod_id += 1

                # Assegna il nuovo Pod a un worker in modo Round Robin
                worker_id = len(self.active_pods) % self.config.NUM_WORKERS

                process = self.env.process(self.pod_worker(pod_id, worker_id))
                self.active_pods.append(self._Pod(pod_id, process))

        elif desired_replicas < current_replicas:
            # La logica di rimozione può rimanere la stessa, rimuove gli ultimi aggiunti.
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