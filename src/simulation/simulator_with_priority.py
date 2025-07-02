import simpy
from src.config import Priority
from src.model.request import PriorityRequest
from src.controller.hpa import HPA
from src.service.service import PodService

class SimulatorWithPriority:
    """
    Simulatore per lo scenario migliorato con code di priorità.
    Usa un pattern Dispatcher-Worker con un dispatcher a polling per garantire
    un corretto scheduling e l'utilizzo di tutti i Pod.
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

        self.priority_queues = {
            Priority.HIGH: simpy.Store(self.env),
            Priority.MEDIUM: simpy.Store(self.env),
            Priority.LOW: simpy.Store(self.env)
        }

        self.work_queue = simpy.Store(self.env)

        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def request_generator(self):
        """Genera richieste e le mette nelle code di priorità appropriate."""
        # Questa funzione è già corretta e rimane invariata.
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
            new_request = PriorityRequest(
                request_id=req_id_counter, req_type=chosen_type,
                arrival_time=self.env.now, priority=assigned_priority,
                service_time=service_time)
            self.metrics.record_request_generation(self.env.now)
            print(f"{self.env.now:.2f} [Generator]: Richiesta {new_request.request_id} ({new_request.req_type.name} -> Priorità: {new_request.priority.name}) generata.")
            yield self.priority_queues[new_request.priority].put(new_request)

    # --- MODIFICA CHIAVE: DISPATCHER A POLLING, SEMPLICE E ROBUSTO ---
    def dispatcher(self):
        """
        Processo centrale che osserva le code di priorità in un ciclo continuo (polling)
        e sposta la richiesta più importante nella coda di lavoro.
        """
        while True:
            found_request = False
            # Scansiona le code dalla priorità più alta alla più bassa
            for prio in sorted(self.config.Priority):
                if len(self.priority_queues[prio].items) > 0:
                    # Trovata una richiesta!
                    request_to_dispatch = yield self.priority_queues[prio].get()
                    yield self.work_queue.put(request_to_dispatch)
                    found_request = True
                    # Interrompe il for e fa ripartire il while, per ricontrollare sempre da HIGH
                    break

            # Se non ha trovato nessuna richiesta in nessuna coda,
            # aspetta un istante infinitesimale prima di ricontrollare.
            # Questo previene un ciclo infinito a vuoto ma mantiene il dispatcher reattivo.
            if not found_request:
                yield self.env.timeout(0.01)

    def pod_worker(self, pod_id):
        """
        Processo che simula un Pod. È un semplice consumatore dalla work_queue.
        Questa logica è robusta.
        """
        print(f"{self.env.now:.2f} [Pod {pod_id}]: Avviato.")
        try:
            while True:
                request = yield self.work_queue.get()
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
        while True:
            queue_lengths_per_prio = {p: len(self.priority_queues[p].items) for p in self.config.Priority}
            # La lunghezza totale della coda è la somma delle code di priorità + la coda di lavoro
            total_queue_len = sum(queue_lengths_per_prio.values()) + len(self.work_queue.items)
            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(self.env.now, pod_count, total_queue_len, queue_lengths_per_prio)
            yield self.env.timeout(1)

    # --- MODIFICA CHIAVE: CALCOLO UTILIZZO CORRETTO PER HPA ---
    def get_busy_pods_count(self):
        """Calcola il numero di pod attualmente occupati."""
        # Un Pod è "busy" se non è in attesa di prendere un task dalla coda di lavoro.
        num_pods_waiting_for_work = len(self.work_queue.get_queue)
        num_active_pods = len(self.active_pods)
        # Il numero di pod occupati sono quelli attivi meno quelli in attesa.
        num_busy_pods = num_active_pods - num_pods_waiting_for_work
        return max(0, num_busy_pods)

    def scale_to(self, desired_replicas):
        # Questa funzione è già corretta e non necessita modifiche.
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
        print("--- Avvio Simulatore con Code di Priorità (Dispatcher Model Robusto) ---")
        self.env.process(self.request_generator())
        self.env.process(self.dispatcher()) # Avvia il nuovo dispatcher
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED:
            HPA(self.env, self)
        self.env.run(until=self.config.SIMULATION_TIME)
        print("--- Simulazione Terminata ---")