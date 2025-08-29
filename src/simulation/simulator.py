import simpy
from src.model.request import Request
from src.controller.hpa import HPA
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler
# NUOVO: Importiamo la nostra nuova classe WorkerNode
from src.model.worker_node import WorkerNode

class Simulator:
    class _Pod:
        # MODIFICATO: Il Pod ora deve sapere a quale worker è assegnato
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

        # RIMOSSO: La coda singola centralizzata non esiste più
        # self.request_queue = simpy.Store(self.env)

        # NUOVO: Coda di ingresso per il dispatcher. Le richieste generate arrivano qui.
        self.dispatcher_inbox = simpy.Store(self.env)

        # NUOVO: Creiamo i Worker Node
        self.worker_nodes = [WorkerNode(self.env, i) for i in range(self.config.NUM_WORKERS)]
        self.next_worker_idx_dispatch = 0  # Per la politica Round Robin del dispatcher
        self.next_worker_idx_scale = 0   # Per la politica Round Robin dello scaling

        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def dispatcher_process(self):
        """
        NUOVO PROCESSO: Simula il Master Node che fa da dispatcher.
        Preleva le richieste dall'inbox e le smista nelle code dei Worker Node.
        """
        print(f"0.00 [Dispatcher]: Avviato.")
        while True:
            request = yield self.dispatcher_inbox.get()

            # Politica di smistamento: Round Robin
            target_worker = self.worker_nodes[self.next_worker_idx_dispatch]
            target_worker.queue.put(request)

            print(f"{self.env.now:.2f} [Dispatcher]: Richiesta {request.request_id} smistata al Worker {target_worker.id}.")

            # Aggiorna l'indice per il prossimo worker
            self.next_worker_idx_dispatch = (self.next_worker_idx_dispatch + 1) % self.config.NUM_WORKERS

    def request_generator(self):
        req_id_counter = 0
        while True:
            # ... (la logica di generazione della richiesta rimane identica)
            current_arrival_rate = self.lambda_function(self.env.now)
            if current_arrival_rate <= 0:
                yield self.env.timeout(1)
                continue
            time_to_next = self.arrival_rng.exponential(1.0 / current_arrival_rate)
            yield self.env.timeout(time_to_next)
            req_types, req_probs = self.traffic_profiler.get_current_probabilities()
            chosen_type = self.choice_rng.choice(req_types, p=req_probs)
            service_time = self.service.get_service_time(chosen_type)
            type_timeout = self.config.REQUEST_TIMEOUTS[chosen_type]
            req_id_counter += 1
            new_request = Request(
                request_id=req_id_counter, req_type=chosen_type,
                arrival_time=self.env.now, timeout=type_timeout,
                service_time=service_time
            )
            self.metrics.record_request_generation(chosen_type)
            print(f"{self.env.now:.2f} [Generator]: Richiesta {new_request.request_id} generata.")
            self.env.process(self.timeout_watcher(new_request))

            # MODIFICATO: La richiesta non va più nella coda centrale, ma nell'inbox del dispatcher
            self.dispatcher_inbox.put(new_request)

    # MODIFICATO: Il pod_worker ora è legato a un worker specifico
    def pod_worker(self, pod_id, worker_node: WorkerNode):
        print(f"{self.env.now:.2f} [Pod {pod_id} @ Worker {worker_node.id}]: Avviato.")
        while True:
            try:
                # MODIFICATO: Il pod estrae una richiesta dalla CODA LOCALE del suo worker
                request = yield worker_node.queue.get()

                # ... (la logica di processamento della richiesta rimane identica)
                request.is_serviced = True
                if request.timed_out:
                    print(f"{self.env.now:.2f} [Pod {pod_id}]: Scartata richiesta {request.request_id} (già scaduta).")
                    continue
                wait_time = self.env.now - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento rich. {request.request_id}. Attesa: {wait_time:.4f}s")
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento rich. {request.request_id}. Tempo di risposta: {response_time:.4f}s")
                self.metrics.record_request_metrics(completion_time, request.req_type, response_time, wait_time)

            except simpy.Interrupt:
                break
        print(f"{self.env.now:.2f} [Pod {pod_id} @ Worker {worker_node.id}]: Terminato.")

    def metrics_recorder(self):
        while True:
            # MODIFICATO: La lunghezza della coda è ora la somma delle code di tutti i worker
            queue_len = sum(len(w.queue.items) for w in self.worker_nodes)
            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(self.env.now, pod_count, queue_len)
            yield self.env.timeout(1)

    # Questo metodo non è più accurato in un modello a code multiple, lo lasciamo per compatibilità
    def get_busy_pods_count(self):
        return len(self.active_pods) # Semplificazione temporanea

    def scale_to(self, desired_replicas):
        current_replicas = len(self.active_pods)
        if desired_replicas > current_replicas:
            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator]: Aggiungo {num_to_add} Pods...")
            for _ in range(num_to_add):
                # MODIFICATO: Decidiamo a quale worker assegnare il nuovo pod (politica Round Robin)
                target_worker = self.worker_nodes[self.next_worker_idx_scale]
                self.next_worker_idx_scale = (self.next_worker_idx_scale + 1) % self.config.NUM_WORKERS

                if self.available_pod_ids: pod_id = self.available_pod_ids.pop()
                else: pod_id = self.next_pod_id; self.next_pod_id += 1

                # MODIFICATO: Avviamo il processo del pod legandolo al worker scelto
                process = self.env.process(self.pod_worker(pod_id, target_worker))
                self.active_pods.append(self._Pod(pod_id, process, target_worker.id))

        elif desired_replicas < current_replicas:
            num_to_remove = current_replicas - desired_replicas
            print(f"{self.env.now:.2f} [Simulator]: Rimuovo {num_to_remove} Pods...")
            pods_to_remove = self.active_pods[-num_to_remove:]
            for pod in pods_to_remove:
                if pod.process.is_alive and not pod.process.triggered: pod.process.interrupt()
                self.available_pod_ids.add(pod.id)
            self.active_pods = self.active_pods[:-num_to_remove]

    def timeout_watcher(self, request: Request):
        # ... (nessuna modifica qui)
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True
            self.metrics.record_timeout(request.req_type, self.env.now)
            print(f"{self.env.now:.2f} [Watcher]: Richiesta {request.request_id} TIMED OUT.")

    def run(self, simulation_duration: float):
        print("--- Avvio Simulatore (Baseline - Rete di Code FIFO) ---")
        self.env.process(self.request_generator())
        # NUOVO: Avviamo il processo del dispatcher
        self.env.process(self.dispatcher_process())
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)
        print("--- Simulazione Baseline Terminata ---")