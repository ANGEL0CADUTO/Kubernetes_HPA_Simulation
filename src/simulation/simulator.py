import simpy
from src.model.request import Request
from src.controller.hpa import HPA
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler
from src.model.worker_node import WorkerNode

class Simulator:
    class _Pod:
        def __init__(self, pod_id, process):
            self.id = pod_id
            self.process = process

    def __init__(self, config_module, metrics_class, arrival_rng, choice_rng, service_rng, lambda_function):
        self.config = config_module
        self.env = simpy.Environment()
        self.arrival_rng = arrival_rng
        self.choice_rng = choice_rng
        self.lambda_function = lambda_function
        self.service = PodService(service_rng, config_module)

        # MODIFICA FINALE: Crea sia le metriche per-worker che quelle aggregate.
        self.metrics_per_worker = [metrics_class(config_module) for _ in range(config_module.NUM_WORKERS)]
        self.metrics_agg = metrics_class(config_module)

        self.traffic_profiler = DynamicTrafficProfiler(self.metrics_agg, config_module)

        self.worker_nodes = [
            WorkerNode(env=self.env, node_id=i, queue_instance=simpy.Store(self.env))
            for i in range(self.config.NUM_WORKERS)
        ]
        self.next_worker_idx = 0

    def request_generator(self):
        req_id_counter = 0
        while True:
            current_arrival_rate = self.lambda_function(self.env.now)
            if current_arrival_rate <= 0: yield self.env.timeout(1); continue
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
            # Registra la generazione solo sulla metrica aggregata.
            self.metrics_agg.record_request_generation(chosen_type)
            self.env.process(self.timeout_watcher(new_request))

            target_worker = self.worker_nodes[self.next_worker_idx]
            target_worker.queue.put(new_request)
            self.next_worker_idx = (self.next_worker_idx + 1) % self.config.NUM_WORKERS

    def pod_worker(self, pod_id, worker: WorkerNode):
        # MODIFICA FINALE: Ottieni l'istanza di metriche specifica per questo worker.
        worker_metrics = self.metrics_per_worker[worker.id]
        while True:
            try:
                request = yield worker.queue.get()
                if request.timed_out:
                    continue
                request.is_serviced = True

                wait_time = self.env.now - request.arrival_time
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time

                # Registra le metriche sia sull'istanza del worker CHE su quella aggregata.
                worker_metrics.record_request_metrics(completion_time, request.req_type, response_time, wait_time)
                self.metrics_agg.record_request_metrics(completion_time, request.req_type, response_time, wait_time)
            except simpy.Interrupt: break

    def metrics_recorder(self):
        while True:
            # Registra le metriche di sistema aggregate
            total_queue_len = sum(len(w.queue.items) for w in self.worker_nodes)
            total_pod_count = sum(len(w.active_pods) for w in self.worker_nodes)
            self.metrics_agg.record_system_metrics(self.env.now, total_pod_count, total_queue_len)

            # MODIFICA FINALE: Registra le metriche di sistema per ogni singolo worker.
            for worker in self.worker_nodes:
                worker_metrics = self.metrics_per_worker[worker.id]
                worker_metrics.record_system_metrics(self.env.now, len(worker.active_pods), len(worker.queue.items))

            yield self.env.timeout(1)

    def scale_worker(self, worker: WorkerNode, desired_replicas: int):
        current_replicas = len(worker.active_pods)
        if desired_replicas > current_replicas:
            for _ in range(desired_replicas - current_replicas):
                pod_id = worker.available_pod_ids.pop() if worker.available_pod_ids else worker.next_pod_id
                if not worker.available_pod_ids: worker.next_pod_id += 1
                process = self.env.process(self.pod_worker(pod_id, worker))
                worker.active_pods.append(self._Pod(pod_id, process))
        elif desired_replicas < current_replicas:
            pods_to_remove = worker.active_pods[desired_replicas:]
            for pod in pods_to_remove:
                if pod.process.is_alive and not pod.process.triggered: pod.process.interrupt()
                worker.available_pod_ids.add(pod.id)
            worker.active_pods = worker.active_pods[:desired_replicas]

    def timeout_watcher(self, request: Request):
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True
            # Registra il timeout solo sulla metrica aggregata.
            self.metrics_agg.record_timeout(request.req_type, self.env.now)

    def run(self, simulation_duration: float):
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())

        for worker in self.worker_nodes:
            self.scale_worker(worker, self.config.INITIAL_PODS_PER_WORKER)

        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)