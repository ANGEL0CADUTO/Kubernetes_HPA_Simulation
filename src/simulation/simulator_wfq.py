from collections import defaultdict
import simpy
from src.controller.hpa import HPA
from src.model.request import PriorityRequest
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler
from src.utils.wfq_store import WFQStore
from src.model.worker_node import WorkerNode

class SimulatorWFQ:
    class _Pod:
        def __init__(self, pod_id, process): self.id = pod_id; self.process = process

    def __init__(self, config, metrics_class, arrival_rng, choice_rng, service_rng, lambda_function, timeouts_enabled=True):
        self.config = config
        self.env = simpy.Environment()
        self.arrival_rng = arrival_rng
        self.choice_rng = choice_rng
        self.service_rng = service_rng
        self.lambda_function = lambda_function
        self.service = PodService(service_rng, config)
        self.timeouts_enabled = timeouts_enabled

        self.metrics_per_worker = [metrics_class(config) for _ in range(config.NUM_WORKERS)]
        self.metrics_agg = metrics_class(config)

        self.traffic_profiler = DynamicTrafficProfiler(self.metrics_agg, config)

        self.NORMAL_WEIGHTS = {self.config.Priority.HIGH: 94, self.config.Priority.MEDIUM: 5, self.config.Priority.LOW: 1}

        self.worker_nodes = []
        for i in range(self.config.NUM_WORKERS):
            wfq_queue = WFQStore(self.env, weights=self.NORMAL_WEIGHTS.copy())
            self.worker_nodes.append(WorkerNode(env=self.env, node_id=i, queue_instance=wfq_queue))

        self.next_worker_idx = 0
        self.shedding_enabled = False

    def request_generator(self):
        req_id_counter = 0
        while True:
            current_arrival_rate = self.lambda_function(self.env.now)
            if current_arrival_rate <= 0: yield self.env.timeout(1); continue
            time_to_next = self.arrival_rng.exponential(1.0 / current_arrival_rate)
            yield self.env.timeout(time_to_next)

            req_types, req_probs = self.traffic_profiler.get_current_probabilities()
            chosen_type = self.choice_rng.choice(req_types, p=req_probs)
            prio = self.config.REQUEST_TYPE_TO_PRIORITY[chosen_type]

            if self.shedding_enabled and prio > self.config.Priority.MEDIUM:
                req_id_counter += 1
                rejected_request = PriorityRequest(request_id=req_id_counter, req_type=chosen_type, arrival_time=self.env.now, timeout=0, service_time=0, priority=prio)
                self.metrics_agg.record_request_failure(rejected_request, self.env.now)
                continue

            service_time = self.service.get_service_time(chosen_type)
            timeout = self.config.REQUEST_TIMEOUTS[chosen_type]
            req_id_counter += 1
            new_request = PriorityRequest(request_id=req_id_counter, req_type=chosen_type, arrival_time=self.env.now, timeout=timeout, service_time=service_time, priority=prio)

            self.metrics_agg.record_request_generation(self.env.now, prio, chosen_type)
            if self.timeouts_enabled: self.env.process(self.timeout_watcher(new_request))

            target_worker = self.worker_nodes[self.next_worker_idx]
            target_worker.queue.put(new_request)
            self.next_worker_idx = (self.next_worker_idx + 1) % self.config.NUM_WORKERS

    def pod_worker(self, pod_id, worker: WorkerNode):
        worker_metrics = self.metrics_per_worker[worker.id]
        while True:
            try:
                get_process = self.env.process(worker.queue.get())
                request = yield get_process
                if request.timed_out:
                    continue
                request.is_serviced = True

                wait_time = self.env.now - request.arrival_time
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time

                worker_metrics.record_request_metrics(completion_time, request, response_time, wait_time)
                self.metrics_agg.record_request_metrics(completion_time, request, response_time, wait_time)
            except simpy.Interrupt: break

    def metrics_recorder(self):
        ALERT_THRESHOLD = 100
        SHEDDING_TIME_THRESHOLD = 300
        CRISIS_WEIGHTS = {self.config.Priority.HIGH: 98, self.config.Priority.MEDIUM: 2, self.config.Priority.LOW: 0}
        in_crisis_mode = False
        time_crisis_started = -1

        while True:
            yield self.env.timeout(1)

            total_queue_len = sum(len(w.queue.items) for w in self.worker_nodes)

            if total_queue_len > ALERT_THRESHOLD and not in_crisis_mode:
                in_crisis_mode = True
                print(f"{self.env.now:.2f} [CRISIS]: Coda aggregata > {ALERT_THRESHOLD}. Attivazione modalità crisi.")
                time_crisis_started = self.env.now

                for worker in self.worker_nodes:
                    # 1. Applica i pesi di crisi per le nuove richieste
                    worker.queue.update_weights(CRISIS_WEIGHTS)

                    # 2. MODIFICA: Attiva il load shedding retroattivo per la classe LOW.
                    # Purga immediatamente tutte le richieste LOW esistenti dalla coda del worker.
                    purged_requests = worker.queue.purge_by_priority(self.config.Priority.LOW)

                    if purged_requests:
                        print(f"{self.env.now:.2f} [SHEDDING]: Rimosse {len(purged_requests)} richieste LOW dal Worker {worker.id}.")
                        # 3. Registra ogni richiesta rimossa come fallita.
                        for req in purged_requests:
                            self.metrics_agg.record_request_failure(req, self.env.now)
                            self.metrics_per_worker[worker.id].record_request_failure(req, self.env.now)

            elif total_queue_len <= ALERT_THRESHOLD and in_crisis_mode:
                in_crisis_mode = False
                print(f"{self.env.now:.2f} [CRISIS]: Coda aggregata <= {ALERT_THRESHOLD}. Ritorno alla normalità.")
                for worker in self.worker_nodes:
                    worker.queue.update_weights(self.NORMAL_WEIGHTS)
                time_crisis_started = -1
                self.shedding_enabled = False

            if in_crisis_mode and not self.shedding_enabled and (self.env.now - time_crisis_started) > SHEDDING_TIME_THRESHOLD:
                print(f"{self.env.now:.2f} [CRISIS]: Crisi prolungata. Attivazione shedding per NUOVI arrivi.")
                self.shedding_enabled = True

            total_pod_count = sum(len(w.active_pods) for w in self.worker_nodes)
            queue_lengths_per_prio_agg = defaultdict(int)

            for worker in self.worker_nodes:
                worker_metrics = self.metrics_per_worker[worker.id]
                queue_lengths_per_prio_worker = defaultdict(int)

                for request_item in worker.queue.items:
                    queue_lengths_per_prio_worker[request_item.priority] += 1

                for prio, length in queue_lengths_per_prio_worker.items():
                    queue_lengths_per_prio_agg[prio] += length

                worker_metrics.record_system_metrics(self.env.now, len(worker.active_pods), len(worker.queue.items), queue_lengths_per_prio_worker)

            self.metrics_agg.record_system_metrics(self.env.now, total_pod_count, total_queue_len, queue_lengths_per_prio_agg)


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

    def timeout_watcher(self, request: PriorityRequest):
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True
            self.metrics_agg.record_request_failure(request, self.env.now)

    def run(self, simulation_duration: float):
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())

        for worker in self.worker_nodes:
            self.scale_worker(worker, self.config.INITIAL_PODS_PER_WORKER)

        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)