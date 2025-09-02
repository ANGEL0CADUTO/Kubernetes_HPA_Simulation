# File: src/simulation/simulator_wfq.py (LA VERSIONE DEFINITIVA)
from collections import defaultdict
import simpy
from src.controller.hpa import HPA
from src.model.request import PriorityRequest
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler
from src.utils.wfq_store import WFQStore

class SimulatorWFQ:
    class _Pod:
        def __init__(self, pod_id, process): self.id = pod_id; self.process = process

    def __init__(self, config, metrics, arrival_rng, choice_rng, service_rng, lambda_function, timeouts_enabled=True):
        self.config = config; self.metrics = metrics; self.env = simpy.Environment()
        self.arrival_rng = arrival_rng; self.choice_rng = choice_rng; self.service_rng = service_rng
        self.lambda_function = lambda_function; self.service = PodService(service_rng, config)
        self.traffic_profiler = DynamicTrafficProfiler(metrics, config); self.timeouts_enabled = timeouts_enabled
        wfq_weights = {config.Priority.HIGH: 50, config.Priority.MEDIUM: 30, config.Priority.LOW: 20}
        self.request_queue = WFQStore(self.env, weights=wfq_weights)
        self.active_pods = []; self.next_pod_id = 0; self.available_pod_ids = set()

    def request_generator(self):
        # ... (invariato e corretto)
        req_id_counter = 0
        while True:
            current_arrival_rate = self.lambda_function(self.env.now)
            if current_arrival_rate <= 0: yield self.env.timeout(1); continue
            time_to_next = self.arrival_rng.exponential(1.0 / current_arrival_rate)
            yield self.env.timeout(time_to_next)
            req_types, req_probs = self.traffic_profiler.get_current_probabilities()
            chosen_type = self.choice_rng.choice(req_types, p=req_probs)
            service_time = self.service.get_service_time(chosen_type)
            prio = self.config.REQUEST_TYPE_TO_PRIORITY[chosen_type]
            timeout = self.config.REQUEST_TIMEOUTS[chosen_type]
            req_id_counter += 1
            new_request = PriorityRequest(req_id_counter, chosen_type, self.env.now, timeout, service_time, prio)
            self.metrics.record_request_generation(self.env.now, prio, chosen_type)
            if self.timeouts_enabled: self.env.process(self.timeout_watcher(new_request))
            self.request_queue.put(new_request)

    def pod_worker(self, pod_id):
        while True:
            try:
                # --- PATTERN CORRETTO DEFINITIVO ---
                request = yield self.env.process(self.request_queue.get())

                request.is_serviced = True
                if request.timed_out: continue
                wait_time = self.env.now - request.arrival_time
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                self.metrics.record_request_metrics(completion_time, request, response_time, wait_time)
            except simpy.Interrupt: break

    def metrics_recorder(self):
        # ... (reinserito il logger di debug, invariato dalla versione che funzionava)
        DEBUG_INTERVAL = 50.0; last_debug_time = 0.0
        served_since_last_debug = defaultdict(int)
        while True:
            # Registrazione Metriche
            pod_count = len(self.active_pods)
            all_items_in_queue = self.request_queue.items
            total_queue_len = len(all_items_in_queue)
            queue_lengths_per_prio = defaultdict(int)
            for item in all_items_in_queue: queue_lengths_per_prio[item.priority] += 1
            self.metrics.record_system_metrics(self.env.now, pod_count, total_queue_len, queue_lengths_per_prio)

            # Debug Logger
            if self.env.now >= last_debug_time + DEBUG_INTERVAL:
                print("\n" + "="*80); print(f"DEBUG REPORT @ Time: {self.env.now:.2f}s")
                print(f"  - Pod Attivi: {pod_count}, Coda Totale: {total_queue_len}")
                print(f"  - Coda Dettaglio: (HIGH: {queue_lengths_per_prio[self.config.Priority.HIGH]}, MEDIUM: {queue_lengths_per_prio[self.config.Priority.MEDIUM]}, LOW: {queue_lengths_per_prio[self.config.Priority.LOW]})")
                stats = self.metrics.get_welford_statistics()
                current_served_count = defaultdict(int)
                for prio in self.config.Priority:
                    for req_type, p_enum in self.config.REQUEST_TYPE_TO_PRIORITY.items():
                        if p_enum == prio and req_type in stats['by_req_type']:
                            current_served_count[prio] += stats['by_req_type'][req_type]['count']
                print("  - Servite (ultimi 50s):")
                for prio in self.config.Priority:
                    served_in_interval = current_served_count[prio] - served_since_last_debug[prio]
                    print(f"    - {prio.name}: {served_in_interval}")
                    served_since_last_debug[prio] = current_served_count[prio]
                print("="*80)
                last_debug_time = self.env.now

            yield self.env.timeout(1)

    def scale_to(self, desired_replicas):
        current_replicas = len(self.active_pods)
        if desired_replicas > current_replicas:
            for _ in range(desired_replicas - current_replicas):
                pod_id = self.available_pod_ids.pop() if self.available_pod_ids else self.next_pod_id
                if not self.available_pod_ids: self.next_pod_id += 1
                process = self.env.process(self.pod_worker(pod_id))
                self.active_pods.append(self._Pod(pod_id, process))
        elif desired_replicas < current_replicas:
            pods_to_remove = self.active_pods[desired_replicas:]
            for pod in pods_to_remove:
                if pod.process.is_alive and not pod.process.triggered: pod.process.interrupt()
                self.available_pod_ids.add(pod.id)
            self.active_pods = self.active_pods[:desired_replicas]

    def timeout_watcher(self, request: PriorityRequest):
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True; self.metrics.record_timeout(request, self.env.now)

    def run(self, simulation_duration: float):
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)