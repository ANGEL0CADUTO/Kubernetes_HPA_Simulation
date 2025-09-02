import simpy
from src.model.request import Request
from src.controller.hpa import HPA
from src.service.service import PodService
from src.service.traffic_profiler import DynamicTrafficProfiler

class Simulator:
    class _Pod:
        def __init__(self, pod_id, process):
            self.id = pod_id
            self.process = process

    def __init__(self, config_module, metrics, arrival_rng, choice_rng, service_rng, lambda_function):
        self.config = config_module
        self.metrics = metrics
        self.env = simpy.Environment()
        self.arrival_rng = arrival_rng
        self.choice_rng = choice_rng
        self.lambda_function = lambda_function
        self.service = PodService(service_rng, config_module)
        self.traffic_profiler = DynamicTrafficProfiler(metrics, config_module)
        self.request_queue = simpy.Store(self.env)
        self.active_pods = []
        self.next_pod_id = 0
        self.available_pod_ids = set()

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
            self.metrics.record_request_generation(chosen_type)
            self.env.process(self.timeout_watcher(new_request))
            self.request_queue.put(new_request)

    def pod_worker(self, pod_id):
        while True:
            try:
                request = yield self.request_queue.get()
                request.is_serviced = True
                if request.timed_out: continue
                wait_time = self.env.now - request.arrival_time
                yield self.env.timeout(request.service_time)
                completion_time = self.env.now
                response_time = completion_time - request.arrival_time
                self.metrics.record_request_metrics(completion_time, request.req_type, response_time, wait_time)
            except simpy.Interrupt: break

    def metrics_recorder(self):
        while True:
            queue_len = len(self.request_queue.items)
            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(self.env.now, pod_count, queue_len)
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

    def timeout_watcher(self, request: Request):
        yield self.env.timeout(request.timeout)
        if not request.is_serviced:
            request.timed_out = True
            self.metrics.record_timeout(request.req_type, self.env.now)

    def run(self, simulation_duration: float):
        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())
        self.scale_to(self.config.INITIAL_PODS)
        if self.config.HPA_ENABLED: HPA(self.env, self)
        self.env.run(until=simulation_duration)