import math

class HPA:
    def __init__(self, env, simulator):
        self.env = env; self.simulator = simulator; self.config = simulator.config
        self.last_scale_up_time = -self.config.SCALE_UP_COOLDOWN
        self.last_scale_down_time = -self.config.SCALE_DOWN_COOLDOWN
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.config.HPA_SYNC_PERIOD)
            num_active_pods = len(self.simulator.active_pods)

            # Legge dalla coda centrale 'request_queue' che ora esiste in entrambi i simulatori
            current_queue_length = len(self.simulator.request_queue.items)

            avg_queue_per_pod = current_queue_length / num_active_pods if num_active_pods > 0 else float('inf')

            if self.config.TARGET_QUEUE_LENGTH_PER_POD > 0:
                desired_replicas_raw = math.ceil(num_active_pods * (avg_queue_per_pod / self.config.TARGET_QUEUE_LENGTH_PER_POD))
            else:
                desired_replicas_raw = num_active_pods

            if desired_replicas_raw > num_active_pods:
                desired_replicas = min(desired_replicas_raw, num_active_pods + self.config.MAX_SCALE_STEP)
            elif desired_replicas_raw < num_active_pods:
                desired_replicas = max(desired_replicas_raw, num_active_pods - self.config.MAX_SCALE_STEP)
            else:
                desired_replicas = num_active_pods

            desired_replicas = int(max(self.config.MIN_PODS, min(self.config.MAX_PODS, desired_replicas)))

            if desired_replicas > num_active_pods:
                if self.env.now >= self.last_scale_up_time + self.config.SCALE_UP_COOLDOWN:
                    self.simulator.scale_to(desired_replicas)
                    self.last_scale_up_time = self.env.now
            elif desired_replicas < num_active_pods:
                if self.env.now >= self.last_scale_down_time + self.config.SCALE_DOWN_COOLDOWN:
                    self.simulator.scale_to(desired_replicas)
                    self.last_scale_down_time = self.env.now