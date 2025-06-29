import simpy
import math

class HPA:
    """
    Rappresenta il processo del Horizontal Pod Autoscaler.
    Monitora l'utilizzo del sistema e prende decisioni di scaling.
    """
    def __init__(self, env, simulator):
        self.env = env
        self.simulator = simulator
        self.config = simulator.config

        self.last_scale_up_time = -self.config.SCALE_UP_COOLDOWN
        self.last_scale_down_time = -self.config.SCALE_DOWN_COOLDOWN
        self.action = env.process(self.run())

    def run(self):
        """Processo principale dell'HPA, eseguito periodicamente."""
        while True:
            yield self.env.timeout(self.config.HPA_SYNC_PERIOD)

            num_active_pods = len(self.simulator.active_pods) # <-- USA LA NUOVA LISTA

            if num_active_pods == 0:
                current_utilization = 0.0
            else:
                num_busy_pods = self.simulator.get_busy_pods_count()
                current_utilization = max(0, num_busy_pods) / num_active_pods

            if self.config.CPU_TARGET > 0:
                desired_replicas = math.ceil(num_active_pods * (current_utilization / self.config.CPU_TARGET))
            else:
                desired_replicas = num_active_pods

            desired_replicas = int(max(self.config.MIN_PODS, min(self.config.MAX_PODS, desired_replicas)))

            print(f"{self.env.now:.2f} [HPA]: Pods attivi: {num_active_pods}, Utilizzo: {current_utilization:.2%}, Repliche Desiderate: {desired_replicas}")

            # Controlla se è necessario uno scaling
            if desired_replicas != num_active_pods:
                # Applica il cooldown appropriato
                if desired_replicas > num_active_pods:
                    if self.env.now >= self.last_scale_up_time + self.config.SCALE_UP_COOLDOWN:
                        print(f"{self.env.now:.2f} [HPA]: Avvio SCALE UP a {desired_replicas} pods.")
                        self.simulator.scale_to(desired_replicas)
                        self.last_scale_up_time = self.env.now
                    else:
                        print(f"{self.env.now:.2f} [HPA]: Scale-Up bloccato da cooldown.")
                else: # desired_replicas < num_active_pods
                    if self.env.now >= self.last_scale_down_time + self.config.SCALE_DOWN_COOLDOWN:
                        print(f"{self.env.now:.2f} [HPA]: Avvio SCALE DOWN a {desired_replicas} pods.")
                        self.simulator.scale_to(desired_replicas)
                        self.last_scale_down_time = self.env.now
                    else:
                        print(f"{self.env.now:.2f} [HPA]: Scale-Down bloccato da cooldown.")