import math

class HPA:
    """
    Rappresenta il processo del Horizontal Pod Autoscaler.
    MODIFICATO: Ora scala in base alla lunghezza della coda per pod,
    una metrica più adatta per carichi di lavoro basati su code.
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

            num_active_pods = len(self.simulator.active_pods)

            # Non usare più l'utilizzo, ma la lunghezza della coda
            # Questo è un indicatore proattivo del carico
            current_queue_length = len(self.simulator.request_queue.items)

            if num_active_pods > 0:
                # Calcola la metrica: quante richieste in attesa ci sono in media per ogni pod?
                avg_queue_per_pod = current_queue_length / num_active_pods
            else:
                # Se non ci sono pod, l'utilizzo metrico è considerato infinito se c'è anche una sola richiesta
                avg_queue_per_pod = float('inf') if current_queue_length > 0 else 0

            # Formula di scaling basata su metrica custom (standard in Kubernetes)
            # desired_replicas = ceil(current_replicas * (current_metric / target_metric))
            if self.config.TARGET_QUEUE_LENGTH_PER_POD > 0:
                desired_replicas = math.ceil(num_active_pods * (avg_queue_per_pod / self.config.TARGET_QUEUE_LENGTH_PER_POD))
            else:
                desired_replicas = num_active_pods

            desired_replicas = int(max(self.config.MIN_PODS, min(self.config.MAX_PODS, desired_replicas)))

            print(
                f"{self.env.now:.2f} [HPA]: Pods attivi: {num_active_pods}, Lunghezza Coda: {current_queue_length}, "
                f"Coda/Pod: {avg_queue_per_pod:.2f}, Repliche Desiderate: {desired_replicas}")

            # La logica di scaling e cooldown rimane invariata
            if desired_replicas != num_active_pods:
                if desired_replicas > num_active_pods:
                    if self.env.now >= self.last_scale_up_time + self.config.SCALE_UP_COOLDOWN:
                        print(f"{self.env.now:.2f} [HPA]: Avvio SCALE UP a {desired_replicas} pods.")
                        self.simulator.scale_to(desired_replicas)
                        self.last_scale_up_time = self.env.now
                    else:
                        print(f"{self.env.now:.2f} [HPA]: Scale-Up bloccato da cooldown.")
                else:
                    if self.env.now >= self.last_scale_down_time + self.config.SCALE_DOWN_COOLDOWN:
                        print(f"{self.env.now:.2f} [HPA]: Avvio SCALE DOWN a {desired_replicas} pods.")
                        self.simulator.scale_to(desired_replicas)
                        self.last_scale_down_time = self.env.now
                    else:
                        print(f"{self.env.now:.2f} [HPA]: Scale-Down bloccato da cooldown.")