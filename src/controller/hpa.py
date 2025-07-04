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
            current_queue_length = len(self.simulator.request_queue.items)

            if num_active_pods > 0:
                avg_queue_per_pod = current_queue_length / num_active_pods
            else:
                avg_queue_per_pod = float('inf') if current_queue_length > 0 else 0

            # 1. Calcola le repliche desiderate in teoria (può essere un valore estremo)
            if self.config.TARGET_QUEUE_LENGTH_PER_POD > 0:
                desired_replicas_raw = math.ceil(num_active_pods * (avg_queue_per_pod / self.config.TARGET_QUEUE_LENGTH_PER_POD))
            else:
                desired_replicas_raw = num_active_pods

            # --- MODIFICA CHIAVE: Applica la politica di stabilità (limita la velocità) ---
            # Limita il numero di pod da aggiungere/rimuovere in un singolo step.
            if desired_replicas_raw > num_active_pods:
                # Se vogliamo fare scale-up, non superare il massimo step consentito
                limited_step = num_active_pods + self.config.MAX_SCALE_STEP
                desired_replicas = min(desired_replicas_raw, limited_step)
            elif desired_replicas_raw < num_active_pods:
                # Se vogliamo fare scale-down, non superare il massimo step consentito
                limited_step = num_active_pods - self.config.MAX_SCALE_STEP
                desired_replicas = max(desired_replicas_raw, limited_step)
            else:
                desired_replicas = num_active_pods
            # --------------------------------------------------------------------------------

            # 2. Applica i limiti MIN e MAX globali
            desired_replicas = int(max(self.config.MIN_PODS, min(self.config.MAX_PODS, desired_replicas)))

            print(
                f"{self.env.now:.2f} [HPA]: Pods attivi: {num_active_pods}, Coda/Pod: {avg_queue_per_pod:.2f}, "
                f"Desiderate (Raw): {desired_replicas_raw}, Desiderate (Limitate): {desired_replicas}")

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