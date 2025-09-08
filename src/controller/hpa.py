import math

class HPA:
    """
    Implementa un Horizontal Pod Autoscaler centralizzato che monitora
    un'architettura a Worker Node distribuiti ("silos").

    La logica di scaling è basata sull'identificazione di "hotspot" per
    garantire reattività anche in scenari di carico non bilanciato.
    """
    def __init__(self, env, simulator):
        self.env = env
        self.simulator = simulator
        self.config = simulator.config
        self.last_scale_up_time = -self.config.SCALE_UP_COOLDOWN
        self.last_scale_down_time = -self.config.SCALE_DOWN_COOLDOWN
        self.action = env.process(self.run())

    def run(self):
        while True:
            yield self.env.timeout(self.config.HPA_SYNC_PERIOD)

            worker_metrics = []
            for worker in self.simulator.worker_nodes:
                pod_count = len(worker.active_pods)
                queue_len = len(worker.queue.items)
                queue_per_pod = queue_len / pod_count if pod_count > 0 else float('inf')
                worker_metrics.append({
                    'worker': worker,
                    'pod_count': pod_count,
                    'queue_len': queue_len,
                    'queue_per_pod': queue_per_pod
                })

            if not worker_metrics:
                continue

            hotspot = max(worker_metrics, key=lambda x: x['queue_per_pod'])
            coldest = min(worker_metrics, key=lambda x: x['queue_per_pod'])
            total_pod_count = sum(m['pod_count'] for m in worker_metrics)

            # --- BLOCCO DI DECISIONE CORRETTO ---
            # Intercetta il caso di metrica infinita per prevenire l'OverflowError.
            if hotspot['queue_per_pod'] == float('inf'):
                # CASO SPECIALE: Un worker ha richieste ma 0 pod. È un'emergenza.
                # Bypassiamo il calcolo e richiediamo uno scale-up immediato dello step massimo.
                desired_replicas_raw = total_pod_count + self.config.MAX_SCALE_STEP
            elif self.config.TARGET_QUEUE_LENGTH_PER_POD > 0:
                # Calcolo standard quando le metriche sono finite e valide.
                desired_replicas_raw = math.ceil(total_pod_count * (hotspot['queue_per_pod'] / self.config.TARGET_QUEUE_LENGTH_PER_POD))
            else:
                # Fallback se la metrica target non è configurata.
                desired_replicas_raw = total_pod_count
            # --- FINE BLOCCO CORRETTO ---

            if desired_replicas_raw > total_pod_count:
                desired_replicas = min(desired_replicas_raw, total_pod_count + self.config.MAX_SCALE_STEP)
            elif desired_replicas_raw < total_pod_count:
                desired_replicas = max(desired_replicas_raw, total_pod_count - self.config.MAX_SCALE_STEP)
            else:
                desired_replicas = total_pod_count

            desired_replicas = int(max(self.config.MIN_PODS, min(self.config.MAX_PODS, desired_replicas)))

            if desired_replicas > total_pod_count:
                if self.env.now >= self.last_scale_up_time + self.config.SCALE_UP_COOLDOWN:
                    pods_to_add = desired_replicas - total_pod_count
                    new_pod_count_on_hotspot = hotspot['pod_count'] + pods_to_add
                    self.simulator.scale_worker(hotspot['worker'], new_pod_count_on_hotspot)
                    self.last_scale_up_time = self.env.now
            elif desired_replicas < total_pod_count:
                if self.env.now >= self.last_scale_down_time + self.config.SCALE_DOWN_COOLDOWN:
                    pods_to_remove = total_pod_count - desired_replicas
                    if coldest['pod_count'] > 0:
                        new_pod_count_on_coldest = max(0, coldest['pod_count'] - pods_to_remove)
                        # Assicuriamoci di non scendere sotto 1 pod se ci sono richieste in coda
                        if coldest['queue_len'] > 0 and new_pod_count_on_coldest == 0:
                            new_pod_count_on_coldest = 1
                        self.simulator.scale_worker(coldest['worker'], new_pod_count_on_coldest)
                        self.last_scale_down_time = self.env.now