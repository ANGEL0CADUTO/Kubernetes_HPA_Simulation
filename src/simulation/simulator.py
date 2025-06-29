import simpy
from src.model.request import Request
from src.controller.hpa import HPA
from src.service.service import PodService

class Simulator:
    """
    Classe principale che orchestra tutti i componenti della simulazione.
    Gestisce un pool dinamico di processi 'pod_worker'.
    """

    class _Pod:
        def __init__(self, pod_id, process):
            self.id = pod_id
            self.process = process

    def __init__(self, config_module, metrics, rng):
        self.config = config_module
        self.metrics = metrics
        self.rng = rng
        self.env = simpy.Environment()

        self.service = PodService(rng, config_module)

        self.request_queue = simpy.Store(self.env)

        self.active_pods = []
        self.next_pod_id = 0
        # --- NUOVA AGGIUNTA: Pool per riutilizzare gli ID ---
        self.available_pod_ids = set()

    def request_generator(self):
        # ... (Questa funzione rimane INVARIATA) ...
        req_id_counter = 0
        req_types = list(self.config.TRAFFIC_PROFILE.keys())
        req_probs = list(self.config.TRAFFIC_PROFILE.values())

        while True:
            time_to_next = self.rng.exponential(1.0 / self.config.TOTAL_ARRIVAL_RATE)
            yield self.env.timeout(time_to_next)

            chosen_type = self.rng.choice(req_types, p=req_probs)
            req_id_counter += 1
            new_request = Request(
                request_id=req_id_counter,
                req_type=chosen_type,
                arrival_time=self.env.now
            )
            self.metrics.record_request_generation()

            print(f"{self.env.now:.2f} [Generator]: Richiesta {new_request.request_id} ({new_request.req_type.name}) generata.")
            self.request_queue.put(new_request)

    def pod_worker(self, pod_id):
        """
        Processo che simula un singolo Pod.
        Preleva richieste dalla coda e le processa finché non riceve un segnale di stop.
        """
        print(f"{self.env.now:.2f} [Pod {pod_id}]: Avviato.")
        while True:
            try:
                request = yield self.request_queue.get()

                arrival_in_service = self.env.now
                wait_time = arrival_in_service - request.arrival_time

                print(f"{self.env.now:.2f} [Pod {pod_id}]: Inizio processamento richiesta {request.request_id}. Attesa: {wait_time:.4f}s")

                service_time = self.service.get_service_time(request.req_type)
                yield self.env.timeout(service_time)

                completion_time = self.env.now # Momento in cui il servizio finisce
                response_time = completion_time - request.arrival_time
                print(f"{self.env.now:.2f} [Pod {pod_id}]: Fine processamento richiesta {request.request_id}. Tempo di risposta: {response_time:.4f}s")

                # ----- MODIFICA QUI -----
                # Passiamo anche il 'completion_time' per il grafico temporale
                self.metrics.record_request_metrics(completion_time, request.req_type, response_time, wait_time)
                # -------------------------

            except simpy.Interrupt:
                break

        print(f"{self.env.now:.2f} [Pod {pod_id}]: Rilevato segnale di stop, terminazione.")
    def metrics_recorder(self):
        # ... (Questa funzione rimane INVARIATA) ...
        while True:
            queue_len = len(self.request_queue.items)
            pod_count = len(self.active_pods)
            self.metrics.record_system_metrics(
                self.env.now, pod_count, queue_len
            )
            yield self.env.timeout(1)

    def get_busy_pods_count(self):
        # ... (Questa funzione rimane INVARIATA) ...
        num_pods_waiting_for_request = len(self.request_queue.get_queue)
        num_active_pods = len(self.active_pods)
        num_busy_pods = num_active_pods - num_pods_waiting_for_request
        return max(0, num_busy_pods)

    def scale_to(self, desired_replicas):
        """
        Adegua il numero di processi 'pod_worker' al numero desiderato,
        gestendo correttamente gli ID.
        """
        current_replicas = len(self.active_pods)

        if desired_replicas > current_replicas:
            # Scale-Up
            num_to_add = desired_replicas - current_replicas
            print(f"{self.env.now:.2f} [Simulator]: Aggiungo {num_to_add} Pods...")
            for _ in range(num_to_add):
                # --- LOGICA ID MIGLIORATA ---
                if self.available_pod_ids:
                    # Riutilizza un ID dal pool di quelli disponibili
                    pod_id = self.available_pod_ids.pop()
                else:
                    # Se non ci sono ID da riutilizzare, ne crea uno nuovo
                    pod_id = self.next_pod_id
                    self.next_pod_id += 1
                # -------------------------

                process = self.env.process(self.pod_worker(pod_id))
                self.active_pods.append(self._Pod(pod_id, process))

        elif desired_replicas < current_replicas:
            # Scale-Down
            num_to_remove = current_replicas - desired_replicas
            print(f"{self.env.now:.2f} [Simulator]: Rimuovo {num_to_remove} Pods...")

            pods_to_remove = self.active_pods[-num_to_remove:]
            for pod in pods_to_remove:
                # Interrompe il processo (se è in attesa)
                if pod.process.is_alive and not pod.process.triggered:
                    pod.process.interrupt()
                # --- LOGICA ID MIGLIORATA ---
                # Aggiunge l'ID del pod rimosso al pool di quelli disponibili
                self.available_pod_ids.add(pod.id)
                # -------------------------

            # Aggiorna la lista dei pod attivi
            self.active_pods = self.active_pods[:-num_to_remove]

    def run(self):
        # ... (Questa funzione rimane INVARIATA) ...
        print("--- Avvio Simulatore (Versione con ID Corretti Definitiva) ---")
        print(f"Modello base: Iniziali {self.config.INITIAL_PODS} Pods (Max: {self.config.MAX_PODS}), Coda FIFO, HPA {'Abilitato' if self.config.HPA_ENABLED else 'Disabilitato'}")

        self.env.process(self.request_generator())
        self.env.process(self.metrics_recorder())

        self.scale_to(self.config.INITIAL_PODS)

        if self.config.HPA_ENABLED:
            HPA(self.env, self)

        self.env.run(until=self.config.SIMULATION_TIME)
        print("--- Simulazione Terminata ---")