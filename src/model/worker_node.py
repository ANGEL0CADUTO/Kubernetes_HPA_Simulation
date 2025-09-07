import simpy
from src.utils.wfq_store import WFQStore # Per WFQ
from src.config import Priority # Utile per WFQStore se necessario

class WorkerNode:
    """
    Rappresenta un singolo Worker Node nel cluster Kubernetes come un "silo"
    indipendente.

    Ogni worker possiede e gestisce la propria coda di richieste locale e un pool
    privato di processi Pod. I pod sono legati a questo worker e non possono
    servire le code di altri worker, modellando un'architettura a risorse partizionate.
    """
    def __init__(self, env: simpy.Environment, node_id: int, queue_instance: simpy.Store | WFQStore):
        """
        Inizializza un Worker Node.

        Args:
            env: L'ambiente di simulazione SimPy.
            node_id: L'identificatore numerico univoco di questo worker.
            queue_instance: Un'istanza di coda SimPy già inizializzata (es. simpy.Store o WFQStore).
                            Questo permette di iniettare diverse logiche di scheduling (FIFO, WFQ).
        """
        self.env = env
        self.id = node_id

        # Coda di richieste locale e privata del worker.
        self.queue = queue_instance

        # Pool di pod privato: lista che contiene le istanze dei Pod (_Pod) attivi su questo worker.
        self.active_pods = []

        # Contatori per la gestione degli ID dei pod, locali a questo worker
        # per facilitare il debugging (es. pod-w1-0, pod-w1-1, ...).
        self.next_pod_id = 0
        self.available_pod_ids = set()

    def start_pod_process(self, pod_worker_function):
        """
        Avvia un nuovo processo Pod associato a questo Worker Node.

        Args:
            pod_worker_function: La funzione di processo SimPy che definisce il ciclo di vita del pod.
                                 Questa funzione DEVE accettare (pod_id, worker_instance) come argomenti.

        Returns:
            _Pod: L'istanza del pod appena creato.
        """
        pod_id = self.available_pod_ids.pop() if self.available_pod_ids else self.next_pod_id
        if not self.available_pod_ids: self.next_pod_id += 1

        # Crea il processo SimPy per questo pod, passando l'istanza del worker
        # per consentire al pod di accedere alla coda corretta.
        process = self.env.process(pod_worker_function(pod_id, self))

        new_pod = self._Pod(pod_id, process)
        self.active_pods.append(new_pod)
        return new_pod

    def stop_pod_process(self, pod_instance):
        """
        Interrompe e rimuove un processo Pod dal pool di questo Worker Node.

        Args:
            pod_instance: L'istanza del pod (_Pod) da rimuovere.
        """
        if pod_instance in self.active_pods:
            if pod_instance.process.is_alive and not pod_instance.process.triggered:
                pod_instance.process.interrupt() # Interrompe il processo SimPy

            self.available_pod_ids.add(pod_instance.id) # Rende l'ID disponibile per riutilizzo
            self.active_pods.remove(pod_instance)

    def get_total_queue_length(self):
        """Restituisce il numero totale di richieste in attesa nella coda di questo worker."""
        return len(self.queue.items)

    # --- Classe interna per rappresentare un Pod ---
    class _Pod:
        """Rappresenta un singolo Pod gestito da questo Worker Node."""
        def __init__(self, pod_id, process):
            self.id = pod_id
            self.process = process # Il processo SimPy attivo per questo pod