import simpy

class WorkerNode:
    """
    Rappresenta un singolo Worker Node nel cluster Kubernetes.

    Ogni worker ha un proprio identificatore e, soprattutto, una propria coda
    locale dove le richieste vengono accodate dal Dispatcher.
    """
    def __init__(self, env: simpy.Environment, node_id: int):
        """
        Inizializza un Worker Node.

        Args:
            env: L'ambiente di simulazione SimPy.
            node_id: L'identificatore numerico di questo worker.
        """
        self.env = env
        self.id = node_id

        # Ogni worker ha la sua coda FIFO locale.
        self.queue = simpy.Store(self.env)

        # Potremmo aggiungere qui altre metriche specifiche del worker in futuro.