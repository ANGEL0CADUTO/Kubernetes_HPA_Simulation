import simpy
import numpy as np
from src.config import Priority

class WFQStore:
    def __init__(self, env: simpy.Environment, weights: dict):
        self.env = env
        self.weights = weights
        self.priorities = sorted(weights.keys(), key=lambda p: p.value)
        self.queues = {p: [] for p in self.priorities}
        self.waiters = []  # Eventi per i pod in attesa

    @property
    def items(self):
        return [item for queue in self.queues.values() for item in queue]

    def _get_total_items(self):
        return sum(len(q) for q in self.queues.values())

    def put(self, item):
        self.queues[item.priority].append(item)
        if self.waiters:
            self.waiters.pop(0).succeed()

    def get(self):
        # Questo è un processo generatore che deve restituire un valore.
        if self._get_total_items() == 0:
            # Se non ci sono item, il pod deve aspettare.
            # Creiamo un evento "promessa" e ci mettiamo in coda.
            wait_event = self.env.event()
            self.waiters.append(wait_event)
            yield wait_event

        # Se siamo stati svegliati, ora c'è sicuramente un item.
        non_empty_queues = {p: w for p, w in self.weights.items() if self.queues[p]}

        queues_with_items = list(non_empty_queues.keys())
        weights_with_items = np.array([non_empty_queues[p] for p in queues_with_items], dtype=float)

        probabilities = weights_with_items / np.sum(weights_with_items)
        prio_to_serve = np.random.choice(queues_with_items, p=probabilities)

        # Estraiamo l'item e usiamo 'return' per passarlo come valore del processo.
        item = self.queues[prio_to_serve].pop(0)
        return item