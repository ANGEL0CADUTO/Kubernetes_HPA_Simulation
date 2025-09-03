import simpy
import numpy as np
from src.config import Priority

class SPSStore:
    def __init__(self, env: simpy.Environment, weights: dict):
        self.env = env
        self.weights = weights
        self.priorities = sorted(weights.keys(), key=lambda p: p.value)
        self.queues = {p: [] for p in self.priorities}

        # Usiamo una lista di eventi, uno per ogni pod in attesa
        self.waiters = []

    @property
    def items(self):
        return [item for queue in self.queues.values() for item in queue]

    def _get_total_items(self):
        return sum(len(q) for q in self.queues.values())

    def put(self, item):
        self.queues[item.priority].append(item)
        # Se c'è un pod in attesa nella lista, svegliamo il primo
        if self.waiters:
            self.waiters.pop(0).succeed()

    def get(self):
        # Se non ci sono item in nessuna coda, il pod deve aspettare
        if self._get_total_items() == 0:
            wait_event = self.env.event()
            self.waiters.append(wait_event)
            yield wait_event

        # Se siamo stati svegliati, ora c'è sicuramente un item.
        # Questa sezione è ora atomica rispetto all'attesa.
        non_empty_queues = {p: w for p, w in self.weights.items() if self.queues[p]}

        # Fallback di sicurezza se un altro pod ha preso l'item nel frattempo
        if not non_empty_queues:
            # Questa è una chiamata ricorsiva sicura per rimettersi in attesa
            return (yield self.env.process(self.get()))

        # Logica di selezione pesata, che ora è sicura
        total_weight = sum(non_empty_queues.values())
        if total_weight > 0:
            queues_with_items = list(non_empty_queues.keys())
            weights_with_items = np.array([non_empty_queues[p] for p in queues_with_items], dtype=float)
            probabilities = weights_with_items / total_weight
            prio_to_serve = np.random.choice(queues_with_items, p=probabilities)
        else:
            # Se le uniche code con item hanno peso 0, serviamo la prima che troviamo
            prio_to_serve = next(p for p in self.priorities if self.queues[p])

        return self.queues[prio_to_serve].pop(0)

    def update_weights(self, new_weights: dict):
        print(f"{self.env.now:.2f} [SPSStore]: AGGIORNAMENTO PESI -> {{ {', '.join([f'{p.name}: {w}' for p,w in new_weights.items()])} }}")
        self.weights = new_weights