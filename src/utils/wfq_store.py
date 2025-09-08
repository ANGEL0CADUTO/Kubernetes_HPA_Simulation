import simpy
import heapq
import numpy as np
from src.config import Priority

class WFQStore:
    """
    Implementazione di una coda Weighted Fair Queuing (WFQ) basata sulla
    teoria standard dei Finish Tag, per SimPy.
    """
    def __init__(self, env: simpy.Environment, weights: dict):
        self.env = env
        self.weights = weights

        self.queue = []
        self.waiters = []

        self.virtual_time = 0.0
        self.arrival_counter = 0

    @property
    def items(self):
        """Restituisce una lista di tutti gli oggetti richiesta in coda."""
        return [item_tuple[2] for item_tuple in self.queue]

    def _update_virtual_time_on_get(self):
        if self.queue:
            next_finish_tag = self.queue[0][0]
            self.virtual_time = max(self.virtual_time, next_finish_tag)

    def put(self, item):
        if self.waiters:
            self.virtual_time = max(self.virtual_time, self.env.now)

        weight = self.weights.get(item.priority, 1)
        if weight <= 0:
            finish_tag = float('inf')
        else:
            start_tag = max(item.arrival_time, self.virtual_time)
            finish_tag = start_tag + item.service_time / weight

        heapq.heappush(self.queue, (finish_tag, self.arrival_counter, item))
        self.arrival_counter += 1

        if self.waiters:
            self.waiters.pop(0).succeed()

    def get(self):
        if not self.queue:
            wait_event = self.env.event()
            self.waiters.append(wait_event)
            yield wait_event

        finish_tag, _, item = heapq.heappop(self.queue)
        self.virtual_time = finish_tag
        return item

    def update_weights(self, new_weights: dict):
        print(f"{self.env.now:.2f} [WFQStore]: AGGIORNAMENTO PESI -> {{ {', '.join([f'{p.name}: {w}' for p,w in new_weights.items()])} }}")
        self.weights = new_weights

    def purge_by_priority(self, priority_to_purge: Priority):
        """
        NUOVO METODO: Rimuove tutte le richieste di una specifica priorità dalla coda.
        Questa è un'operazione O(N) in quanto richiede di ricostruire la heap.

        Args:
            priority_to_purge: La classe di priorità da rimuovere.

        Returns:
            list: Una lista degli oggetti richiesta che sono stati rimossi.
        """
        purged_items = []
        # Crea una nuova lista contenente solo gli item che NON devono essere rimossi.
        remaining_items = []
        for item_tuple in self.queue:
            request_item = item_tuple[2]
            if request_item.priority == priority_to_purge:
                purged_items.append(request_item)
            else:
                remaining_items.append(item_tuple)

        # Se abbiamo rimosso qualcosa, ricostruiamo la heap.
        if len(purged_items) > 0:
            self.queue = remaining_items
            heapq.heapify(self.queue) # Ripristina l'invariante della heap

        return purged_items