# File: src/utils/wfq_store.py

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

        # La coda è una min-heap che ordina le richieste per il loro finish_tag.
        # Formato tupla: (finish_tag, arrival_order, request_object)
        self.queue = []
        self.waiters = [] # Eventi per i pod in attesa

        self.virtual_time = 0.0
        self.arrival_counter = 0

    @property
    def items(self):
        """Restituisce una lista di tutti gli oggetti richiesta in coda."""
        return [item_tuple[2] for item_tuple in self.queue]

    def _update_virtual_time_on_get(self):
        """Il tempo virtuale avanza al finish tag del prossimo item da servire."""
        if self.queue:
            next_finish_tag = self.queue[0][0]
            self.virtual_time = max(self.virtual_time, next_finish_tag)

    def put(self, item):
        """Aggiunge una richiesta, calcola il suo Finish Tag e la inserisce in coda."""
        # Se un pod sta aspettando, la coda era vuota. Il tempo virtuale
        # deve essere aggiornato al tempo reale per evitare che rimanga indietro.
        if self.waiters:
            self.virtual_time = max(self.virtual_time, self.env.now)

        weight = self.weights.get(item.priority, 1)
        if weight <= 0:
            finish_tag = float('inf')
        else:
            # Il Finish Tag è la somma del tempo di inizio virtuale e del costo normalizzato
            start_tag = max(item.arrival_time, self.virtual_time)
            finish_tag = start_tag + item.service_time / weight

        heapq.heappush(self.queue, (finish_tag, self.arrival_counter, item))
        self.arrival_counter += 1

        if self.waiters:
            self.waiters.pop(0).succeed()

    def get(self):
        """Estrae la richiesta con il Finish Tag più basso."""
        if not self.queue:
            wait_event = self.env.event()
            self.waiters.append(wait_event)
            yield wait_event

        # Estrai l'item con il finish_tag più basso
        finish_tag, _, item = heapq.heappop(self.queue)

        # Aggiorna il tempo virtuale del sistema a quello dell'item appena servito
        self.virtual_time = finish_tag

        return item

    def update_weights(self, new_weights: dict):
        """Permette di cambiare i pesi dello scheduler dinamicamente."""
        print(f"{self.env.now:.2f} [WFQStore]: AGGIORNAMENTO PESI -> {{ {', '.join([f'{p.name}: {w}' for p,w in new_weights.items()])} }}")
        self.weights = new_weights