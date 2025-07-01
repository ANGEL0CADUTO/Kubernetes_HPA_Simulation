# src/utils/metrics_with_priority.py

import pandas as pd
from collections import defaultdict

from src.config import Priority
from src.model.request import PriorityRequest

class MetricsWithPriority:
    """
    Raccoglie e calcola le metriche di performance per la simulazione
    con code di priorità, disaggregando i risultati per classe di priorità.
    """
    def __init__(self, config_module):
        self.config = config_module

        # Metriche di sistema (uguali alla baseline)
        self.timestamps = []
        self.pod_counts = []
        self.queue_lengths = []  # Lunghezza totale di tutte le code
        self.request_generation_timestamps = []

        # --- MODIFICA CHIAVE: Metriche per Priorità ---
        # Usiamo defaultdict per creare automaticamente una lista per una nuova priorità
        # quando vi accediamo per la prima volta. La chiave sarà l'enum Priority.
        self.requests_completed_by_priority = defaultdict(int)
        self.requests_timed_out_by_priority = defaultdict(int)

        self.response_times_by_priority = defaultdict(list)
        self.wait_times_by_priority = defaultdict(list)

        # Per i grafici temporali, potremmo volerli separati
        self.completion_timestamps_by_priority = defaultdict(list)
        self.response_times_at_completion_by_priority = defaultdict(list)
        # -----------------------------------------------

    def record_request_generation(self):
        """Registra il timestamp di quando una richiesta è generata."""
        self.request_generation_timestamps.append(self.config.env.now) # Assumendo che config abbia l'env

    def record_system_metrics(self, timestamp, pod_count, queue_len):
        """Registra lo stato del sistema a intervalli regolari."""
        self.timestamps.append(timestamp)
        self.pod_counts.append(pod_count)
        self.queue_lengths.append(queue_len)

    def record_request_metrics(self, completion_time: float, request: PriorityRequest, response_time: float, wait_time: float):
        """
        Registra le metriche di una singola richiesta completata,
        catalogandole in base alla sua priorità.
        """
        prio = request.priority

        # Incrementa i contatori
        self.requests_completed_by_priority[prio] += 1

        # Aggiungi i valori alle liste appropriate
        self.response_times_by_priority[prio].append(response_time)
        self.wait_times_by_priority[prio].append(wait_time)

        # Salva dati per grafici temporali
        self.completion_timestamps_by_priority[prio].append(completion_time)
        self.response_times_at_completion_by_priority[prio].append(response_time)

    def record_timeout(self, request: PriorityRequest, timestamp: float):
        """Registra una richiesta che è andata in timeout (se implementato)."""
        self.requests_timed_out_by_priority[request.priority] += 1

    def to_dataframe(self):
        """
        Converte le metriche di sistema in un DataFrame pandas per un'analisi più semplice.
        Potrebbe essere esteso per includere anche le metriche per priorità.
        """
        return pd.DataFrame({
            'Timestamp': self.timestamps,
            'PodCount': self.pod_counts,
            'QueueLength': self.queue_lengths
        })

    def print_summary(self):
        """Stampa un riepilogo delle metriche finali, divise per priorità."""
        print("\n--- Riepilogo Metriche di Performance (con Priorità) ---")

        total_generated = len(self.request_generation_timestamps)
        total_completed = sum(self.requests_completed_by_priority.values())
        total_timeouts = sum(self.requests_timed_out_by_priority.values())

        print(f"Richieste totali generate: {total_generated}")
        print(f"Richieste totali completate: {total_completed}")
        print(f"Richieste totali perse (timeout): {total_timeouts}")

        print("\n--- Dettaglio per Classe di Priorità ---")
        # Itera sulle priorità in ordine (HIGH, MEDIUM, LOW)
        for prio in sorted(Priority):
            if prio not in self.requests_completed_by_priority and prio not in self.requests_timed_out_by_priority:
                continue

            num_completed = self.requests_completed_by_priority[prio]
            num_timeouts = self.requests_timed_out_by_priority[prio]

            print(f"\nClasse di Priorità: {prio.name}")
            print(f"  - Richieste Servite: {num_completed}")
            print(f"  - Richieste Perse:   {num_timeouts}")

            if num_completed > 0:
                avg_response_time = sum(self.response_times_by_priority[prio]) / num_completed
                avg_wait_time = sum(self.wait_times_by_priority[prio]) / num_completed
                max_response_time = max(self.response_times_by_priority[prio])

                print(f"  - Tempo di Risposta Medio: {avg_response_time:.4f}s")
                print(f"  - Tempo di Attesa Medio:   {avg_wait_time:.4f}s")
                print(f"  - Tempo di Risposta Massimo: {max_response_time:.4f}s")