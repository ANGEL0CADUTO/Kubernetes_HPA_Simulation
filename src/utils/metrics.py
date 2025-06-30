from collections import defaultdict
import numpy as np
from src.config import RequestType


class Metrics:
    """
    Classe per raccogliere e calcolare le metriche di performance durante la simulazione.
    """

    def __init__(self):
        # Le liste ora conterranno tuple (timestamp, valore) per i grafici temporali
        self.response_times_history = defaultdict(list)
        self.wait_times_history = defaultdict(list)

        # Le liste semplici sono ancora utili per calcolare le medie finali e gli istogrammi
        self.response_times_data = defaultdict(list)
        self.wait_times_data = defaultdict(list)

        # Metriche a livello di sistema
        self.pod_count_history = []
        self.queue_length_history = []
        self.total_requests_generated = 0
        self.total_requests_served = 0

    def record_request_generation(self):
        self.total_requests_generated += 1

    def record_request_metrics(self, timestamp, req_type, response_time, wait_time):
        """Registra le metriche per una singola richiesta completata."""
        # Per i grafici temporali
        self.response_times_history[req_type].append((timestamp, response_time))
        self.wait_times_history[req_type].append((timestamp, wait_time))

        # Per le statistiche finali e gli istogrammi
        self.response_times_data[req_type].append(response_time)
        self.wait_times_data[req_type].append(wait_time)

        self.total_requests_served += 1

    def record_system_metrics(self, timestamp, pod_count, queue_length):
        """Registra lo stato del sistema a un dato istante."""
        self.pod_count_history.append((timestamp, pod_count))
        self.queue_length_history.append((timestamp, queue_length))

    def print_summary(self):
        """Stampa un riassunto delle metriche a fine simulazione."""
        print("\n--- Riepilogo Metriche di Simulazione ---")
        print(f"Numero totale di richieste generate: {self.total_requests_generated}")
        print(f"Numero totale di richieste servite: {self.total_requests_served}")

        print("\n--- Tempo Medio di Risposta per Tipo di Richiesta (s) ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if self.response_times_data[req_type]:
                avg_resp_time = np.mean(self.response_times_data[req_type])
                print(f"- {req_type.name:12}: {avg_resp_time:.4f}")

        print("\n--- Tempo Medio di Attesa per Tipo di Richiesta (s) ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if self.wait_times_data[req_type]:
                avg_wait_time = np.mean(self.wait_times_data[req_type])
                print(f"- {req_type.name:12}: {avg_wait_time:.4f}")
