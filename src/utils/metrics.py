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

        # Timeout
        self.total_timeouts = 0
        # Un dizionario per contare quante richieste di ogni tipo sono state generate
        self.requests_generated_data = defaultdict(int)
        # Un dizionario per contare quante richieste di ogni tipo sono andate in timeout
        self.requests_timed_out_data = defaultdict(int)

    def record_request_generation(self, req_type: RequestType):
        self.total_requests_generated += 1
        """Registra la generazione di una richiesta, catalogandola per tipo."""
        self.requests_generated_data[req_type] += 1

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

    def record_timeout(self, req_type: RequestType):
        """Registra una richiesta che è andata in timeout."""
        self.requests_timed_out_data[req_type] += 1

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

        print("\n--- Analisi dei Timeout per Tipo di Richiesta ---")
        # Itera sui tipi di richiesta in ordine alfabetico per un output consistente
        for req_type in sorted(self.requests_generated_data.keys(), key=lambda e: e.name):
            generated_count = self.requests_generated_data[req_type]
            timed_out_count = self.requests_timed_out_data[req_type]

            if generated_count > 0:
                p_loss_type = timed_out_count / generated_count
                print(f"- {req_type.name:12}: {timed_out_count} persi su {generated_count} -> P_loss = {p_loss_type:.2%}")
            else:
                print(f"- {req_type.name:12}: 0 generati")
