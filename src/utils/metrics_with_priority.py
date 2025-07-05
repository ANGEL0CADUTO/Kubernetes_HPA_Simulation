# src/utils/metrics_with_priority.py

import pandas as pd
from collections import defaultdict
import numpy as np

# --- Import corretti ---
from src.config import Priority, RequestType
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
        self.queue_lengths_per_priority = defaultdict(list)

        # --- MODIFICA CHIAVE: Metriche per Priorità e TIMEOUT---
        # Usiamo defaultdict per creare automaticamente una lista per una nuova priorità
        # quando vi accediamo per la prima volta. La chiave sarà l'enum Priority.
        self.requests_completed_by_priority = defaultdict(int)
        self.requests_generated_by_priority = defaultdict(int)
        self.requests_timed_out_by_priority = defaultdict(int)
        self.requests_timed_out_by_req_type = defaultdict(int)

        self.response_times_by_priority = defaultdict(list)
        self.wait_times_by_priority = defaultdict(list)

        # Per i grafici temporali, potremmo volerli separati
        self.completion_timestamps_by_priority = defaultdict(list)
        self.completion_timestamps_by_req_type = defaultdict(list)
        self.response_times_at_completion_by_priority = defaultdict(list)
        # -----------------------------------------------

        # --- AGGIUNTA: Strutture dati per tracciare per tipo di richiesta ---
        self.response_times_by_req_type = defaultdict(list)
        self.wait_times_by_req_type = defaultdict(list)
        # -----------------------------------------------------------------

    def record_request_generation(self, timestamp: float, priority: Priority):
        """Registra il timestamp di quando una richiesta è generata."""
        self.request_generation_timestamps.append(timestamp)
        self.requests_generated_by_priority[priority] += 1

    def record_system_metrics(self, timestamp, pod_count, queue_len, queue_len_per_prio: dict):
        """Registra lo stato del sistema a intervalli regolari."""
        self.timestamps.append(timestamp)
        self.pod_counts.append(pod_count)
        self.queue_lengths.append(queue_len)

        # --- logica per salvare i nuovi dati --- #
        if queue_len_per_prio:
            for prio, length in queue_len_per_prio.items():
                self.queue_lengths_per_priority[prio].append(length)

    def record_request_metrics(self, completion_time: float, request: PriorityRequest,
                               response_time: float, wait_time: float):
        """
        Registra le metriche di una singola richiesta completata,
        catalogandole in base alla sua priorità e al suo tipo.
        """
        prio = request.priority
        req_type = request.req_type

        # Incrementa i contatori e registra i dati per Priorità
        self.requests_completed_by_priority[prio] += 1
        self.response_times_by_priority[prio].append(response_time)
        self.wait_times_by_priority[prio].append(wait_time)
        self.completion_timestamps_by_priority[prio].append(completion_time)
        self.completion_timestamps_by_req_type[req_type].append(completion_time)
        self.response_times_at_completion_by_priority[prio].append(response_time)

        # --- AGGIUNTA: Registra gli stessi dati anche per Tipo di Richiesta ---
        self.response_times_by_req_type[req_type].append(response_time)
        self.wait_times_by_req_type[req_type].append(wait_time)

        # ------------------------------------------------------------------

    def record_timeout(self, request: PriorityRequest):
        """Registra una richiesta che è andata in timeout (se implementato)."""
        self.requests_timed_out_by_priority[request.priority] += 1
        self.requests_timed_out_by_req_type[request.req_type] += 1


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
        """Stampa un riepilogo delle metriche finali, divise per priorità e tipo."""
        print("\n--- Riepilogo Metriche di Performance (con Priorità) ---")

        total_generated = len(self.request_generation_timestamps)
        total_completed = sum(self.requests_completed_by_priority.values())
        total_timeouts = sum(self.requests_timed_out_by_priority.values())

        print(f"Richieste totali generate: {total_generated}")
        print(f"Richieste totali completate: {total_completed}")
        print(f"Richieste totali perse (timeout): {total_timeouts}")
        print(f"Richieste rimaste in coda alla fine: {total_generated - total_completed - total_timeouts}")

        # --- STAMPA PER PRIORITÀ (INVARIATA) ---
        print("\n--- Dettaglio per Classe di Priorità ---")
        for prio in sorted(Priority):
            num_completed = self.requests_completed_by_priority[prio]
            generated_count = self.requests_generated_by_priority[prio]
            num_timeouts = self.requests_timed_out_by_priority[prio]

            if num_completed == 0:
                continue

            print(f"\nClasse di Priorità: {prio.name}")
            print(f"  - Richieste Servite: {num_completed}")

            avg_response_time = np.mean(self.response_times_by_priority[prio])
            avg_wait_time = np.mean(self.wait_times_by_priority[prio])
            max_response_time = max(self.response_times_by_priority[prio])

            print(f"  - Tempo di Risposta Medio: {avg_response_time:.4f}s")
            print(f"  - Tempo di Attesa Medio:   {avg_wait_time:.4f}s")
            print(f"  - Tempo di Risposta Massimo: {max_response_time:.4f}s")

            # --- P_LOSS ---
            if generated_count > 0:
                p_loss_prio = num_timeouts / generated_count
                print(f"  - P_loss Specifica:   {p_loss_prio:.2%}")

        # --- NUOVA SEZIONE DI STAMPA PER TIPO DI RICHIESTA ---
        print("\n\n--- Dettaglio per Tipo di Richiesta (per Confronto Diretto con Baseline) ---")
        print("\n--- Tempo Medio di Risposta per Tipo di Richiesta (s) ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if self.response_times_by_req_type[req_type]:
                avg_resp_time = np.mean(self.response_times_by_req_type[req_type])
                print(f"- {req_type.name:12}: {avg_resp_time:.4f}")

        print("\n--- Tempo Medio di Attesa per Tipo di Richiesta (s) ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if self.wait_times_by_req_type[req_type]:
                avg_wait_time = np.mean(self.wait_times_by_req_type[req_type])
                print(f"- {req_type.name:12}: {avg_wait_time:.4f}")
        # -------------------------------------------------------------