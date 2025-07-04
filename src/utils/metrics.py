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
        self.timeout_history = [] # <-- NUOVO: per salvare i timestamp dei timeout


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

    def record_timeout(self, req_type: RequestType, timestamp: float):
        """Registra una richiesta che è andata in timeout."""
        self.requests_timed_out_data[req_type] += 1
        self.timeout_history.append((timestamp, req_type)) # <-- NUOVO


    def print_summary(self):
        """Stampa un riassunto delle metriche a fine simulazione."""
        print("\n--- Riepilogo Metriche di Performance (Simulazione Baseline) ---")
        print(f"Numero totale di richieste generate: {self.total_requests_generated}")
        print(f"Numero totale di richieste servite: {self.total_requests_served}")

        print("\n--- Numero di Richieste Servite per Tipo ---")
    # Itera sui tipi di richiesta in ordine per un output consistente
        for req_type in sorted(RequestType, key=lambda e: e.name):
            # Usiamo self.response_times_data perché registra solo le richieste completate
            served_count = len(self.response_times_data.get(req_type, []))
            print(f"- {req_type.name:12}: {served_count}")

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

    def get_all_response_times_with_timestamps(self):
        """
        Appiattisce i dati dei tempi di risposta da tutti i tipi di richiesta
        in un'unica lista di tuple (timestamp, valore), ordinata per timestamp.
        Questo è un prerequisito per l'analisi Batch Means.

        Returns:
            list: Una lista di tuple (timestamp, response_time) ordinata.
        """
        # 1. self.response_times_history è un dizionario dove le chiavi sono RequestType
        #    e i valori sono liste di tuple (timestamp, valore).
        #    Dobbiamo solo unire tutte queste liste in una sola.
        all_data = []
        for req_type_history in self.response_times_history.values():
            all_data.extend(req_type_history)

        # 2. Ordina la lista combinata in base al timestamp, che è il primo
        #    elemento (indice 0) di ogni tupla. Questo è fondamentale per
        #    l'analisi temporale e per la rimozione corretta del warm-up.
        all_data.sort(key=lambda x: x[0])

        return all_data

    def get_all_outcomes_as_binary_stream(self):
        """
        Crea una lista cronologica di tutti gli esiti (servito o perso),
        rappresentati come 0 (servito) e 1 (perso/timeout).
        """
        # Richieste servite (esito = 0)
        serviced = []
        for req_type, history in self.response_times_history.items():
            serviced.extend([(timestamp, 0) for timestamp, _ in history])

        # Richieste perse (esito = 1)
        timed_out = [(timestamp, 1) for timestamp, _ in self.timeout_history]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes

    def get_outcomes_by_type_as_binary_stream(self, req_type_to_filter: RequestType):
        """
        Crea una lista cronologica di esiti (0=servito, 1=perso) per un TIPO di richiesta specifico.
        """
        # Richieste servite di questo tipo
        serviced_history = self.response_times_history.get(req_type_to_filter, [])
        serviced = [(timestamp, 0) for timestamp, _ in serviced_history]

        # Richieste perse di questo tipo
        # Assumendo che timeout_history contenga tuple (timestamp, req_type)
        timed_out_history = self.timeout_history
        timed_out = [(timestamp, 1) for timestamp, req_type in timed_out_history if req_type == req_type_to_filter]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes
