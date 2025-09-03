import pandas as pd
from collections import defaultdict
import numpy as np

from src.config import Priority, RequestType
from src.model.request import PriorityRequest
from src.utils.welford import Welford


class MetricsWithPriority:
    """
    Raccoglie e calcola le metriche di performance per la simulazione
    con code di priorità, disaggregando i risultati per classe di priorità.
    """
    def __init__(self, config_module):
        self.config = config_module

        # Welford per statistiche incrementali per priorità
        self.response_times_welford_by_priority = defaultdict(lambda: Welford())
        self.wait_times_welford_by_priority = defaultdict(lambda: Welford())

        # Welford per statistiche incrementali per tipo di richiesta
        self.response_times_welford_by_req_type = defaultdict(lambda: Welford())
        self.wait_times_welford_by_req_type = defaultdict(lambda: Welford())

        # Welford globale
        self.global_welford_response = Welford()
        self.global_welford_wait = Welford()


        self.requests_that_waited = 0
        self.response_times_history_by_prio = defaultdict(list)

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
        self.requests_generated_by_req_type = defaultdict(int)
        self.timeout_history = [] # <-- NUOVO

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


    def record_request_generation(self, timestamp: float, priority: Priority, req_type: RequestType):
        """Registra il timestamp di quando una richiesta è generata."""
        self.request_generation_timestamps.append(timestamp)
        self.requests_generated_by_priority[priority] += 1
        # Usa il parametro 'req_type' invece della variabile inesistente 'request'
        self.requests_generated_by_req_type[req_type] += 1

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

        #  Welford per priorità
        self.response_times_welford_by_priority[prio].add(response_time)
        self.wait_times_welford_by_priority[prio].add(wait_time)

        # Welford per tipo di richiesta
        self.response_times_welford_by_req_type[req_type].add(response_time)
        self.wait_times_welford_by_req_type[req_type].add(wait_time)

        # Welford globale
        self.global_welford_response.add(response_time)
        self.global_welford_wait.add(wait_time)


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
        if wait_time > 1e-9: # Usiamo una piccola tolleranza per i float
            self.requests_that_waited += 1

        self.response_times_history_by_prio[request.priority].append((completion_time, response_time))



    def record_timeout(self, request: PriorityRequest, timestamp: float):
        """Registra una richiesta che è andata in timeout (se implementato)."""
        self.requests_timed_out_by_priority[request.priority] += 1
        self.requests_timed_out_by_req_type[request.req_type] += 1
        self.timeout_history.append((timestamp, request.req_type))   # <-- NUOVO

    def record_request_failure(self, request: PriorityRequest, timestamp: float):
        """Metodo unificato per registrare una richiesta fallita (timeout o rifiutata)."""
        self.requests_timed_out_by_priority[request.priority] += 1
        self.requests_timed_out_by_req_type[request.req_type] += 1
        self.timeout_history.append((timestamp, request.req_type))


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

    def get_welford_statistics(self):
        """
        Restituisce tutte le statistiche calcolate con Welford.
        """
        stats = {
            'by_priority': {},
            'by_req_type': {},
            'global': {}
        }

        # Statistiche per priorità
        for priority in Priority:
            resp_welford = self.response_times_welford_by_priority[priority]
            wait_welford = self.wait_times_welford_by_priority[priority]

            if resp_welford.count > 0:
                stats['by_priority'][priority] = {
                    'count': resp_welford.count,
                    'response_time': {
                        'mean': resp_welford.mean,
                        'variance': resp_welford.var_s,
                        'std_dev': np.sqrt(resp_welford.var_s) if resp_welford.var_s is not None else None
                    },
                    'wait_time': {
                        'mean': wait_welford.mean,
                        'variance': wait_welford.var_s,
                        'std_dev': np.sqrt(wait_welford.var_s) if wait_welford.var_s is not None else None
                    }
                }

        # Statistiche per tipo di richiesta
        for req_type in RequestType:
            resp_welford = self.response_times_welford_by_req_type[req_type]
            wait_welford = self.wait_times_welford_by_req_type[req_type]

            if resp_welford.count > 0:
                stats['by_req_type'][req_type] = {
                    'count': resp_welford.count,
                    'response_time': {
                        'mean': resp_welford.mean,
                        'variance': resp_welford.var_s,
                        'std_dev': np.sqrt(resp_welford.var_s) if resp_welford.var_s is not None else None
                    },
                    'wait_time': {
                        'mean': wait_welford.mean,
                        'variance': wait_welford.var_s,
                        'std_dev': np.sqrt(wait_welford.var_s) if wait_welford.var_s is not None else None
                    }
                }

        # Statistiche globali
        if self.global_welford_response.count > 0:
            stats['global'] = {
                'response_time': {
                    'mean': self.global_welford_response.mean,
                    'variance': self.global_welford_response.var_s,
                    'std_dev': np.sqrt(self.global_welford_response.var_s)
                    if self.global_welford_response.var_s is not None else None
                },
                'wait_time': {
                    'mean': self.global_welford_wait.mean,
                    'variance': self.global_welford_wait.var_s,
                    'std_dev': np.sqrt(self.global_welford_wait.var_s)
                    if self.global_welford_wait.var_s is not None else None
                }
            }

        return stats

    def print_summary(self):
        """Stampa migliorato che usa le statistiche di Welford."""
        print("\n--- Riepilogo Metriche di Performance (con Priorità + Welford) ---")

        total_generated = len(self.request_generation_timestamps)
        total_completed = sum(self.requests_completed_by_priority.values())
        total_timeouts = sum(self.requests_timed_out_by_priority.values())

        print(f"Richieste totali generate: {total_generated}")
        print(f"Richieste totali completate: {total_completed}")
        print(f"Richieste totali perse (timeout): {total_timeouts}")

        # Ottieni statistiche Welford
        welford_stats = self.get_welford_statistics()

        print("\n--- Statistiche Welford per Classe di Priorità ---")
        for priority in sorted(Priority):
            if priority in welford_stats['by_priority']:
                stats = welford_stats['by_priority'][priority]
                print(f"\nPriorità {priority.name}:")
                resp_mean = float(stats['response_time']['mean'])
                resp_std = float(stats['response_time']['std_dev'])
                wait_mean = float(stats['wait_time']['mean'])
                wait_std = float(stats['wait_time']['std_dev'])

                print(f"  - Richieste completate: {stats['count']}")
                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std:.4f}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std:.4f}s")

        print("\n--- Statistiche Welford per Tipo di Richiesta ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if req_type in welford_stats['by_req_type']:
                stats = welford_stats['by_req_type'][req_type]
                print(f"\n{req_type.name}:")
                print(f"  - Richieste completate: {stats['count']}")
                resp_mean = float(stats['response_time']['mean'])
                resp_std = float(stats['response_time']['std_dev'])
                wait_mean = float(stats['wait_time']['mean'])
                wait_std = float(stats['wait_time']['std_dev'])
                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std:.4f}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std:.4f}s")




    def get_all_response_times_with_timestamps(self):
        """
        Appiattisce i dati dei tempi di risposta da tutte le priorità
        in un'unica lista di tuple (timestamp, valore), ordinata per timestamp.
        Necessario per l'analisi Batch Means.
        """
        all_data = []
        for prio, times in self.response_times_by_priority.items():
            timestamps = self.completion_timestamps_by_priority[prio]
            # Si assicura che le lunghezze corrispondano
            if len(timestamps) == len(times):
                all_data.extend(zip(timestamps, times))

        # Ordina per timestamp, che è il primo elemento della tupla
        all_data.sort(key=lambda x: x[0])
        return all_data

    def get_all_outcomes_as_binary_stream(self):
        """
        Crea una lista cronologica di tutti gli esiti (servito o perso),
        rappresentati come 0 (servito) e 1 (perso/timeout).
        """
        # 1. Raccogliamo i timestamp di tutte le richieste servite.
        serviced = []
        for req_type, timestamps in self.completion_timestamps_by_req_type.items():
            # Per ogni timestamp di completamento, registriamo un successo (0).
            serviced.extend([(timestamp, 0) for timestamp in timestamps])

        # 2. Raccogliamo i timestamp delle richieste perse.
        #    `self.timeout_history` salva tuple (timestamp, request).
        timed_out = [(timestamp, 1) for timestamp, _ in self.timeout_history]

        # 3. Combiniamo e ordiniamo cronologicamente.
        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])

        return all_outcomes

    def get_outcomes_by_type_as_binary_stream(self, req_type_to_filter: RequestType):
        """
        Crea una lista cronologica di esiti (0=servito, 1=perso) per un TIPO di richiesta specifico.
        """
        # Richieste servite di questo tipo
        serviced_timestamps = self.completion_timestamps_by_req_type.get(req_type_to_filter, [])
        serviced = [(timestamp, 0) for timestamp in serviced_timestamps]

        # Richieste perse di questo tipo
        # Assumendo che timeout_history contenga tuple (timestamp, request_object)
        timed_out_history = self.timeout_history
        timed_out = [(timestamp, 1) for timestamp, req_type in timed_out_history if req_type == req_type_to_filter]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes
    def get_cumulative_timeouts(self):

        if not self.timeout_history:
            return [], []

        sorted_timeouts = sorted(self.timeout_history, key=lambda x: x[0])

        timestamps = [t for t, r in sorted_timeouts]
        cumulative_counts = np.arange(1, len(timestamps) + 1)

        return timestamps, cumulative_counts
