from collections import defaultdict
import numpy as np
from src.config import RequestType
from src.utils.welford import Welford


class Metrics:
    """
    Classe per raccogliere e calcolare le metriche di performance durante la simulazione.
    """

    def __init__(self,config_module):
        """ NUOVO"""
        #Statistiche incrementali con Welford per ogni tipo di richiesta
        self.response_times_welford= defaultdict(lambda: Welford())
        self.wait_times_welford = defaultdict(lambda: Welford())
        # Welford globale per tutte le richieste (utile per statistiche aggregate)
        self.global_response_times_welford = Welford()
        self.global_wait_times_welford = Welford()
        # Metriche per validazione steady-state
        self.system_state_history = []  # Per monitorare convergenza
        self.steady_state_detected = False
        self.steady_state_start_time = None
        # Per analisi batch means
        self.batch_means_data = defaultdict(list)
        """FINE NUOVO"""

        # Le liste ora conterranno tuple (timestamp, valore) per i grafici temporali
        self.response_times_history = defaultdict(list)
        self.wait_times_history = defaultdict(list)
        self.config = config_module
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
        # Validazione input
        if response_time < 0 or wait_time < 0:
            raise ValueError(f"Tempi negativi non validi: resp={response_time}, wait={wait_time}")

        # Per i grafici temporali
        self.response_times_history[req_type].append((timestamp, response_time))
        self.wait_times_history[req_type].append((timestamp, wait_time))

        #Uso welford per calcolo incrementale
        self.response_times_welford[req_type].add(response_time)
        self.wait_times_welford[req_type].add(wait_time)
        self.global_response_times_welford.add(response_time)
        self.global_wait_times_welford.add(wait_time)

        self.total_requests_served += 1

    def record_system_metrics(self, timestamp, pod_count, queue_length):
        """Registra lo stato del sistema a un dato istante."""
        self.pod_count_history.append((timestamp, pod_count))
        self.queue_length_history.append((timestamp, queue_length))

    def record_timeout(self, req_type: RequestType, timestamp: float):
        """Registra una richiesta che è andata in timeout."""
        self.requests_timed_out_data[req_type] += 1
        self.timeout_history.append((timestamp, req_type)) # <-- NUOVO

    def get_statistics_summary(self):
        """
        Restituisce un dizionario con tutte le statistiche calcolate incrementalmente.
        """
        summary = {}

        for req_type in RequestType:
            welford_resp = self.response_times_welford[req_type]
            welford_wait = self.wait_times_welford[req_type]

            if welford_resp.count > 0:
                summary[req_type] = {
                    'count': welford_resp.count,
                    'response_time': {
                        'mean': welford_resp.mean,
                        'variance': welford_resp.var_s,  # Sample variance
                        'std_dev': np.sqrt(welford_resp.var_s) if welford_resp.var_s is not None else None
                    },
                    'wait_time': {
                        'mean': welford_wait.mean,
                        'variance': welford_wait.var_s,
                        'std_dev': np.sqrt(welford_wait.var_s) if welford_wait.var_s is not None else None
                    }
                }

        # Statistiche globali
        if self.global_response_times_welford.count > 0:
            summary['global'] = {
                'response_time': {
                    'mean': self.global_response_times_welford.mean,
                    'variance': self.global_response_times_welford.var_s,
                    'std_dev': np.sqrt(self.global_response_times_welford.var_s)
                    if self.global_response_times_welford.var_s is not None else None
                },
                'wait_time': {
                    'mean': self.global_wait_times_welford.mean,
                    'variance': self.global_wait_times_welford.var_s,
                    'std_dev': np.sqrt(self.global_wait_times_welford.var_s)
                    if self.global_wait_times_welford.var_s is not None else None
                }
            }

        return summary


    def print_summary(self):
        """Stampa un riassunto delle metriche usando le statistiche di Welford."""
        print("\n--- Riepilogo Metriche di Performance (Simulazione Baseline con Welford) ---")
        print(f"Numero totale di richieste generate: {self.total_requests_generated}")
        print(f"Numero totale di richieste servite: {self.total_requests_served}")

        print("\n--- Statistiche Dettagliate per Tipo di Richiesta ---")
        stats = self.get_statistics_summary()

        for req_type in sorted(RequestType, key=lambda e: e.name):
            if req_type in stats:
                req_stats = stats[req_type]
                print(f"\n{req_type.name}:")
                print(f"  - Richieste servite: {req_stats['count']}")
                # Conversione in float per evitare TypeError
                resp_mean = float(req_stats['response_time']['mean'])
                resp_std = float(req_stats['response_time']['std_dev'])
                wait_mean = float(req_stats['wait_time']['mean'])
                wait_std = float(req_stats['wait_time']['std_dev'])
                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std:.4f}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std:.4f}s")


        # Statistiche globali
        if 'global' in stats:
            global_stats = stats['global']
            global_resp_mean = float(global_stats['response_time']['mean'])
            global_resp_std = float(global_stats['response_time']['std_dev'])

            print(f"\n--- STATISTICHE GLOBALI ---")
            print(f"Tempo di risposta medio globale: {global_resp_mean:.4f}s")
            print(f"Std dev globale tempo di risposta: {global_resp_std:.4f}s")


        # Analisi timeout (invariata)
        print("\n--- Analisi dei Timeout per Tipo di Richiesta ---")
        for req_type in sorted(self.requests_generated_data.keys(), key=lambda e: e.name):
            generated_count = self.requests_generated_data[req_type]
            timed_out_count = self.requests_timed_out_data[req_type]

            if generated_count > 0:
                p_loss_type = timed_out_count / generated_count
                print(f"- {req_type.name:12}: {timed_out_count} persi su {generated_count} -> P_loss = {p_loss_type:.2%}")

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

    def get_cumulative_timeouts(self):
        if not self.timeout_history:
            return [], []

        sorted_timeouts = sorted(self.timeout_history, key=lambda x: x[0])

        timestamps = [t for t, r in sorted_timeouts]
        cumulative_counts = np.arange(1, len(timestamps) + 1)

        return timestamps, cumulative_counts


    #Analisi secondo Kurkoswki
    def remove_warmup(self, warmup_time: float):
        """Rimuove dati prima del warm-up time."""
        self.response_times_history = {
            k: [(t, v) for (t, v) in values if t >= warmup_time]
            for k, values in self.response_times_history.items()
        }
        self.wait_times_history = {
            k: [(t, v) for (t, v) in values if t >= warmup_time]
            for k, values in self.wait_times_history.items()
        }
        self.timeout_history = [(t, r) for (t, r) in self.timeout_history if t >= warmup_time]

    def compute_batch_means(self, batch_size: int):
        data = [v for _, v in self.get_all_response_times_with_timestamps()]
        if len(data) < batch_size:
            raise ValueError("Dati insufficienti per batch means")
        num_batches = len(data) // batch_size
        batches = np.array(data[:num_batches * batch_size]).reshape(num_batches, batch_size)
        return batches.mean(axis=1)

    def check_steady_state(self, batch_means, tolerance=0.05):
        """Verifica se le ultime due medie batch sono entro la tolleranza."""
        if len(batch_means) < 2:
            return False
        last, prev = batch_means[-1], batch_means[-2]
        return abs(last - prev) / prev < tolerance

    def get_response_times_by_type(self, req_type: RequestType):
        """Restituisce lista (timestamps, valori) per un dato tipo di richiesta."""
        return self.response_times_history.get(req_type, [])

    def get_all_completion_timestamps(self):
        """Restituisce tutti i timestamps di completamento (uniti per tutti i tipi)."""
        return sorted([t for history in self.response_times_history.values() for (t, _) in history])


