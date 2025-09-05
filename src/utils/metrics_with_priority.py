import pandas as pd
from collections import defaultdict
import numpy as np

from src.config import Priority, RequestType, REQUEST_TYPE_TO_PRIORITY
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


        self.requests_that_waited = 0 # Questo verrà ricalcolato post-warmup

        # response_times_history_by_prio è l'unica lista che memorizza (timestamp, value) fin dall'inizio
        self.response_times_history_by_prio = defaultdict(list)

        # Metriche di sistema
        self.timestamps = [] # Global system timestamps (e.g., for pod_counts, queue_lengths)
        self.pod_counts = []
        self.queue_lengths = []  # Lunghezza totale di tutte le code
        self.queue_lengths_per_priority = defaultdict(list)


        self.request_generation_timestamps = [] # Tutti i timestamp di generazione

        # Contatori (verranno ricalcolati/filtrati)
        self.requests_completed_by_priority = defaultdict(int)
        self.requests_generated_by_priority = defaultdict(int) # Se usato per totale, deve essere ricalcolato.
        self.requests_timed_out_by_priority = defaultdict(int)
        self.requests_timed_out_by_req_type = defaultdict(int)
        self.requests_generated_by_req_type = defaultdict(int) # Se usato per totale, deve essere ricalcolato.
        self.timeout_history = [] # Lista di (timestamp, request_type) per timeout

        # Liste raw di valori e timestamp (allineate per indice)
        self.response_times_by_priority = defaultdict(list) # List of floats
        self.wait_times_by_priority = defaultdict(list)     # List of floats
        self.completion_timestamps_by_priority = defaultdict(list) # List of floats (timestamps)

        self.response_times_by_req_type = defaultdict(list) # List of floats
        self.wait_times_by_req_type = defaultdict(list)     # List of floats
        self.completion_timestamps_by_req_type = defaultdict(list) # List of floats (timestamps)


    def record_request_generation(self, timestamp: float, priority: Priority, req_type: RequestType):
        """Registra il timestamp di quando una richiesta è generata."""
        self.request_generation_timestamps.append(timestamp)
        self.requests_generated_by_priority[priority] += 1
        self.requests_generated_by_req_type[req_type] += 1

    def record_system_metrics(self, timestamp, pod_count, queue_len, queue_len_per_prio: dict):
        """Registra lo stato del sistema a intervalli regolari."""
        self.timestamps.append(timestamp)
        self.pod_counts.append(pod_count)
        self.queue_lengths.append(queue_len)

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

        # Aggiorna gli accumulatori Welford
        self.response_times_welford_by_priority[prio].add(response_time)
        self.wait_times_welford_by_priority[prio].add(wait_time)
        self.response_times_welford_by_req_type[req_type].add(response_time)
        self.wait_times_welford_by_req_type[req_type].add(wait_time)
        self.global_welford_response.add(response_time)
        self.global_welford_wait.add(wait_time)

        # Aggiorna le liste raw per priorità
        self.requests_completed_by_priority[prio] += 1
        self.response_times_by_priority[prio].append(response_time)
        self.wait_times_by_priority[prio].append(wait_time)
        self.completion_timestamps_by_priority[prio].append(completion_time)
        self.response_times_history_by_prio[prio].append((completion_time, response_time))


        # Aggiorna le liste raw per tipo di richiesta
        self.response_times_by_req_type[req_type].append(response_time)
        self.wait_times_by_req_type[req_type].append(wait_time)
        self.completion_timestamps_by_req_type[req_type].append(completion_time)

        if wait_time > 1e-9: # Usiamo una piccola tolleranza per i float
            self.requests_that_waited += 1 # Questo verrà ricalcolato

    def record_timeout(self, request: PriorityRequest, timestamp: float):
        """Registra una richiesta che è andata in timeout."""
        self.requests_timed_out_by_priority[request.priority] += 1
        self.requests_timed_out_by_req_type[request.req_type] += 1
        self.timeout_history.append((timestamp, request.req_type))

    def record_request_failure(self, request: PriorityRequest, timestamp: float):
        """Metodo unificato per registrare una richiesta fallita (timeout o rifiutata)."""
        # Questo è essenzialmente lo stesso di record_timeout per ora
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
            if priority in welford_stats['by_priority'] and welford_stats['by_priority'][priority]['count'] > 0:
                stats = welford_stats['by_priority'][priority]
                print(f"\nPriorità {priority.name}:")
                resp_mean = stats['response_time']['mean']
                resp_std = stats['response_time']['std_dev']
                wait_mean = stats['wait_time']['mean']
                wait_std = stats['wait_time']['std_dev']

                # Gestione di None se le deviazioni standard sono None (es. count < 2)
                resp_std_str = f"{resp_std:.4f}" if resp_std is not None else "N/A"
                wait_std_str = f"{wait_std:.4f}" if wait_std is not None else "N/A"

                print(f"  - Richieste completate: {stats['count']}")
                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std_str}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std_str}s")

        print("\n--- Statistiche Welford per Tipo di Richiesta ---")
        for req_type in sorted(RequestType, key=lambda e: e.name):
            if req_type in welford_stats['by_req_type'] and welford_stats['by_req_type'][req_type]['count'] > 0:
                stats = welford_stats['by_req_type'][req_type]
                print(f"\n{req_type.name}:")
                print(f"  - Richieste completate: {stats['count']}")
                resp_mean = stats['response_time']['mean']
                resp_std = stats['response_time']['std_dev']
                wait_mean = stats['wait_time']['mean']
                wait_std = stats['wait_time']['std_dev']

                resp_std_str = f"{resp_std:.4f}" if resp_std is not None else "N/A"
                wait_std_str = f"{wait_std:.4f}" if wait_std is not None else "N/A"

                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std_str}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std_str}s")


    def get_all_response_times_with_timestamps(self):
        """
        Appiattisce i dati dei tempi di risposta da tutte le priorità
        in un'unica lista di tuple (timestamp, valore), ordinata per timestamp.
        Necessario per l'analisi Batch Means.
        """
        all_data = []
        for prio, times in self.response_times_by_priority.items(): # Questi sono già i dati post-warmup
            timestamps = self.completion_timestamps_by_priority[prio] # Questi sono già i timestamp post-warmup

            if len(timestamps) == len(times): # Questo controllo dovrebbe sempre passare dopo remove_warmup
                all_data.extend(zip(timestamps, times))
            else:
                print(f"ATTENZIONE (get_all_response_times_with_timestamps): Mismatch in lengths for {prio.name}. Data might be inconsistent.")

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
            serviced.extend([(timestamp, 0) for timestamp in timestamps])

        # 2. Raccogliamo i timestamp delle richieste perse.
        #    `self.timeout_history` salva tuple (timestamp, request_type).
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
        timed_out_history = self.timeout_history
        timed_out = [(timestamp, 1) for timestamp, rt in timed_out_history if rt == req_type_to_filter]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes

    def get_cumulative_timeouts(self):
        if not self.timeout_history:
            return [], []

        sorted_timeouts = sorted(self.timeout_history, key=lambda x: x[0])

        timestamps = [t for t, _ in sorted_timeouts] # _ al posto di r per ignorare il request_type
        cumulative_counts = np.arange(1, len(timestamps) + 1)

        return timestamps, cumulative_counts



    def remove_warmup(self, estimated_warmup_duration):
        """
        Rimuove i dati di risposta, attesa, timeout e metriche di sistema
        che avvengono prima del timestamp di fine warm-up.
        Modifica le strutture metrics in-place, garantendo l'allineamento.
        """
        print(f"Rimuovendo warmup fino a {estimated_warmup_duration:.2f}s dalla classe MetricsWithPriority.")

        # --- Aggiorna i dati per tipo di richiesta ---
        new_completion_timestamps_by_req_type = defaultdict(list)
        new_response_times_by_req_type = defaultdict(list)
        new_wait_times_by_req_type = defaultdict(list)

        for req_type in RequestType:
            timestamps = self.completion_timestamps_by_req_type.get(req_type, [])
            response_times = self.response_times_by_req_type.get(req_type, [])
            wait_times = self.wait_times_by_req_type.get(req_type, [])

            if not (len(timestamps) == len(response_times) == len(wait_times)):
                print(f"ATTENZIONE: Mismatch lunghezze per {req_type.name} (timestamps={len(timestamps)}, "
                      f"response={len(response_times)}, wait={len(wait_times)}). "
                      f"Skipping warmup removal per questo tipo di richiesta.")
                new_completion_timestamps_by_req_type[req_type] = timestamps
                new_response_times_by_req_type[req_type] = response_times
                new_wait_times_by_req_type[req_type] = wait_times
                continue

            combined_data_filtered = [
                (t, rt, wt) for t, rt, wt in zip(timestamps, response_times, wait_times)
                if t >= estimated_warmup_duration
            ]

            if combined_data_filtered:
                new_t, new_rt, new_wt = zip(*combined_data_filtered)
                new_completion_timestamps_by_req_type[req_type] = list(new_t)
                new_response_times_by_req_type[req_type] = list(new_rt)
                new_wait_times_by_req_type[req_type] = list(new_wt)

        self.completion_timestamps_by_req_type = new_completion_timestamps_by_req_type
        self.response_times_by_req_type = new_response_times_by_req_type
        self.wait_times_by_req_type = new_wait_times_by_req_type

        # --- Aggiorna i dati per priorità ---
        new_completion_timestamps_by_priority = defaultdict(list)
        new_response_times_by_priority = defaultdict(list)
        new_wait_times_by_priority = defaultdict(list)
        new_response_times_history_by_prio = defaultdict(list)

        for prio in Priority:
            timestamps = self.completion_timestamps_by_priority.get(prio, [])
            response_times = self.response_times_by_priority.get(prio, [])
            wait_times = self.wait_times_by_priority.get(prio, [])
            response_history_tuples = self.response_times_history_by_prio.get(prio, [])

            if not (len(timestamps) == len(response_times) == len(wait_times)):
                print(f"ATTENZIONE: Mismatch lunghezze per PRIORITY {prio.name} "
                      f"(timestamps={len(timestamps)}, response={len(response_times)}, wait={len(wait_times)}). "
                      f"Skipping warmup removal per questa priorità.")
                new_completion_timestamps_by_priority[prio] = timestamps
                new_response_times_by_priority[prio] = response_times
                new_wait_times_by_priority[prio] = wait_times
                new_response_times_history_by_prio[prio] = response_history_tuples
                continue

            combined_data_filtered = [
                (t, rt, wt, rt_tuple)
                for t, rt, wt, rt_tuple in zip(timestamps, response_times, wait_times, response_history_tuples)
                if t >= estimated_warmup_duration
            ]

            if combined_data_filtered:
                new_t, new_rt, new_wt, new_rt_tuple = zip(*combined_data_filtered)
                new_completion_timestamps_by_priority[prio] = list(new_t)
                new_response_times_by_priority[prio] = list(new_rt)
                new_wait_times_by_priority[prio] = list(new_wt)
                new_response_times_history_by_prio[prio] = list(new_rt_tuple)

        self.completion_timestamps_by_priority = new_completion_timestamps_by_priority
        self.response_times_by_priority = new_response_times_by_priority
        self.wait_times_by_priority = new_wait_times_by_priority
        self.response_times_history_by_prio = new_response_times_history_by_prio

        # --- Filtra timeout_history ---
        self.timeout_history = [(t, r) for (t, r) in self.timeout_history if t >= estimated_warmup_duration]

        # --- Filtra metriche di sistema ---
        if len(self.timestamps) == len(self.pod_counts) == len(self.queue_lengths):
            prio_queue_lists = {prio: self.queue_lengths_per_priority.get(prio, []) for prio in Priority}
            all_prio_match = all(len(lst) == len(self.timestamps) for lst in prio_queue_lists.values())

            if all_prio_match:
                combined = []
                for idx, t in enumerate(self.timestamps):
                    if t >= estimated_warmup_duration:
                        combined.append((t, self.pod_counts[idx], self.queue_lengths[idx],
                                         *[prio_queue_lists[prio][idx] for prio in Priority]))

                if combined:
                    self.timestamps = [c[0] for c in combined]
                    self.pod_counts = [c[1] for c in combined]
                    self.queue_lengths = [c[2] for c in combined]
                    for i, prio in enumerate(Priority):
                        self.queue_lengths_per_priority[prio] = [c[3 + i] for c in combined]
                else:
                    self.timestamps, self.pod_counts, self.queue_lengths = [], [], []
                    for prio in Priority:
                        self.queue_lengths_per_priority[prio] = []
            else:
                print("ATTENZIONE: mismatch lunghezze in queue_lengths_per_priority. "
                      "Skipping warmup removal per system history.")
        else:
            print("ATTENZIONE: mismatch lunghezze tra timestamps/pod_counts/queue_lengths. "
                  "Skipping warmup removal per system history.")

        # --- Ricostruisci accumulatori ---
        self._rebuild_welford_accumulators()

        # --- Aggiorna contatori ---
        self.requests_that_waited = 0
        self.requests_completed_by_priority = defaultdict(int)
        for prio in Priority:
            n = len(self.completion_timestamps_by_priority.get(prio, []))
            self.requests_completed_by_priority[prio] = n
            self.requests_that_waited += sum(1 for wt in self.wait_times_by_priority.get(prio, []) if wt > 1e-9)

        self.requests_generated_by_priority = defaultdict(int)
        self.requests_generated_by_req_type = defaultdict(int)

        self.requests_timed_out_by_priority = defaultdict(int)
        self.requests_timed_out_by_req_type = defaultdict(int)

        for t, rt in self.timeout_history:
            prio = None
            req_type = None

            if isinstance(rt, Priority):
                prio = rt
            elif isinstance(rt, RequestType):
                req_type = rt
                prio = REQUEST_TYPE_TO_PRIORITY.get(rt, None)
            elif hasattr(rt, "req_type"):  # caso oggetto Request
                req_type = rt.req_type
                prio = REQUEST_TYPE_TO_PRIORITY.get(req_type, None)
            elif hasattr(rt, "priority"):
                prio = rt.priority

            if prio is not None:
                self.requests_timed_out_by_priority[prio] += 1
            else:
                print(f"ATTENZIONE: impossibile dedurre Priority per timeout a t={t}, rt={rt!r}")

            if req_type is not None:
                self.requests_timed_out_by_req_type[req_type] += 1


    def _rebuild_welford_accumulators(self):
        """Ricostruisce tutti gli accumulatori Welford con i dati filtrati."""

        # Reset global accumulators
        self.global_welford_response = Welford()
        self.global_welford_wait = Welford()

        # Reset and rebuild for each RequestType
        for req_type in RequestType:
            self.response_times_welford_by_req_type[req_type] = Welford()
            self.wait_times_welford_by_req_type[req_type] = Welford()

            response_times = self.response_times_by_req_type.get(req_type, [])
            wait_times = self.wait_times_by_req_type.get(req_type, [])

            if response_times:
                self.response_times_welford_by_req_type[req_type].add_all(np.array(response_times))
                self.global_welford_response.add_all(np.array(response_times))
            if wait_times:
                self.wait_times_welford_by_req_type[req_type].add_all(np.array(wait_times))
                self.global_welford_wait.add_all(np.array(wait_times))

        # Reset and rebuild for each Priority
        for prio in Priority:
            self.response_times_welford_by_priority[prio] = Welford()
            self.wait_times_welford_by_priority[prio] = Welford()

            response_times = self.response_times_by_priority.get(prio, [])
            wait_times = self.wait_times_by_priority.get(prio, [])

            if response_times:
                self.response_times_welford_by_priority[prio].add_all(np.array(response_times))
            if wait_times:
                self.wait_times_welford_by_priority[prio].add_all(np.array(wait_times))