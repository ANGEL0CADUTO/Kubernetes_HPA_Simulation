from collections import defaultdict
import numpy as np
from src.config import RequestType
from src.utils.welford import Welford


class Metrics:
    """
    Classe per raccogliere e calcolare le metriche di performance durante la simulazione.
    """

    def __init__(self, config_module):
        self.config = config_module

        # Statistiche incrementali con Welford per ogni tipo di richiesta
        self.response_times_welford = defaultdict(lambda: Welford())
        self.wait_times_welford = defaultdict(lambda: Welford())
        # Welford globale per tutte le richieste (utile per statistiche aggregate)
        self.global_response_times_welford = Welford()
        self.global_wait_times_welford = Welford()

        # Metriche per validazione steady-state (non toccate da remove_warmup)
        self.system_state_history = []
        self.steady_state_detected = False
        self.steady_state_start_time = None
        self.batch_means_data = defaultdict(list)

        # Le liste conterranno tuple (timestamp, valore) per i grafici temporali. OK per filtro diretto.
        self.response_times_history = defaultdict(list)
        self.wait_times_history = defaultdict(list)

        # Liste semplici di valori (verranno ricostruite post-filtro)
        self.response_times_data = defaultdict(list)
        self.wait_times_data = defaultdict(list)

        # Metriche a livello di sistema (liste di tuple (timestamp, valore))
        self.pod_count_history = []
        self.queue_length_history = []

        # Contatori (verranno ricalcolati post-filtro)
        self.total_requests_generated = 0 # Di solito non filtrato da warmup
        self.requests_generated_data = defaultdict(int) # Di solito non filtrato da warmup

        self.total_requests_served = 0 # Verrà ricalcolato
        self.total_timeouts = 0        # Verrà ricalcolato
        self.requests_timed_out_data = defaultdict(int) # Verrà ricalcolato
        self.timeout_history = [] # Lista di (timestamp, req_type) per timeout


    def record_request_generation(self, req_type: RequestType):
        self.total_requests_generated += 1
        self.requests_generated_data[req_type] += 1

    def record_request_metrics(self, timestamp, req_type, response_time, wait_time):
        """Registra le metriche per una singola richiesta completata."""
        if response_time < 0 or wait_time < 0:
            raise ValueError(f"Tempi negativi non validi: resp={response_time}, wait={wait_time}")

        # Per i grafici temporali (già in formato (timestamp, value))
        self.response_times_history[req_type].append((timestamp, response_time))
        self.wait_times_history[req_type].append((timestamp, wait_time))

        # Uso welford per calcolo incrementale
        self.response_times_welford[req_type].add(response_time)
        self.wait_times_welford[req_type].add(wait_time)
        self.global_response_times_welford.add(response_time)
        self.global_wait_times_welford.add(wait_time)

        # Aggiunta alle liste semplici (che verranno ricostruite/filtrate se necessario)
        self.response_times_data[req_type].append(response_time)
        self.wait_times_data[req_type].append(wait_time)

        self.total_requests_served += 1 # Questo verrà ricalcolato

    def record_system_metrics(self, timestamp, pod_count, queue_length):
        """Registra lo stato del sistema a un dato istante."""
        self.pod_count_history.append((timestamp, pod_count))
        self.queue_length_history.append((timestamp, queue_length))

    def record_timeout(self, req_type: RequestType, timestamp: float):
        """Registra una richiesta che è andata in timeout."""
        self.requests_timed_out_data[req_type] += 1 # Questo verrà ricalcolato
        self.total_timeouts += 1 # Questo verrà ricalcolato
        self.timeout_history.append((timestamp, req_type))

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
                        'variance': welford_resp.var_s,
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
        print(f"Numero totale di richieste perse (timeout): {self.total_timeouts}") # Aggiunto timeout

        print("\n--- Statistiche Dettagliate per Tipo di Richiesta ---")
        stats = self.get_statistics_summary()

        for req_type in sorted(RequestType, key=lambda e: e.name):
            if req_type in stats and stats[req_type]['count'] > 0: # Controllo count per evitare stampa di N/A per tutti
                req_stats = stats[req_type]
                print(f"\n{req_type.name}:")
                print(f"  - Richieste servite: {req_stats['count']}")

                resp_mean = req_stats['response_time']['mean']
                resp_std = req_stats['response_time']['std_dev']
                wait_mean = req_stats['wait_time']['mean']
                wait_std = req_stats['wait_time']['std_dev']

                # Gestione di None se le deviazioni standard sono None (es. count < 2)
                resp_std_str = f"{resp_std:.4f}" if resp_std is not None else "N/A"
                wait_std_str = f"{wait_std:.4f}" if wait_std is not None else "N/A"

                print(f"  - Tempo di risposta medio: {resp_mean:.4f}s")
                print(f"  - Std dev tempo di risposta: {resp_std_str}s")
                print(f"  - Tempo di attesa medio: {wait_mean:.4f}s")
                print(f"  - Std dev tempo di attesa: {wait_std_str}s")


        # Statistiche globali
        if 'global' in stats: # Controlla se le statistiche globali esistono (count > 0)
            global_stats = stats['global']
            global_resp_mean = global_stats['response_time']['mean']
            global_resp_std = global_stats['response_time']['std_dev']
            global_wait_mean = global_stats['wait_time']['mean']
            global_wait_std = global_stats['wait_time']['std_dev']


            # Gestione di None per le deviazioni standard
            global_resp_std_str = f"{global_resp_std:.4f}" if global_resp_std is not None else "N/A"
            global_wait_std_str = f"{global_wait_std:.4f}" if global_wait_std is not None else "N/A"

            print(f"\n--- STATISTICHE GLOBALI ---")
            print(f"Tempo di risposta medio globale: {global_resp_mean:.4f}s")
            print(f"Std dev globale tempo di risposta: {global_resp_std_str}s")
            print(f"Tempo di attesa medio globale: {global_wait_mean:.4f}s")
            print(f"Std dev globale tempo di attesa: {global_wait_std_str}s")


        # Analisi timeout
        print("\n--- Analisi dei Timeout per Tipo di Richiesta ---")
        for req_type in sorted(self.requests_generated_data.keys(), key=lambda e: e.name):
            generated_count = self.requests_generated_data[req_type]
            # Usa il conteggio timeout filtrato
            timed_out_count = self.requests_timed_out_data.get(req_type, 0) # Usa .get per evitare KeyError

            if generated_count > 0:
                p_loss_type = timed_out_count / generated_count
                print(f"- {req_type.name:12}: {timed_out_count} persi su {generated_count} -> P_loss = {p_loss_type:.2%}")
            else:
                print(f"- {req_type.name:12}: 0 generati (o non registrati).")


    def get_all_response_times_with_timestamps(self):
        """
        Appiattisce i dati dei tempi di risposta da tutti i tipi di richiesta
        in un'unica lista di tuple (timestamp, valore), ordinata per timestamp.
        Returns:
            list: Una lista di tuple (timestamp, response_time) ordinata.
        """
        all_data = []
        for req_type_history in self.response_times_history.values():
            all_data.extend(req_type_history)
        all_data.sort(key=lambda x: x[0])
        return all_data

    def get_all_outcomes_as_binary_stream(self):
        """
        Crea una lista cronologica di tutti gli esiti (servito o perso),
        rappresentati come 0 (servito) e 1 (perso/timeout).
        """
        serviced = []
        for req_type, history in self.response_times_history.items():
            serviced.extend([(timestamp, 0) for timestamp, _ in history])

        timed_out = [(timestamp, 1) for timestamp, _ in self.timeout_history]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes

    def get_outcomes_by_type_as_binary_stream(self, req_type_to_filter: RequestType):
        """
        Crea una lista cronologica di esiti (0=servito, 1=perso) per un TIPO di richiesta specifico.
        """
        serviced_history = self.response_times_history.get(req_type_to_filter, [])
        serviced = [(timestamp, 0) for timestamp, _ in serviced_history]

        timed_out_history = self.timeout_history
        timed_out = [(timestamp, 1) for timestamp, rt in timed_out_history if rt == req_type_to_filter]

        all_outcomes = serviced + timed_out
        all_outcomes.sort(key=lambda x: x[0])
        return all_outcomes

    def get_cumulative_timeouts(self):
        if not self.timeout_history:
            return [], []

        sorted_timeouts = sorted(self.timeout_history, key=lambda x: x[0])

        timestamps = [t for t, _ in sorted_timeouts]
        cumulative_counts = np.arange(1, len(timestamps) + 1)

        return timestamps, cumulative_counts


    def remove_warmup(self, warmup_time: float):
        """
        Rimuove i dati prima del warm-up time per tutte le metriche storiche
        e ricostruisce gli accumulatori Welford e i contatori.
        """
        print(f"Rimuovendo warmup fino a {warmup_time:.2f}s dalla classe Metrics.")

        # --- 1. Filtra le history che memorizzano (timestamp, value) ---
        self.response_times_history = {
            k: [(t, v) for (t, v) in values if t >= warmup_time]
            for k, values in self.response_times_history.items()
        }
        self.wait_times_history = {
            k: [(t, v) for (t, v) in values if t >= warmup_time]
            for k, values in self.wait_times_history.items()
        }
        self.pod_count_history = [(t, v) for (t, v) in self.pod_count_history if t >= warmup_time]
        self.queue_length_history = [(t, v) for (t, v) in self.queue_length_history if t >= warmup_time]
        self.timeout_history = [(t, r) for (t, r) in self.timeout_history if t >= warmup_time]

        # --- 2. Ricostruisci le liste semplici di valori (`_data`) dalle history filtrate ---
        # Queste sono usate per le Welford, e dovrebbero riflettere solo i dati post-warmup.
        new_response_times_data = defaultdict(list)
        new_wait_times_data = defaultdict(list)
        for req_type in RequestType: # Itera su tutti i RequestType per coerenza
            new_response_times_data[req_type].extend([v for _, v in self.response_times_history.get(req_type, [])])
            new_wait_times_data[req_type].extend([v for _, v in self.wait_times_history.get(req_type, [])])
        self.response_times_data = new_response_times_data
        self.wait_times_data = new_wait_times_data


        # --- 3. Ricostruisci tutti gli accumulatori Welford con i dati filtrati ---
        self._rebuild_welford_accumulators()

        # --- 4. Ricalcola i contatori basati sui dati filtrati ---
        self.total_requests_served = sum(len(v) for v in self.response_times_history.values())
        self.total_timeouts = len(self.timeout_history) # Numero totale di timeout filtrati

        # Ricalcola requests_timed_out_data per tipo di richiesta
        new_requests_timed_out_data = defaultdict(int)
        for _, req_type in self.timeout_history: # timeout_history è (timestamp, req_type)
            new_requests_timed_out_data[req_type] += 1
        self.requests_timed_out_data = new_requests_timed_out_data

        # Nota: total_requests_generated e requests_generated_data
        # di solito non vengono tagliati dal warmup, si riferiscono
        # al totale generato nell'intera simulazione. Se si desidera
        # che riflettano solo le generazioni post-warmup, sarebbe
        # necessaria una logica aggiuntiva per filtrare request_generation_timestamps.


    def _rebuild_welford_accumulators(self):
        """Ricostruisce tutti gli accumulatori Welford con i dati correnti (filtrati)."""

        # Reset global accumulators
        self.global_response_times_welford = Welford()
        self.global_wait_times_welford = Welford()

        # Reset and rebuild for each RequestType
        for req_type in RequestType:
            self.response_times_welford[req_type] = Welford()
            self.wait_times_welford[req_type] = Welford()

            # Usa le liste semplici _data che sono state ricostruite
            response_times = self.response_times_data.get(req_type, [])
            wait_times = self.wait_times_data.get(req_type, [])

            if response_times:
                self.response_times_welford[req_type].add_all(np.array(response_times))
                self.global_response_times_welford.add_all(np.array(response_times))
            if wait_times:
                self.wait_times_welford[req_type].add_all(np.array(wait_times))
                self.global_wait_times_welford.add_all(np.array(wait_times))


    def get_response_times_by_type(self, req_type: RequestType):
        """Restituisce lista (timestamps, valori) per un dato tipo di richiesta."""
        return self.response_times_history.get(req_type, [])

    def get_all_completion_timestamps(self):
        """Restituisce tutti i timestamps di completamento (uniti per tutti i tipi)."""
        return sorted([t for history in self.response_times_history.values() for (t, _) in history])