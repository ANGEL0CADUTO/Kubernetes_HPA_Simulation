import numpy as np
from src.config import TRAFFIC_PROFILE, RequestType, Priority
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority

class DynamicTrafficProfiler:
    """
    Calcola dinamicamente le probabilità del profilo di traffico basandosi
    sulla "salute" del funnel di conversione.
    """
    def __init__(self, metrics: Metrics | MetricsWithPriority, config_module):
        self.metrics = metrics
        self.config = config_module
        self.base_profile = self.config.TRAFFIC_PROFILE.copy()

        # Le dipendenze del funnel: una richiesta del tipo "chiave"
        # dipende dalla salute della richiesta del tipo "valore".
        self.funnel_dependencies = {
            RequestType.ADD_TO_CART: RequestType.NAVIGATION,
            RequestType.CHECKOUT: RequestType.NAVIGATION
        }

        # Soglia minima di richieste generate prima di iniziare ad adattare,
        # per evitare fluttuazioni estreme all'inizio della simulazione.
        self.min_data_threshold = 50

    def get_current_probabilities(self):
        """
        Restituisce la lista dei tipi di richiesta e la lista delle loro
        probabilità di arrivo, adattate in base alle performance attuali.
        """
        adjusted_profile = self.base_profile.copy()

        # Itera sulle dipendenze che abbiamo definito
        for dependent_req, source_req in self.funnel_dependencies.items():

            # Calcola la "salute" della fase sorgente del funnel
            health_factor = self._calculate_health_factor(source_req)

            # Applica il fattore di salute alla richiesta dipendente.
            # Se la navigazione fallisce (health < 1), la probabilità di
            # aggiungere al carrello diminuisce proporzionalmente.
            adjusted_profile[dependent_req] *= health_factor

        # Rinormalizza le probabilità in modo che la loro somma sia 1
        total_prob = sum(adjusted_profile.values())
        if total_prob == 0:
            # Fallback nel caso (improbabile) che tutte le probabilità siano zero
            return list(self.base_profile.keys()), list(self.base_profile.values())

        req_types = list(adjusted_profile.keys())
        req_probs = [p / total_prob for p in adjusted_profile.values()]

        return req_types, req_probs


    # ... (altre parti della classe) ...

    def _calculate_health_factor(self, req_type: RequestType) -> float:
        # Ora possiamo usare una logica unificata
        if isinstance(self.metrics, MetricsWithPriority):
            generated_count = self.metrics.requests_generated_by_req_type.get(req_type, 0)
            timed_out_count = self.metrics.requests_timed_out_by_req_type.get(req_type, 0)
        else: # Metrics
            generated_count = self.metrics.requests_generated_data.get(req_type, 0)
            timed_out_count = self.metrics.requests_timed_out_data.get(req_type, 0)

        # --- DE-INDENTA QUESTA PARTE ---
        # Questa logica deve essere eseguita per entrambi gli scenari,
        # non solo per la baseline.
        if generated_count < self.min_data_threshold:
            return 1.0

        if generated_count == 0: # Aggiungiamo un controllo per evitare la divisione per zero
            return 1.0

        p_loss = timed_out_count / generated_count
        return 1.0 - p_loss