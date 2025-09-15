# File: src/verification/service_verification.py

import numpy as np
from src.config import RequestType

class PodService:
    """
    Una versione del PodService ESCLUSIVAMENTE per la verifica.
    È progettato per accettare e utilizzare un modulo di configurazione
    "iniettato" al momento della creazione, invece di usare quello globale.
    """
    def __init__(self, service_rng: np.random.Generator, config_module):
        self.rng = service_rng
        # Salva l'intero modulo di configurazione passato al costruttore
        self.config = config_module

    def get_service_time(self, req_type: RequestType) -> float:
        """
        Genera un tempo di servizio stocastico basato sul tipo di richiesta,
        usando la configurazione locale fornita.
        """
        service_config = self.config.SERVICE_TIME_CONFIG.get(req_type)

        if not service_config:
            return 0.1

        dist_type = service_config["dist"]
        params = service_config["params"]

        if dist_type == "exponential":
            return self.rng.exponential(scale=params["scale"])
        elif dist_type == "lognormal":
            mu, sigma = params
            return self.rng.lognormal(mean=mu, sigma=sigma)
        elif dist_type == "mixture":
            probs = [p["prob"] for p in params]
            chosen_dist_index = self.rng.choice(len(params), p=probs)
            chosen_dist = params[chosen_dist_index]

            if chosen_dist["dist"] == "exponential":
                return self.rng.exponential(scale=chosen_dist["params"]["scale"])
            elif chosen_dist["dist"] == "lognormal":
                mu, sigma = chosen_dist["params"]
                return self.rng.lognormal(mean=mu, sigma=sigma)

        return 0.1