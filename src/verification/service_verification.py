import numpy as np

from src.config import RequestType

class PodService:
    """
    Modella il tempo di servizio per le diverse tipologie di richieste.
    """
    def __init__(self, service_rng: np.random.Generator, config_module):
        self.rng = service_rng
        self.config = config_module

    def get_service_time(self, req_type: RequestType) -> float:
        """
        Genera un tempo di servizio stocastico basato sul tipo di richiesta.
        """
        service_config = self.config.SERVICE_TIME_CONFIG.get(req_type)

        if not service_config:
            # Fallback se il tipo di richiesta non è in configurazione
            return 0.1

        dist_type = service_config["dist"]
        params = service_config["params"]

        if dist_type == "exponential":
            return self.rng.exponential(scale=params["scale"])
        elif dist_type == "lognormal":
            mu, sigma = params
            return self.rng.lognormal(mean=mu, sigma=sigma)
        elif dist_type == "mixture":
            # Scegli una delle distribuzioni della mistura
            probs = [p["prob"] for p in params]
            chosen_dist_index = self.rng.choice(len(params), p=probs)
            chosen_dist = params[chosen_dist_index]

            # Genera il tempo dalla distribuzione scelta
            if chosen_dist["dist"] == "exponential":
                return self.rng.exponential(scale=chosen_dist["params"]["scale"])
            elif chosen_dist["dist"] == "lognormal":
                mu, sigma = chosen_dist["params"]
                return self.rng.lognormal(mean=mu, sigma=sigma)

        return 0.1 # Fallback
