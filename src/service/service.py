class PodService:
    """
    Contiene la logica di business per processare le richieste.
    In questo caso, si occupa di calcolare il tempo di servizio.
    """
    def __init__(self, rng, config):
        self.rng = rng
        self.config = config

    def get_service_time(self, req_type):
        """
        Restituisce un tempo di servizio campionato dalla distribuzione
        corretta per il tipo di richiesta specificato.
        """
        service_config = self.config.SERVICE_TIME_CONFIG[req_type]
        dist_type = service_config["dist"]
        params = service_config["params"]

        if dist_type == "lognormal":
            return self.rng.lognormal(*params)
        elif dist_type == "exponential":
            return self.rng.exponential(**params)
        else:
            # Ritorna un valore di default o lancia un errore se la distribuzione non è supportata
            print(f"ATTENZIONE: Distribuzione '{dist_type}' non riconosciuta. Uso 0.1s di default.")
            return 0.1