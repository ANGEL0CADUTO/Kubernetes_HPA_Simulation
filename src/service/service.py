class PodService:
    """
    Contiene la logica di business per processare le richieste.
    In questo caso, si occupa di calcolare il tempo di servizio.
    """

    def __init__(self, rng, config):
        self.rng = rng # Questo ora è il "service_rng"
        self.config = config

    def get_service_time(self, req_type):
        """
        Restituisce un tempo di servizio campionato dalla distribuzione
        corretta per il tipo di richiesta specificato.
        ORA SUPPORTA ANCHE LE DISTRIBUZIONI A MISTURA.
        """
        service_config = self.config.SERVICE_TIME_CONFIG[req_type]
        dist_type = service_config["dist"]
        params = service_config["params"]

        if dist_type == "lognormal":
            # Usiamo rng.lognormal che accetta mu e sigma (i parametri logaritmici)
            return self.rng.lognormal(mean=params[0], sigma=params[1])

        elif dist_type == "exponential":
            return self.rng.exponential(scale=params["scale"])

        # --- NUOVA LOGICA PER LA DISTRIBUZIONE A MISTURA ---
        elif dist_type == "mixture":
            # params sarà una lista di dizionari, ognuno con 'prob', 'dist', 'params'

            # Scegliamo quale distribuzione usare in base alla probabilità
            rand_choice = self.rng.random() # Genera un numero tra 0.0 e 1.0
            cumulative_prob = 0.0

            for component in params:
                cumulative_prob += component['prob']
                if rand_choice < cumulative_prob:
                    # Abbiamo scelto questo componente, ora campioniamo da esso
                    comp_dist = component['dist']
                    comp_params = component['params']

                    if comp_dist == "exponential":
                        return self.rng.exponential(scale=comp_params["scale"])
                    elif comp_dist == "lognormal":
                        return self.rng.lognormal(mean=comp_params[0], sigma=comp_params[1])
                    # (si potrebbero aggiungere altre distribuzioni qui se necessario)

            # Fallback nel caso improbabile di errore di arrotondamento
            print(f"ATTENZIONE: Nessun componente scelto nella mistura per {req_type.name}. Uso 0.1s di default.")
            return 0.1
        # --- FINE NUOVA LOGICA ---

        else:
            print(f"ATTENZIONE: Distribuzione '{dist_type}' non riconosciuta. Uso 0.1s di default.")
            return 0.1

