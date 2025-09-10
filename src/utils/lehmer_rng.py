

import numpy as np
import math

class LehmerRNG:
    """
    Gestisce la creazione di stream di numeri casuali statisticamente indipendenti
    per le varie fonti di incertezza della simulazione.

    Utilizza un generatore Lehmer "master" per garantire la riproducibilità
    dell'intero set di esperimenti a partire da un singolo seed.
    Ogni stream è un'istanza indipendente di numpy.random.Generator.
    """

    # Definiamo i nomi degli stream che la nostra simulazione necessita.
    # Questo rende il codice più chiaro e facile da estendere.
    STREAM_NAMES = ['arrivals', 'choice', 'service']

    class _LehmerRNG:
        """
        Implementazione interna e privata del generatore Lehmer.
        Il suo unico scopo è generare i seed per la SeedSequence di NumPy.
        """
        def __init__(self, seed):
            self.seed = seed
            self.m = 2**31 - 1
            self.a = 48271

        def _next_seed(self):
            """Genera il prossimo seed nella sequenza."""
            self.seed = (self.a * self.seed) % self.m
            return self.seed

    def __init__(self, master_seed: int):
        """
        Inizializza il gestore con un singolo seed master.

        Args:
            master_seed: Il seed iniziale per il generatore Lehmer.
        """
        self._master_rng = self._LehmerRNG(seed=master_seed)
        # "Riscalda" il generatore per evitare i primi valori, come da best practice.
        for _ in range(100):
            self._master_rng._next_seed()

    def get_replication_streams(self) -> tuple[dict[str, np.random.Generator], int]:
        """
        Genera un set di stream indipendenti per una SINGOLA replica della simulazione.

        Questo metodo va chiamato all'inizio di ogni nuova esecuzione/replica.

        Returns:
            Un dizionario dove le chiavi sono i nomi degli stream (es. 'arrivals')
            e i valori sono oggetti numpy.random.Generator pronti all'uso.
        """
        # 1. Genera un nuovo seed dal nostro Lehmer master.
        #    Questo assicura che ogni replica usi un set di stream diverso dalle altre.
        replication_seed = self._master_rng._next_seed()
        # 2. Usa la SeedSequence di NumPy. È il modo moderno e corretto per
        #    creare stream indipendenti. "Sviluppa" il singolo seme in un
        #    set di stati iniziali di alta qualità.
        ss = np.random.SeedSequence(replication_seed)

        # 3. "Figlia" la SeedSequence per creare stati indipendenti per ogni stream.
        #    Il metodo .spawn() è progettato esattamente per questo scopo.
        child_states = ss.spawn(len(self.STREAM_NAMES))

        # 4. Crea un dizionario di generatori, uno per ogni stato figlio.
        streams = {
            name: np.random.default_rng(s)
            for name, s in zip(self.STREAM_NAMES, child_states)
        }

        return streams, replication_seed