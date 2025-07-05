import math

class LehmerRNG:
    """
    Implementazione di un generatore di numeri casuali Lehmer (Park-Miller).
    Viene usato per generare un seed per il generatore di NumPy per garantire
    la riproducibilità partendo da un algoritmo classico.
    """
    def __init__(self, seed):
        self.seed = seed
        # Parametri comuni per il Lehmer RNG (MINSTD)
        self.m = 2**31 - 1
        self.a = 48271

    def _next_seed(self):
        """Genera il prossimo seed nella sequenza."""
        self.seed = (self.a * self.seed) % self.m
        return self.seed

    def get_numpy_seeds(self, count=1):
        """
        Restituisce una lista di 'count' seed validi per NumPy,
        garantendo che siano diversi tra loro.
        """
        for _ in range(100):  # "Riscalda" il generatore
            self._next_seed()

        seeds = []
        for _ in range(count):
            seeds.append(self._next_seed())
        return seeds