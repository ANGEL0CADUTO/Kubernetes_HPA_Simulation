import numpy as np
class Welford:
    """ attributi:
     count (int) : numero di valori accumulati
     mean (array(D,)) media dei valori accumulati
     var_s (array(D,)): campione della varianza dei valori accumulati
     var_p(array(D,)): popolazione della varianza dei valori accumulati
     """

    def __init__(self,elements=None):
        """ inizializzo i dati """
        if elements is None:
            self.__shape=None
            self.__count= 0
            self.__m=None
            self.__s=None
            self.__count_old=None
            self.__m_old=None
            self.__s_old=None
        else:
            self.__shape = elements[0].shape

            self.__count = elements.shape[0]
            self.__m = np.mean(elements, axis=0)
            self.__s = np.var(elements, axis=0, ddof=0) * elements.shape[0]
            # rollback per elementi vecchi
            self.__count_old = None
            self.__init_old_with_nan()

    @property
    def count(self):
        return self.__count

    @property
    def mean(self):
        if self.__m is None:
            return None
        # Se è un array NumPy e contiene esattamente un elemento, restituiscilo come float.
        # Questo gestisce sia array 0-D (shape == ()) sia array 1-D di dimensione 1 (shape == (1,)).
        if isinstance(self.__m, np.ndarray) and self.__m.size == 1:
            return float(self.__m.item()) # .item() estrae il valore scalare dall'array
        return self.__m
    @property
    def var_s(self):
        v = self.__getvar(ddof=1)
        if v is None: # Aggiungi un controllo esplicito per None
            return None
        # Se è un array NumPy e contiene esattamente un elemento, restituiscilo come float.
        if isinstance(v, np.ndarray) and v.size == 1:
            return float(v.item()) # .item() estrae il valore scalare dall'array
        return v

    @property
    def var_p(self):
        v = self.__getvar(ddof=0)
        if v is None: # Aggiungi un controllo esplicito per None
            return None
        # Se è un array NumPy e contiene esattamente un elemento, restituiscilo come float.
        if isinstance(v, np.ndarray) and v.size == 1:
            return float(v.item()) # .item() estrae il valore scalare dall'array
        return v

    def add(self, element, backup_flg=True):
        """
            element (array(D, )): campioni dei dati .
            backup_flg (boolean): se imposto a true va in rollback.

        """
        # Initialize.
        if np.isscalar(element):
            element = np.array([element])
        else:
            element = np.asarray([element])
        if self.__shape is None:
            self.__shape = element.shape
            self.__m = np.zeros(element.shape)
            self.__s = np.zeros(element.shape)
            self.__init_old_with_nan()

        else:
            assert element.shape == self.__shape

        # backup rollbacking
        if backup_flg:
            self.__backup_attrs()

        # Welford algoritmo
        self.__count += 1
        delta = element - self.__m
        self.__m += delta / self.__count
        self.__s += delta * (element - self.__m)

    def add_all(self, elements, backup_flg=True):
        """ aggiungo più campioni.

        Args:
            elements (array(S, D)): campioni .
            backup_flg (boolean): stessa cosa di su.

        """
        # backup  rollbacking
        if backup_flg:
            self.__backup_attrs()

        for elem in elements:
            self.add(elem, backup_flg=False)

    def rollback(self):
        self.__count = self.__count_old
        self.__m[...] = self.__m_old[...]
        self.__s[...] = self.__s_old[...]

    def merge(self, other, backup_flg=True):
        """Merge accumulatore con un altro."""

        if backup_flg:
            self.__backup_attrs()

        count = self.__count + other.__count
        delta = self.__m - other.__m
        delta2 = delta * delta
        m = (self.__count * self.__m + other.__count * other.__m) / count
        s = self.__s + other.__s + delta2 * (self.__count * other.__count) / count

        self.__count = count
        self.__m = m
        self.__s = s

    def __getvar(self, ddof):
        if self.__count <= 0:
            return None
        min_count = ddof
        if self.__count <= min_count:
            return np.full(self.__shape, np.nan)
        else:
            return self.__s / (self.__count - ddof)

    def __backup_attrs(self):
        if self.__shape is None:
            pass
        else:
            self.__count_old = self.__count
            self.__m_old[...] = self.__m[...]
            self.__s_old[...] = self.__s[...]

    def __init_old_with_nan(self):
        self.__m_old = np.empty(self.__shape)
        self.__m_old[...] = np.nan
        self.__s_old = np.empty(self.__shape)
        self.__s_old[...] = np.nan