from dataclasses import dataclass
from src.config import RequestType, Priority


@dataclass
class Request:
    """
    Classe per rappresentare una singola richiesta in arrivo nel sistema.
    L'uso di @dataclass genera automaticamente metodi come __init__ e __repr__.

    Attributi:
        request_id (int): Identificatore univoco della richiesta.
        req_type (RequestType, optional): Tipo di richiesta (es. checkout, ricerca).
                                          Può essere utile per analisi più dettagliate.
        arrival_time (float): Tempo di simulazione in cui la richiesta arriva.
        priority (Priority): Classe di priorità della richiesta (es. ALTA, MEDIA, BASSA).
                             Determina da quale coda verrà servita.
        service_time (float): Tempo necessario a un Pod per processare questa specifica
                              richiesta. Dipende dal tipo/priorità.
    """
    request_id: int
    req_type: RequestType
    arrival_time: float
    priority: Priority
    service_time: float  # con le code di priorità diventa specifico della classe!

