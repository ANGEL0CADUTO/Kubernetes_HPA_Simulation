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

    """
    request_id: int
    req_type: RequestType
    arrival_time: float



#--- CLASSE DERIVATA (PER IL MIGLIORAMENTO) ---
# Usiamo l'ereditarietà. PriorityRequest ha tutti i campi di Request più i suoi campi specifici.
@dataclass
class PriorityRequest(Request):
    """
    Rappresenta una richiesta specializzata con priorità e tempo di servizio.
    Eredita da Request e aggiunge:
        priority (Priority): Classe di priorità della richiesta (es. ALTA, MEDIA, BASSA).
                             Determina da quale coda verrà servita.
        service_time (float): Tempo necessario a un Pod per processare questa specifica
                              richiesta. Dipende dal tipo/priorità.
    """
    priority: Priority
    service_time: float