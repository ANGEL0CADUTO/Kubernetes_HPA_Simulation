from dataclasses import dataclass
from src.config import RequestType


@dataclass
class Request:
    """
    Classe per rappresentare una singola richiesta in arrivo nel sistema.
    L'uso di @dataclass genera automaticamente metodi come __init__ e __repr__.
    """
    request_id: int
    req_type: RequestType
    arrival_time: float
