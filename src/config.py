import math
from enum import Enum
from enum import IntEnum



# --- CONFIGURAZIONI GENERALI DELLA SIMULAZIONE ---
SIMULATION_TIME = 1000      # Tempo totale di simulazione (in secondi)
LEHMER_SEED = 123456789     # Seed iniziale per il nostro generatore Lehmer

# --- CONFIGURAZIONE DEL WORKER E DEI POD ---
# Concettualmente abbiamo un solo worker node.
# L'HPA scalerà i pod su questo nodo fino a un massimo di 8.
NUM_WORKERS = 1
INITIAL_PODS = 2            # Partiamo con 2 Pod attivi
MAX_PODS = 8                # Massimo numero di Pod consentito



# --- CONFIGURAZIONE HPA (Horizontal Pod Autoscaler) ---
HPA_ENABLED = True
HPA_SYNC_PERIOD = 7        # HPA controlla le metriche ogni 15 secondi
CPU_TARGET = 0.60           # Utilizzo CPU target (60%) - un po' più basso per reagire prima
MIN_PODS = 1                # Numero minimo di Pod
# MAX_PODS è già definito sopra e verrà usato anche qui


MAX_SCALE_STEP = 2
SCALE_UP_COOLDOWN = 30      # 1 minuto prima di poter fare un altro scale-up
SCALE_DOWN_COOLDOWN = 150  # 5 minuti prima di poter fare un altro scale-down
# Non modelliamo più il POD_STARTUP_TIME perché la capacità della risorsa è istantanea


# --- DEFINIZIONE TIPI DI RICHIESTA ---
class RequestType(Enum):
    LOGIN = 1
    NAVIGATION = 2
    CHECKOUT = 3
    ANALYTICS = 4
    ADD_TO_CART = 5


# --- PROFILO DEL CARICO DI LAVORO (WORKLOAD) ---
TOTAL_ARRIVAL_RATE = 70
TARGET_QUEUE_LENGTH_PER_POD = 2  # NUOVA METRICA: scala se ci sono più di 2 richieste in attesa per pod

TRAFFIC_PROFILE = {
    RequestType.LOGIN: 0.15,
    RequestType.NAVIGATION: 0.40,
    RequestType.CHECKOUT: 0.05,
    RequestType.ANALYTICS: 0.25,
    RequestType.ADD_TO_CART: 0.15
}


# --- FUNZIONE HELPER PER CALCOLARE I PARAMETRI LOG-NORMALE ---
def get_lognormal_params(mean, stdev):
    if mean <= 0: return (0, 0)
    mu_log = math.log(mean**2 / math.sqrt(stdev**2 + mean**2))
    sigma_log = math.sqrt(math.log(stdev**2 / mean**2 + 1))
    return mu_log, sigma_log


# --- CONFIGURAZIONE TEMPI DI SERVIZIO ---
SERVICE_TIME_CONFIG = {
    RequestType.LOGIN: {
        "dist": "lognormal",
        "params": get_lognormal_params(mean=0.05, stdev=0.02)
    },
    RequestType.NAVIGATION: {
        "dist": "lognormal",
        "params": get_lognormal_params(mean=0.1, stdev=0.08)
    },
    RequestType.CHECKOUT: {
        "dist": "lognormal",
        "params": get_lognormal_params(mean=0.5, stdev=0.4)
    },
    RequestType.ANALYTICS: {
        "dist": "exponential",
        "params": {"scale": 0.02}
    },
    RequestType.ADD_TO_CART: {
        "dist": "lognormal",
        "params": get_lognormal_params(mean=0.08, stdev=0.04)
    }
}

# --- TIMEOUT IN SECONDI ---
REQUEST_TIMEOUTS = {
    RequestType.LOGIN: 1.0,
    RequestType.NAVIGATION: 1.0,
    RequestType.ADD_TO_CART: 3.0,
    RequestType.CHECKOUT: 5.0,  # L'utente è più paziente durante il checkout
    RequestType.ANALYTICS: 5.0   # Richiesta interna, può essere scartata
}

# --- SOLUZIONE MIGLIORATIVA: ABSTRACT PRIORITY SCHEDULING ---
PRIORITY_SCHEDULING_ENABLED = True  # O False, per eseguire la versione baseline


class Priority(IntEnum):
    """
    IntEnum per le classi di priorità.
    IntEnum è migliore di Enum in questo caso perché permette di ordinare i valori in maniera semplice
     (ogni classe si comporta come il numero che la identifica)
    Un valore più basso indica una priorità più alta.
    """
    HIGH = 0
    MEDIUM = 1
    LOW = 2


# --- Mappatura fissa da Tipo di Richiesta a Priorità ---
REQUEST_TYPE_TO_PRIORITY = {
    RequestType.LOGIN:       Priority.HIGH,     # Utente va servito subito
    RequestType.NAVIGATION:  Priority.HIGH,     # Delay qui uccide il funnel
    RequestType.ADD_TO_CART: Priority.MEDIUM,   # Intermedia
    RequestType.CHECKOUT:    Priority.LOW,      # Può aspettare qualche secondo
    RequestType.ANALYTICS:   Priority.LOW       # Background
}
