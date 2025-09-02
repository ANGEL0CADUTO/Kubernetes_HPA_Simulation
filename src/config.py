import math
from enum import Enum
from enum import IntEnum

# ==============================================================================
# CONFIGURAZIONI GENERALI DELLA SIMULAZIONE
# ==============================================================================
SIMULATION_TIME = 500     # Durata delle simulazioni a orizzonte finito (s)
LEHMER_SEED = 123456789     # Seed per la riproducibilità

# --- ANALISI A ORIZZONTE INFINITO ---
STEADY_SIMULATION_TIME = 30000  # Durata della simulazione lunga per l'analisi steady-state
WARM_UP_TO_STEADY = 2500        # Periodo di transitorio da scartare, determinato dall'analisi di convergenza
BATCH_K = 64        # numero di batch
BATCH_THRESHOLD = 0.2  # soglia autocorrelazione
CONFIDENCE_LEVEL = 0.95
STEADY_ENABLED = False           # Flag per attivare l'esecuzione della simulazione lunga

# ==============================================================================
# MODELLO DEL SISTEMA (Cluster Kubernetes)
# ==============================================================================
# Giustificazione: Modella un cluster di medie dimensioni con alta disponibilità,
# distribuito su 3 Availability Zones (AZ), una pratica standard su cloud come AWS.
NUM_WORKERS = 3
INITIAL_PODS = 3            # Un Pod per worker all'avvio
MIN_PODS = 2                # Minimo per garantire l'alta disponibilità
MAX_PODS = 24               # Limite di budget/risorse (es. 8 pod max per worker)

# --- CONFIGURAZIONE HPA (Horizontal Pod Autoscaler) ---
# Giustificazione: Parametri standard di Kubernetes (API v2) per un HPA reattivo.
HPA_ENABLED = True
HPA_SYNC_PERIOD = 15        # Intervallo di polling (`--horizontal-pod-autoscaler-sync-period`).
TARGET_QUEUE_LENGTH_PER_POD = 5 # Metrica custom per HPA: scala se ci sono più di 5 richieste in attesa per pod.
MAX_SCALE_STEP = 4          # Kubernetes 1.18+ può aggiungere/rimuovere fino a 4 pod ogni 15s.
SCALE_UP_COOLDOWN = 60      # Cooldown prima di un altro scale-up.
SCALE_DOWN_COOLDOWN = 300   # Cooldown di 5 minuti prima di uno scale-down (standard per evitare oscillazioni).

# ==============================================================================
# MODELLO DEL CARICO DI LAVORO (WORKLOAD E-COMMERCE)
# ==============================================================================

# --- TIPI DI RICHIESTA ---
class RequestType(Enum):
    LOGIN = 1
    NAVIGATION = 2
    CHECKOUT = 3
    ANALYTICS = 4
    ADD_TO_CART = 5

# --- PROFILO DEL TRAFFICO ---
# Giustificazione: Derivato da un modello di funnel di conversione calibrato su dati
# di settore e misurazioni empiriche (Tasso di conversione ≈2.5%, Bounce Rate ≈38%,
# Cart Abandonment Rate ≈70%).
TRAFFIC_PROFILE = {
    RequestType.NAVIGATION:  0.70,
    RequestType.LOGIN:       0.15,
    RequestType.ADD_TO_CART: 0.10,
    RequestType.CHECKOUT:    0.025,
    RequestType.ANALYTICS:   0.025
}

# --- TEMPI DI SERVIZIO ---
# Giustificazione: Calibrati tramite misurazioni empiriche del Time To First Byte (TTFB)
# e del tempo di risposta di chiamate API reali su zalando.it. La distribuzione Lognormale
# è scelta per la sua capacità di modellare la "coda lunga" (long tail) dei tempi di risposta web.
def get_lognormal_params(mean, stdev):
    if mean <= 0: return (0, 0)
    mu_log = math.log(mean**2 / math.sqrt(stdev**2 + mean**2))
    sigma_log = math.sqrt(math.log(stdev**2 / mean**2 + 1))
    return mu_log, sigma_log

SERVICE_TIME_CONFIG = {
    RequestType.LOGIN:       {"dist": "lognormal", "params": get_lognormal_params(mean=0.30, stdev=0.20)},
    RequestType.NAVIGATION:  {"dist": "lognormal", "params": get_lognormal_params(mean=0.30, stdev=0.15)},
    RequestType.ADD_TO_CART: {"dist": "lognormal", "params": get_lognormal_params(mean=0.50, stdev=0.25)},
    RequestType.CHECKOUT:    {"dist": "lognormal", "params": get_lognormal_params(mean=0.85, stdev=0.50)},
    RequestType.ANALYTICS:   {"dist": "exponential", "params": {"scale": 0.05}}
}

# --- TIMEOUT (Pazienza dell'utente) ---
# Giustificazione: Basati su dati di usabilità web (Google) e metriche di performance
# reali misurate su zalando.it (es. First Contentful Paint).
REQUEST_TIMEOUTS = {
    RequestType.LOGIN:       3.0,
    RequestType.NAVIGATION:  3.0,   # Soglia critica di abbandono per il caricamento di una pagina.
    RequestType.ADD_TO_CART: 6.0,
    RequestType.CHECKOUT:    20.0,  # Alta tolleranza dell'utente in fase di pagamento.
    RequestType.ANALYTICS:   15.0  # Timeout tecnico, non legato all'utente.
}

# ==============================================================================
# CONFIGURAZIONE DELLA SOLUZIONE MIGLIORATIVA (DA SOSTITUIRE)
# ==============================================================================
# Questa sezione verrà sostituita dall'implementazione della soluzione avanzata
# (es. Admission Control, WFQ, o HPA Predittivo).
class Priority(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

# Mappatura basata su criticità di business (usata come base per la nuova soluzione)
REQUEST_TYPE_TO_PRIORITY = {
    RequestType.NAVIGATION:  Priority.HIGH,
    RequestType.CHECKOUT:    Priority.HIGH,
    RequestType.LOGIN:       Priority.MEDIUM,
    RequestType.ADD_TO_CART: Priority.MEDIUM,
    RequestType.ANALYTICS:   Priority.LOW
}