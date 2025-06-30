from os import makedirs

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.utils.metrics import Metrics
import os

matplotlib.use('Qt5Agg')


# Le funzioni plot_pod_history e plot_queue_history rimangono INVARIATE...
def plot_pod_history(metrics: Metrics, config):
    if not metrics.pod_count_history:
        return
    times, counts = zip(*metrics.pod_count_history)
    plt.figure(figsize=(12, 6))
    plt.step(times, counts, where='post', label='Numero di Pod')
    plt.axhline(y=config.MIN_PODS, color='r', linestyle='--', label=f'Min Pods ({config.MIN_PODS})')
    plt.axhline(y=config.MAX_PODS, color='g', linestyle='--', label=f'Max Pods ({config.MAX_PODS})')
    plt.title("Evoluzione del Numero di Pod nel Tempo (HPA)")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Numero di Pod Attivi")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Salva il grafico in una cartella 'plots'
    makedirs('plots', exist_ok=True)
    plt.savefig('plots/pod_count_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_queue_history(metrics: Metrics):
    if not metrics.queue_length_history:
        return
    times, lengths = zip(*metrics.queue_length_history)
    plt.figure(figsize=(12, 6))
    plt.plot(times, lengths)
    plt.title("Evoluzione della Lunghezza della Coda FIFO nel Tempo")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Numero di Richieste in Coda")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    makedirs('plots', exist_ok=True)
    plt.savefig('plots/queue_length_history.png', dpi=300, bbox_inches='tight')
    plt.show()


# ----- NUOVA FUNZIONE DI PLOT -----
def plot_response_time_trend(metrics: Metrics):
    """
    Genera un grafico che mostra l'andamento del tempo di risposta
    nel tempo usando una media mobile.
    """
    plt.figure(figsize=(12, 6))

    # Calcola una media mobile per il tempo di risposta di TUTTE le richieste
    all_responses = []
    for req_type in sorted(metrics.response_times_history.keys(), key=lambda e: e.name):
        all_responses.extend(metrics.response_times_history[req_type])

    # Ordina per timestamp
    all_responses.sort(key=lambda x: x[0])

    if not all_responses:
        print("Nessun dato sui tempi di risposta da plottare.")
        return

    times, responses = zip(*all_responses)

    # Calcola la media mobile
    # La 'window_size' determina quanto "smussato" sarà il grafico.
    # Una finestra più grande dà una linea più liscia ma reagisce più lentamente ai cambiamenti.
    window_size = 50
    if len(responses) >= window_size:
        # Usa np.convolve per un calcolo efficiente della media mobile
        moving_avg = np.convolve(responses, np.ones(window_size) / window_size, mode='valid')
        # Gli istanti di tempo per la media mobile corrispondono alla fine di ogni finestra
        moving_avg_times = times[window_size - 1:]
        plt.plot(moving_avg_times, moving_avg, label=f'Media Mobile (finestra={window_size})')

    plt.title("Andamento del Tempo di Risposta Medio nel Tempo")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Tempo di Risposta Medio (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    makedirs('plots', exist_ok=True)
    plt.savefig('plots/ response_time_trend.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_response_time_histogram(metrics: Metrics):
    """
    Genera ISTOGRAMMI dei tempi di risposta per ogni tipo di richiesta.
    """
    req_types = sorted(list(metrics.response_times_data.keys()), key=lambda e: e.name)
    if not req_types:
        print("Nessun dato per l'istogramma dei tempi di risposta.")
        return

    num_types = len(req_types)
    plt.figure(figsize=(15, 8))

    cols = (num_types + 1) // 2
    rows = 2

    for i, req_type in enumerate(req_types):
        if metrics.response_times_data[req_type]:
            ax = plt.subplot(rows, cols, i + 1)
            # Usa i dati da response_times_data per l'istogramma
            ax.hist(metrics.response_times_data[req_type], bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f"Istogramma Tempi Risposta: {req_type.name}")
            ax.set_xlabel("Tempo (s)")
            if i % cols == 0:
                ax.set_ylabel("Frequenza")
            ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Distribuzione (Istogrammi) dei Tempi di Risposta", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    makedirs('plots', exist_ok=True)
    plt.savefig('plots/response_time_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
