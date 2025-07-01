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
    ensure_plot_dir()
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
    ensure_plot_dir()
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
    ensure_plot_dir()
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
    ensure_plot_dir()
    plt.savefig('plots/response_time_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_response_time_boxplot(metrics: Metrics):
    # Mostra boxplot dei tempi di risposta per ogni tipo di richiesta.

    req_types = sorted(metrics.response_times_data.keys(), key=lambda e: e.name)
    data = [metrics.response_times_data[rt] for rt in req_types if metrics.response_times_data[rt]]

    if not data:
        print("Nessun dato per il boxplot.")
        return

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, patch_artist=True, notch=True, boxprops=dict(facecolor='skyblue', color='black'))
    plt.xticks(ticks=range(1, len(req_types) + 1), labels=[rt.name for rt in req_types], rotation=45)
    plt.title("Boxplot dei Tempi di Risposta per Tipo di Richiesta")
    plt.ylabel("Tempo di Risposta (s)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/response_time_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_request_heatmap(metrics: Metrics, bin_size=10):
    # Heatmap: intensità delle richieste nel tempo divise per tipo.
    # Bin_size = secondi per ogni colonna della heatmap

    import seaborn as sns
    import pandas as pd

    req_types = sorted(metrics.response_times_history.keys(), key=lambda e: e.name)

    all_data = []

    for req_type in req_types:
        for timestamp, _ in metrics.response_times_history[req_type]:
            all_data.append((req_type.name, int(timestamp // bin_size) * bin_size))  # Bucket

    if not all_data:
        print("Nessun dato per la heatmap.")
        return

    df = pd.DataFrame(all_data, columns=["Tipo", "TimeBin"])
    heatmap_data = df.groupby(["Tipo", "TimeBin"]).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap="YlGnBu", linewidths=0.5, linecolor='gray')
    plt.title("Heatmap: Intensità delle Richieste nel Tempo")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Tipo di Richiesta")
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/request_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_response_time_scatter(metrics: Metrics):
    # Scatter plot: ogni punto è una richiesta con il suo tempo di risposta.

    plt.figure(figsize=(12, 6))

    for req_type in sorted(metrics.response_times_history.keys(), key=lambda e: e.name):
        data = metrics.response_times_history[req_type]
        if not data:
            continue
        times, responses = zip(*data)
        plt.scatter(times, responses, s=10, alpha=0.6, label=req_type.name)

    plt.title("Scatter Plot dei Tempi di Risposta")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Tempo di Risposta (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/response_time_scatter.png", dpi=300, bbox_inches='tight')
    plt.show()


# analizzare la velocità di smaltimento delle richieste nel tempo.
def plot_cumulative_requests(metrics: Metrics):
    served = metrics.total_requests_served
    generated = metrics.total_requests_generated

    if served == 0:
        print("Nessuna richiesta servita da plottare.")
        return

    plt.figure(figsize=(12, 6))
    served_timeline = [0]
    generated_timeline = [0]
    time_points = [0]

    current_served = 0
    current_generated = 0

    # Simulazione incrementale temporale (Da usare eventi veri)
    for ts, _ in sorted(metrics.queue_length_history):
        current_served += 1
        current_generated += 1
        time_points.append(ts)
        served_timeline.append(current_served)
        generated_timeline.append(current_generated)

    plt.plot(time_points, generated_timeline, label="Richieste Generate", color='blue')#sul grafico escono sovrapposte
    plt.plot(time_points, served_timeline, label="Richieste Servite", color='green')
    plt.title("Andamento Cumulativo delle Richieste nel Tempo")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Numero Cumulativo")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/cumulative_requests.png", dpi=300, bbox_inches='tight')
    plt.show()


# analizzare l'andamento dei tempi di attesa nel tempo.
def plot_wait_time_trend(metrics: Metrics):
    plt.figure(figsize=(12, 6))
    all_waits = []
    for req_type in sorted(metrics.wait_times_history.keys(), key=lambda e: e.name):
        all_waits.extend(metrics.wait_times_history[req_type])

    if not all_waits:
        print("Nessun dato per i tempi di attesa.")
        return

    all_waits.sort(key=lambda x: x[0])
    times, waits = zip(*all_waits)

    window_size = 50
    if len(waits) >= window_size:
        moving_avg = np.convolve(waits, np.ones(window_size) / window_size, mode='valid')
        moving_avg_times = times[window_size - 1:]
        plt.plot(moving_avg_times, moving_avg, label=f'Media Mobile (finestra={window_size})')

    plt.title("Andamento del Tempo Medio di Attesa nel Tempo")
    plt.xlabel("Tempo di Simulazione (s)")
    plt.ylabel("Tempo di Attesa Medio (s)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/wait_time_trend.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_wait_time_boxplot(metrics: Metrics):
    req_types = sorted(metrics.wait_times_data.keys(), key=lambda e: e.name)
    data = [metrics.wait_times_data[rt] for rt in req_types if metrics.wait_times_data[rt]]

    if not data:
        print("Nessun dato per il boxplot dei tempi di attesa.")
        return

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, patch_artist=True, notch=True, boxprops=dict(facecolor='lightgreen', color='black'))
    plt.xticks(ticks=range(1, len(req_types) + 1), labels=[rt.name for rt in req_types], rotation=45)
    plt.title("Boxplot dei Tempi di Attesa per Tipo di Richiesta")
    plt.ylabel("Tempo di Attesa (s)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    ensure_plot_dir()
    plt.savefig("plots/wait_time_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()


# evito codice duplicato
def ensure_plot_dir():
    makedirs('plots', exist_ok=True)
