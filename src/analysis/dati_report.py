import os
from xlsxwriter import Workbook
import pandas as pd
import numpy as np

from src.utils.metrics import Metrics


def export_summary_to_excel(metrics: Metrics):
    # Sheet 1: Tempo Medio di Risposta
    response_data = []
    for req_type in sorted(metrics.response_times_data.keys(), key=lambda e: e.name):
        if metrics.response_times_data[req_type]:
            avg_resp = np.mean(metrics.response_times_data[req_type])
            response_data.append({
                'TipoRichiesta': req_type.name,
                'TempoMedioRisposta (s)': round(avg_resp, 4)
            })

    df_response = pd.DataFrame(response_data)

    # Sheet 2: Tempo Medio di Attesa
    wait_data = []
    for req_type in sorted(metrics.wait_times_data.keys(), key=lambda e: e.name):
        if metrics.wait_times_data[req_type]:
            avg_wait = np.mean(metrics.wait_times_data[req_type])
            wait_data.append({
                'TipoRichiesta': req_type.name,
                'TempoMedioAttesa (s)': round(avg_wait, 4)
            })

    df_wait = pd.DataFrame(wait_data)

    # Sheet 3: Totali
    df_totals = pd.DataFrame({
        'TotaleRichiesteGenerate': [metrics.total_requests_generated],
        'TotaleRichiesteServite': [metrics.total_requests_served]
    })

    # Scrive su più fogli
    os.makedirs('report', exist_ok=True)  # Assicurati che la cartella esista
    with pd.ExcelWriter('report/output_dati_excel.xlsx',
                        engine='xlsxwriter') as writer:  # Nota il file viene sovrascritto ogni volta
        df_response.to_excel(writer, sheet_name='ResponseTimes',
                             index=False)  # if os.path.exists('output_dati_excel.xlsx'): print("esiste già") return
        df_wait.to_excel(writer, sheet_name='WaitTimes', index=False)
        df_totals.to_excel(writer, sheet_name='Totali', index=False)

    print(f" Riepilogo metriche esportato su '{'output_dati_excel.xlsx'}' con 3 fogli (Risposta, Attesa, Totali).")


def export_summary_to_csv(metrics):
    summary_data = {
        "Tipo Richiesta": [],
        "Tempo Medio di Risposta (s)": [],
        "Tempo Medio di Attesa (s)": []
    }

    for req_type in sorted(metrics.response_times_data.keys(), key=lambda e: e.name):
        response_times = metrics.response_times_data[req_type]
        wait_times = metrics.wait_times_data[req_type]

        if response_times or wait_times:
            summary_data["Tipo Richiesta"].append(req_type.name)
            summary_data["Tempo Medio di Risposta (s)"].append(np.mean(response_times) if response_times else np.nan)
            summary_data["Tempo Medio di Attesa (s)"].append(np.mean(wait_times) if wait_times else np.nan)

    df = pd.DataFrame(summary_data)

    # Aggiunge una riga finale con il totale richieste generate e servite
    total_row = pd.DataFrame([{
        "Tipo Richiesta": "TOTALE",
        "Tempo Medio di Risposta (s)": metrics.total_requests_generated,
        "Tempo Medio di Attesa (s)": metrics.total_requests_served
    }])

    df = pd.concat([df, total_row], ignore_index=True)

    df.to_csv('report/output_dati_csv.csv', index=False)
    print(f"✅ File CSV salvato in: {'report/output_dati_csv.csv'}")


# Funzione per salvare i dati di esecuzione in CSV ed Excel
def save_run_data(metrics: Metrics):
    folder = "report/run_data"
    os.makedirs(folder, exist_ok=True)

    # 1. Dati richieste (tutti i tipi e metriche)
    rows = []
    for req_type in metrics.response_times_history.keys():
        for (timestamp, resp_time), (_, wait_time) in zip(metrics.response_times_history[req_type],
                                                          metrics.wait_times_history[req_type]):
            rows.append({
                "Timestamp": timestamp,
                "RequestType": req_type.name,
                "ResponseTime": resp_time,
                "WaitTime": wait_time
            })
    df_requests = pd.DataFrame(rows)
    df_requests.to_csv(f"{folder}/requests_details.csv", index=False)
    df_requests.to_excel(f"{folder}/requests_details.xlsx", index=False)

    # 2. Pod count nel tempo
    df_pods = pd.DataFrame(metrics.pod_count_history, columns=["Timestamp", "PodCount"])
    df_pods.to_csv(f"{folder}/pod_count_history.csv", index=False)
    df_pods.to_excel(f"{folder}/pod_count_history.xlsx", index=False)

    # 3. Queue length nel tempo
    df_queue = pd.DataFrame(metrics.queue_length_history, columns=["Timestamp", "QueueLength"])
    df_queue.to_csv(f"{folder}/queue_length_history.csv", index=False)
    df_queue.to_excel(f"{folder}/queue_length_history.xlsx", index=False)

    print(f" Dati run salvati in '{folder}' (csv e xlsx)")
