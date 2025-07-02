# data_report.py

import os
import pandas as pd


def export_summary(metrics, output_dir: str, label: str, by_priority: bool = False):
    """
    Esporta il riepilogo delle metriche in formato CSV ed Excel.
    :param metrics: oggetto Metrics o MetricsWithPriority
    :param output_dir: cartella dove salvare i file
    :param label: prefisso file (es: 'prioritized', 'non_prioritized')
    :param by_priority: True se basato su priorità, False se per tipo richiesta
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_data = []

    if by_priority:
        total_generated = len(metrics.request_generation_timestamps)
        total_completed = sum(metrics.requests_completed_by_priority.values())
        total_timeouts = sum(metrics.requests_timed_out_by_priority.values())

        for prio in sorted(metrics.requests_completed_by_priority.keys(), key=lambda p: p.name):
            completed = metrics.requests_completed_by_priority.get(prio, 0)
            timeouts = metrics.requests_timed_out_by_priority.get(prio, 0)
            if completed > 0:
                avg_resp = sum(metrics.response_times_by_priority[prio]) / completed
                avg_wait = sum(metrics.wait_times_by_priority[prio]) / completed
                max_resp = max(metrics.response_times_by_priority[prio])
            else:
                avg_resp = avg_wait = max_resp = 0

            summary_data.append({
                "category": prio.name,
                "completed": completed,
                "timed_out": timeouts,
                "avg_response_time": avg_resp,
                "avg_wait_time": avg_wait,
                "max_response_time": max_resp
            })

        summary_data.append({
            "category": "TOTAL",
            "completed": total_completed,
            "timed_out": total_timeouts,
            "avg_response_time": "",
            "avg_wait_time": "",
            "max_response_time": ""
        })

    else:
        for req_type in sorted(metrics.response_times_data.keys(), key=lambda t: t.name):
            rt_list = metrics.response_times_data[req_type]
            wt_list = metrics.wait_times_data[req_type]
            if rt_list:
                summary_data.append({
                    "category": req_type.name,
                    "completed": len(rt_list),
                    "avg_response_time": sum(rt_list) / len(rt_list),
                    "avg_wait_time": sum(wt_list) / len(wt_list),
                    "max_response_time": max(rt_list),
                })

        summary_data.append({
            "category": "TOTAL",
            "completed": metrics.total_requests_served,
            "avg_response_time": "",
            "avg_wait_time": "",
            "max_response_time": ""
        })

    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, f"{label}_summary.csv"), index=False)
    df.to_excel(os.path.join(output_dir, f"{label}_summary.xlsx"), index=False)
