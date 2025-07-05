import os
import pandas as pd
from statistics import mean

def export_summary(metrics, output_dir="output", label='non_prioritized', by_priority=False):
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f"{label}_metrics.xlsx")
    csv_path = os.path.join(output_dir, f"{label}_metrics.csv")

    # -------- Sheet 1: Aggregated Summary --------
    summary_rows = []

    if by_priority:
        # Gestione MetricsWithPriority
        total_gen = len(metrics.request_generation_timestamps)
        total_completed = sum(metrics.requests_completed_by_priority.values())
        total_timed_out = sum(metrics.requests_timed_out_by_priority.values())

        summary_rows.append({
            'Metric': 'Total Requests Generated',
            'Value': total_gen
        })
        summary_rows.append({
            'Metric': 'Total Requests Completed',
            'Value': total_completed
        })
        summary_rows.append({
            'Metric': 'Total Requests Timed Out',
            'Value': total_timed_out
        })
        summary_rows.append({
            'Metric': 'Remaining in Queue',
            'Value': total_gen - total_completed - total_timed_out
        })

        # Metriche per priorità
        for prio, lst in metrics.response_times_by_priority.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{prio.name} - Avg Response Time",
                    'Value': mean(lst)
                })
                summary_rows.append({
                    'Metric': f"{prio.name} - Max Response Time",
                    'Value': max(lst)
                })

        for prio, lst in metrics.wait_times_by_priority.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{prio.name} - Avg Wait Time",
                    'Value': mean(lst)
                })

        # Metriche per tipo di richiesta
        for req_type, lst in metrics.response_times_by_req_type.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{req_type.name} - Avg Response Time (type)",
                    'Value': mean(lst)
                })

        for req_type, lst in metrics.wait_times_by_req_type.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{req_type.name} - Avg Wait Time (type)",
                    'Value': mean(lst)
                })

        # P_loss per priorità
        for prio in metrics.requests_generated_by_priority:
            generated = metrics.requests_generated_by_priority[prio]
            timed_out = metrics.requests_timed_out_by_priority[prio]
            if generated > 0:
                p_loss = timed_out / generated
                summary_rows.append({
                    'Metric': f"{prio.name} - P_loss",
                    'Value': p_loss
                })

        # P_loss per tipo di richiesta
        for req_type in metrics.requests_timed_out_by_req_type:
            # Dobbiamo calcolare le richieste generate per tipo
            # Assumiamo che possiamo dedurlo dai dati disponibili
            completed = len(metrics.response_times_by_req_type[req_type])
            timed_out = metrics.requests_timed_out_by_req_type[req_type]
            total_for_type = completed + timed_out
            if total_for_type > 0:
                p_loss = timed_out / total_for_type
                summary_rows.append({
                    'Metric': f"{req_type.name} - P_loss (type)",
                    'Value': p_loss
                })

    else:
        # Gestione Metrics (baseline)
        summary_rows.append({
            'Metric': 'Total Requests Generated',
            'Value': metrics.total_requests_generated
        })
        summary_rows.append({
            'Metric': 'Total Requests Served',
            'Value': metrics.total_requests_served
        })
        summary_rows.append({
            'Metric': 'Total Requests Timed Out',
            'Value': sum(metrics.requests_timed_out_data.values())
        })

        # Metriche per tipo di richiesta
        for req_type, lst in metrics.response_times_data.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{req_type.name} - Avg Response Time",
                    'Value': mean(lst)
                })
                summary_rows.append({
                    'Metric': f"{req_type.name} - Max Response Time",
                    'Value': max(lst)
                })

        for req_type, lst in metrics.wait_times_data.items():
            if lst:  # Solo se ci sono dati
                summary_rows.append({
                    'Metric': f"{req_type.name} - Avg Wait Time",
                    'Value': mean(lst)
                })

        # P_loss per tipo di richiesta (baseline)
        for req_type in metrics.requests_generated_data:
            generated = metrics.requests_generated_data[req_type]
            timed_out = metrics.requests_timed_out_data[req_type]
            if generated > 0:
                p_loss = timed_out / generated
                summary_rows.append({
                    'Metric': f"{req_type.name} - P_loss",
                    'Value': p_loss
                })

    df_summary = pd.DataFrame(summary_rows)

    # -------- Sheet 2: System Snapshots --------
    if by_priority:
        snapshots = []
        for i, ts in enumerate(metrics.timestamps):
            row = {
                'timestamp': ts,
                'pod_count': metrics.pod_counts[i],
                'queue_length': metrics.queue_lengths[i]
            }
            # Aggiungi lunghezze delle code per priorità
            for prio, lst in metrics.queue_lengths_per_priority.items():
                if i < len(lst):
                    row[f"queue_length_{prio.name}"] = lst[i]
            snapshots.append(row)
        df_snapshots = pd.DataFrame(snapshots)
    else:
        # Baseline metrics
        df_snapshots = pd.DataFrame([
            {
                'timestamp': ts,
                'pod_count': pod,
                'queue_length': qlen
            }
            for (ts, pod), (_, qlen) in zip(metrics.pod_count_history, metrics.queue_length_history)
        ])

    # -------- Sheet 3: Completion Log --------
    if by_priority:
        completion_data = []
        for prio in metrics.completion_timestamps_by_priority:
            for ts, resp in zip(metrics.completion_timestamps_by_priority[prio],
                                metrics.response_times_at_completion_by_priority[prio]):
                completion_data.append({
                    'priority': prio.name,
                    'completion_timestamp': ts,
                    'response_time': resp
                })
        df_completion = pd.DataFrame(completion_data)
    else:
        # Per baseline, creiamo un log basato sui dati storici
        completion_data = []
        for req_type in metrics.response_times_history:
            for ts, resp_time in metrics.response_times_history[req_type]:
                completion_data.append({
                    'request_type': req_type.name,
                    'completion_timestamp': ts,
                    'response_time': resp_time
                })
        df_completion = pd.DataFrame(completion_data)

    # -------- Sheet 4: Raw Response Times --------
    response_raw = []
    if by_priority:
        for prio, lst in metrics.response_times_by_priority.items():
            for val in lst:
                response_raw.append({'group': f"priority_{prio.name}", 'response_time': val})
        for req_type, lst in metrics.response_times_by_req_type.items():
            for val in lst:
                response_raw.append({'group': f"type_{req_type.name}", 'response_time': val})
    else:
        for req_type, lst in metrics.response_times_data.items():
            for val in lst:
                response_raw.append({'group': f"type_{req_type.name}", 'response_time': val})
    df_resp_raw = pd.DataFrame(response_raw)

    # -------- Sheet 5: Raw Wait Times --------
    wait_raw = []
    if by_priority:
        for prio, lst in metrics.wait_times_by_priority.items():
            for val in lst:
                wait_raw.append({'group': f"priority_{prio.name}", 'wait_time': val})
        for req_type, lst in metrics.wait_times_by_req_type.items():
            for val in lst:
                wait_raw.append({'group': f"type_{req_type.name}", 'wait_time': val})
    else:
        for req_type, lst in metrics.wait_times_data.items():
            for val in lst:
                wait_raw.append({'group': f"type_{req_type.name}", 'wait_time': val})
    df_wait_raw = pd.DataFrame(wait_raw)

    # -------- Sheet 6: Timeout Analysis --------
    timeout_data = []
    if by_priority:
        # Timeout per priorità
        for prio in metrics.requests_generated_by_priority:
            generated = metrics.requests_generated_by_priority[prio]
            timed_out = metrics.requests_timed_out_by_priority[prio]
            completed = metrics.requests_completed_by_priority[prio]
            timeout_data.append({
                'category': 'priority',
                'group': prio.name,
                'generated': generated,
                'completed': completed,
                'timed_out': timed_out,
                'p_loss': timed_out / generated if generated > 0 else 0
            })

        # Timeout per tipo di richiesta
        for req_type in metrics.requests_timed_out_by_req_type:
            completed = len(metrics.response_times_by_req_type[req_type])
            timed_out = metrics.requests_timed_out_by_req_type[req_type]
            total_for_type = completed + timed_out
            timeout_data.append({
                'category': 'request_type',
                'group': req_type.name,
                'generated': total_for_type,
                'completed': completed,
                'timed_out': timed_out,
                'p_loss': timed_out / total_for_type if total_for_type > 0 else 0
            })
    else:
        # Baseline timeout analysis
        for req_type in metrics.requests_generated_data:
            generated = metrics.requests_generated_data[req_type]
            timed_out = metrics.requests_timed_out_data[req_type]
            completed = len(metrics.response_times_data[req_type])
            timeout_data.append({
                'category': 'request_type',
                'group': req_type.name,
                'generated': generated,
                'completed': completed,
                'timed_out': timed_out,
                'p_loss': timed_out / generated if generated > 0 else 0
            })

    df_timeout = pd.DataFrame(timeout_data)

    # -------- Scrittura Excel multi-foglio --------
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='aggregated', index=False)
        df_snapshots.to_excel(writer, sheet_name='system_snapshots', index=False)
        df_completion.to_excel(writer, sheet_name='completion_log', index=False)
        df_resp_raw.to_excel(writer, sheet_name='raw_response', index=False)
        df_wait_raw.to_excel(writer, sheet_name='raw_wait', index=False)
        df_timeout.to_excel(writer, sheet_name='timeout_analysis', index=False)

    # CSV di backup (solo il summary principale)
    df_summary.to_csv(csv_path, index=False)
    print(f"\nDati esportati in:\n- {excel_path} (multi-foglio)\n- {csv_path} (riepilogo)")