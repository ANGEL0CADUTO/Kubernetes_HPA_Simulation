import os
import pandas as pd
from collections import defaultdict
from statistics import mean

def export_summary(metrics, output_dir="output", label='non_prioritized', by_priority=False):
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, f"{label}_metrics.xlsx")
    csv_path = os.path.join(output_dir, f"{label}_metrics.csv")

    # -------- Sheet 1: Aggregated Summary --------
    summary_rows = []

    if by_priority:
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

        for prio, lst in metrics.response_times_by_priority.items():
            summary_rows.append({
                'Metric': f"{prio.name} - Avg Response Time",
                'Value': mean(lst) if lst else None
            })
            summary_rows.append({
                'Metric': f"{prio.name} - Max Response Time",
                'Value': max(lst) if lst else None
            })
        for prio, lst in metrics.wait_times_by_priority.items():
            summary_rows.append({
                'Metric': f"{prio.name} - Avg Wait Time",
                'Value': mean(lst) if lst else None
            })

        for req_type, lst in metrics.response_times_by_req_type.items():
            summary_rows.append({
                'Metric': f"{req_type.name} - Avg Response Time (type)",
                'Value': mean(lst)
            })
        for req_type, lst in metrics.wait_times_by_req_type.items():
            summary_rows.append({
                'Metric': f"{req_type.name} - Avg Wait Time (type)",
                'Value': mean(lst)
            })
    else:
        summary_rows.append({
            'Metric': 'Total Requests Generated',
            'Value': metrics.total_requests_generated
        })
        summary_rows.append({
            'Metric': 'Total Requests Completed',
            'Value': metrics.total_requests_served
        })

        for req_type, lst in metrics.response_times_data.items():
            summary_rows.append({
                'Metric': f"{req_type.name} - Avg Response Time",
                'Value': mean(lst)
            })
        for req_type, lst in metrics.wait_times_data.items():
            summary_rows.append({
                'Metric': f"{req_type.name} - Avg Wait Time",
                'Value': mean(lst)
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
        df_completion = pd.DataFrame(columns=['completion_timestamp', 'response_time'])  # Empty for baseline

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

    # -------- Scrittura Excel multi-foglio --------
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_summary.to_excel(writer, sheet_name='aggregated', index=False)
        df_snapshots.to_excel(writer, sheet_name='system_snapshots', index=False)
        df_completion.to_excel(writer, sheet_name='completion_log', index=False)
        df_resp_raw.to_excel(writer, sheet_name='raw_response', index=False)
        df_wait_raw.to_excel(writer, sheet_name='raw_wait', index=False)

    # Anche CSV di backup (solo il summary principale)
    df_summary.to_csv(csv_path, index=False)
    print(f"\n✅ Dati esportati in:\n- {excel_path} (multi-foglio)\n- {csv_path} (riepilogo)")
