import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

plt.rcParams['figure.facecolor'] = 'white'        # Rende bianco lo sfondo dell'intera figura.
plt.rcParams['savefig.facecolor'] = 'white'       # Assicura che il colore di sfondo della figura salvata sia bianco.
plt.rcParams['savefig.transparent'] = False
def save_plot( output_dir, filename, fig):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafico salvato in: {save_path}")


def plot_transient_comparison( metrics_base, metrics_prio,metrics_wfq, scenario_name, replica_idx, output_dir):
    """
    Confronta le metriche nel transiente (orizzonte finito) tra Baseline e Priorità,
    per una singola replica.
    """

    # --- Estrazione serie temporali dai metrics ---
    base_resp = metrics_base.get_all_response_times_with_timestamps()
    prio_resp = metrics_prio.get_all_response_times_with_timestamps()
    wfq_resp = metrics_wfq.get_all_response_times_with_timestamps()

    if not base_resp or not prio_resp:
        print(f"[Replica {replica_idx}] Nessun dato disponibile per {scenario_name}.")
        return

    df_base = pd.DataFrame(base_resp, columns=["time", "response_time"])
    df_prio = pd.DataFrame(prio_resp, columns=["time", "response_time"])
    df_wfq = pd.DataFrame(wfq_resp, columns=["time", "response_time"])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df_base["time"], df_base["response_time"],
            label="Baseline (FIFO)", alpha=0.7, color="red")
    ax.plot(df_prio["time"], df_prio["response_time"],
            label="Con Priorità", alpha=0.7, color="blue")
    ax.plot(df_wfq["time"], df_wfq["response_time"],
            label="WFQ", alpha=0.7, color="green")

    ax.set_title(f"Confronto Transiente - {scenario_name} (Replica {replica_idx+1})", fontsize=14)
    ax.set_xlabel("Tempo di Simulazione")
    ax.set_ylabel("Tempo di Risposta")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    save_plot(output_dir, f"transient_comparison_{scenario_name}_rep{replica_idx+1}.png", fig)
