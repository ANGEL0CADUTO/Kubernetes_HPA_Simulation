from statistics import mean

from src import config
from src.config import CONFIDENCE_LEVEL, RequestType, Priority
from src.simulation.simulator_blackfriday import SimulatorBlackFridayBaseline, SimulatorBlackFriday as SimulatorBlackFridayDWFQ
from src.utils.metrics import Metrics
from src.utils.metrics_with_priority import MetricsWithPriority
from analysis.plotter_blackfriday import PlotterBlackFriday
from src.utils.lehmer_rng import LehmerRNG as RNGManager
import scipy.stats as st

def mean_and_ci(data, confidence=0.95):
    n = len(data)
    if n == 0:
        return None, None
    m = mean(data)
    if n == 1:
        return m, 0
    sem = st.sem(data)
    h = sem * st.t.ppf((1 + confidence) / 2., n - 1)
    return m, h

def main_blackfriday_analysis():
    print("="*80)
    print("ANALISI SCENARIO 'BLACK FRIDAY' ")
    print("="*80)

    NUM_REPLICATIONS = 50
    config.SIMULATION_TIME = 3000

    t_notte_fine = config.SIMULATION_TIME * 0.15
    t_mattina_fine = config.SIMULATION_TIME * 0.30
    t_pranzo_fine = config.SIMULATION_TIME * 0.45
    t_picco_fine = config.SIMULATION_TIME * 0.55
    t_pomeriggio_fine = config.SIMULATION_TIME * 0.70
    t_sera_fine = config.SIMULATION_TIME * 0.85

    CARICO_1 = 170
    CARICO_2 = 170
    CARICO_3 = 170
    CARICO_4 = 170

    def lambda_black_friday(t: float) -> float:
        if t < t_notte_fine:
            return CARICO_1          # 20
        if t < t_mattina_fine:
            return CARICO_2          # 40
        if t < t_pranzo_fine:
            return CARICO_3          # 85
        if t < t_picco_fine:
            return CARICO_4          # 170
        if t < t_pomeriggio_fine:
            return CARICO_3          # 85
        if t < t_sera_fine:
            return CARICO_2          # 40
        return CARICO_1              # 20

    print(f"\nConfigurazione Esperimento 'Black Friday':")
    print(f" - Durata Simulazione: {config.SIMULATION_TIME}s, Repliche: {NUM_REPLICATIONS}")

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)

    all_results = {}

    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        replication_streams, rep_seed = rng_manager.get_replication_streams()
        print(f"--- Replica {i+1} utilizzerà il SEED master: {rep_seed} per entrambi i sistemi ---")

        # 1. Baseline (FIFO)
        simulator_base = SimulatorBlackFridayBaseline(config, Metrics, replication_streams['arrivals'], replication_streams['choice'], replication_streams['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 1: Baseline (FIFO) ---")
        simulator_base.run(config.SIMULATION_TIME)

        # 2. Dynamic Weighted Fair Queuing (DWFQ)
        simulator_wfq = SimulatorBlackFridayDWFQ(config, MetricsWithPriority, replication_streams['arrivals'], replication_streams['choice'], replication_streams['service'], lambda_black_friday)
        print("\n--- Esecuzione Modello 2: Dynamic Weighted Fair Queuing (DWFQ) ---")

        simulator_wfq.run(config.SIMULATION_TIME)


        all_results[i] = {
            'baseline': simulator_base.metrics_agg,
            'wfq': simulator_wfq.metrics_agg,
            'seed': rep_seed
        }

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE {'='*30}")

    baseline_results_by_type = {req_type: [] for req_type in RequestType}
    all_baseline_resp_times_total = []
    all_baseline_wait_times_total = []

    wfq_results_by_priority = {prio: [] for prio in Priority}
    all_wfq_resp_times_total = []
    all_wfq_wait_times_total = []

    # --- Aggregazione dati da tutte le repliche ---
    for rep_id, rep_data in all_results.items():
        # Baseline
        baseline_metrics = rep_data['baseline']
        if baseline_metrics:
            # Response times
            for req_type, resp_times in baseline_metrics.response_times_data.items():
                baseline_results_by_type[req_type].extend(resp_times)
                all_baseline_resp_times_total.extend(resp_times)
            # Wait times
            for req_type, wait_times in baseline_metrics.wait_times_data.items():
                all_baseline_wait_times_total.extend(wait_times)

        # WFQ
        wfq_metrics = rep_data['wfq']
        if wfq_metrics:
            # Response times by priority
            for prio, resp_times in wfq_metrics.response_times_by_priority.items():
                wfq_results_by_priority[prio].extend(resp_times)
                all_wfq_resp_times_total.extend(resp_times)
            # Wait times by priority
            for prio, wait_times in wfq_metrics.wait_times_by_priority.items():
                all_wfq_wait_times_total.extend(wait_times)

    # --- Stampa finale Baseline ---
    print(f"\n{'='*30} RISULTATI FINALI FIFO (BASELINE) {'='*30}")
    print(f"Intervallo di Confidenza al {CONFIDENCE_LEVEL * 100:.0f}%")
    print(f"{'Tipo Richiesta':<20} | {'Media Resp (s)':>15} | {'Semi-Ampiezza (±s)':>20} | {'IC (s)'}")
    print("-"*100)

    # Totale Response Time
    m_total, h_total = mean_and_ci(all_baseline_resp_times_total, confidence=CONFIDENCE_LEVEL)
    if m_total is not None:
        print(f"{'TOTALE':<20} | {m_total:15.6f} | {h_total:20.6f} | [{m_total-h_total:.6f}, {m_total+h_total:.6f}]")
    else:
        print(f"{'TOTALE':<20} | {'N/A':>15} | {'N/A':>20} | {'[N/A, N/A]':>27}")

    # Response time per RequestType
    for req_type in RequestType:
        data = baseline_results_by_type[req_type]
        m, h = mean_and_ci(data, confidence=CONFIDENCE_LEVEL)
        if m is not None:
            print(f"{req_type.name:<20} | {m:15.6f} | {h:20.6f} | [{m-h:.6f}, {m+h:.6f}]")
        else:
            print(f"{req_type.name:<20} | {'N/A':>15} | {'N/A':>20} | {'[N/A, N/A]':>27}")

    # Totale Wait Time
    m_total_wait, h_total_wait = mean_and_ci(all_baseline_wait_times_total, confidence=CONFIDENCE_LEVEL)
    if m_total_wait is not None:
        print(f"{'TOTALE Wait':<20} | {m_total_wait:15.6f} | {h_total_wait:20.6f} | [{m_total_wait-h_total_wait:.6f}, {m_total_wait+h_total_wait:.6f}]")

    # --- Stampa finale WFQ ---
    print(f"\n{'='*30} RISULTATI FINALI DWFQ {'='*30}")
    print(f"Intervallo di Confidenza al {CONFIDENCE_LEVEL * 100:.0f}%")
    print(f"{'Priorità':<20} | {'Media Resp (s)':>15} | {'Semi-Ampiezza (±s)':>20} | {'IC (s)'}")
    print("-"*100)

    # Totale Response Time WFQ
    m_total_wfq, h_total_wfq = mean_and_ci(all_wfq_resp_times_total, confidence=CONFIDENCE_LEVEL)
    if m_total_wfq is not None:
        print(f"{'TOTALE':<20} | {m_total_wfq:15.6f} | {h_total_wfq:20.6f} | [{m_total_wfq-h_total_wfq:.6f}, {m_total_wfq+h_total_wfq:.6f}]")

    # Response time per Priority
    for prio in Priority:
        data = wfq_results_by_priority[prio]
        m, h = mean_and_ci(data, confidence=CONFIDENCE_LEVEL)
        if m is not None:
            print(f"{prio.name:<20} | {m:15.6f} | {h:20.6f} | [{m-h:.6f}, {m+h:.6f}]")
        else:
            print(f"{prio.name:<20} | {'N/A':>15} | {'N/A':>20} | {'[N/A, N/A]':>27}")

    m_total_wait_wfq, h_total_wait_wfq = mean_and_ci(all_wfq_wait_times_total, confidence=CONFIDENCE_LEVEL)
    if m_total_wait_wfq is not None:
        print(f"{'TOTALE Wait':<20} | {m_total_wait_wfq:15.6f} | {h_total_wait_wfq:20.6f} | [{m_total_wait_wfq-h_total_wait_wfq:.6f}, {m_total_wait_wfq+h_total_wait_wfq:.6f}]")



    if all_results:
        # Inizializziamo il plotter con i dati della prima replica, solo per soddisfare il costruttore
        plotter = PlotterBlackFriday(
            metrics_base_agg=all_results[0]['baseline'],
            metrics_prio_agg=None,
            metrics_wfq_agg=all_results[0]['wfq'],
            metrics_per_worker_base=None, # Non ci servono per il nuovo grafico
            metrics_per_worker_prio=None,
            metrics_per_worker_wfq=None,
            config_module=config
        )

        # ANALISI DEL TRANSITORIO #
        print("\n--- Generazione grafici di confronto repliche per scenario Black Friday ---")
        plotter.plot_blackfriday_replication_traces(
            all_results,
            lambda_func=lambda_black_friday,
            output_dir="output/black_friday_analysis/aggregated_traces"
        )
        plotter.plot_confidence_interval_trace(
            all_results,
            lambda_black_friday,
            output_dir="output/black_friday_analysis/aggregated_traces/IC"
        )


if __name__ == "__main__":
    main_blackfriday_analysis()