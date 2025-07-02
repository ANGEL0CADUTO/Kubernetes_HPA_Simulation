import numpy as np
from src import config
from src.analysis.dati_report import export_summary_to_excel, export_summary_to_csv, save_run_data
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
# ----- MODIFICA QUI -----
from src.analysis.plotter import plot_pod_history, plot_queue_history, plot_response_time_trend, \
    plot_response_time_histogram, plot_request_heatmap, plot_response_time_scatter, plot_response_time_boxplot, \
    plot_cumulative_requests, plot_wait_time_trend, plot_wait_time_boxplot
# -------------------------
from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority


def main():
    """
    Funzione principale che orchestra l'intero processo.
    """
    print("--- Inizio Progetto di Simulazione E-commerce ---")

    lehmer_rng = LehmerRNG(seed=config.LEHMER_SEED)
    numpy_seed = lehmer_rng.get_numpy_seed()
    rng = np.random.default_rng(seed=numpy_seed)

    metrics = Metrics()

    simulator = Simulator(config_module=config, metrics=metrics, rng=rng)
    simulator.run()

    save_run_data(metrics)  # Salva i dati di esecuzione in CSV ed Excel

    #metrics.print_summary()
    #export_summary_to_excel(metrics)
    #export_summary_to_csv(metrics)


    # --- MODIFICA QUI ---
    # Genera i grafici di andamento temporale
    #generate_all_plots(metrics, config)
    # --------------------

    print("\n--- Esecuzione baseline Terminata ---")

    # --- ESECUZIONE DELLA SIMULAZIONE MIGLIORATA (PRIORITÀ PER WORKER) ---
    print("\n--- SCENARIO MIGLIORATO (ABSTRACT PRIORITY) ---")

    rng_prio = np.random.default_rng(seed=numpy_seed)   #stesso seed di baseline
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(config, metrics_prio, rng_prio)
    simulator_prio.run()

    metrics_prio.print_summary()
    print(".............................................")
    metrics.print_summary()
    #export_summary_to_excel(metrics_prio)
    #export_summary_to_csv(metrics_prio)

    #generate_all_plots(metrics_prio, config)

    print("\n--- Esecuzione migliorativa Terminata ---")




def generate_all_plots(metrics: Metrics, config):
    plot_pod_history(metrics, config)
    plot_queue_history(metrics)
    plot_response_time_trend(metrics)
    plot_response_time_histogram(metrics)
    plot_response_time_boxplot(metrics)
    plot_response_time_scatter(metrics)
    plot_request_heatmap(metrics)
    plot_cumulative_requests(metrics)
    plot_wait_time_trend(metrics)
    plot_wait_time_boxplot(metrics)
    # plot_arrival_vs_service_rate(metrics) può servire?
    # i box_plot escono parecchio schiacciati lasciare?


"""        Min       Q1     Median     Q3       Max
        |---------|=======|=======|---------|
                |       |       |
                baffo    box    baffo              """

if __name__ == "__main__":
    main()

