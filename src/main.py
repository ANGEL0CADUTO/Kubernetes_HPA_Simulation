import numpy as np
from src import config

from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
from analysis.data_report import *
from analysis.plotter import Plotter

from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority

csv1 = "output/non_prioritized_summary.csv",
csv2 = "output/prioritized_summary.csv",
label1 = "Senza Priorità",
label2 = "Con Priorità"

#tassi di arrivo dinamici
tassi_costanti=[70,85,89] # stabile, vicino l'instabilità e instabile si posso modificare

def main():
    """
    Funzione principale che orchestra l'intero processo.
    """
    print("--- Inizio Simulazione Steady-State ---")
    steady_state_simulation_baseline()
    steady_state_simulation_priority()

    # simple_simulation()

def simple_simulation():
    print("--- Inizio Progetto di Simulazione E-commerce ---")

    lehmer_rng = LehmerRNG(seed=config.LEHMER_SEED)
    numpy_seed = lehmer_rng.get_numpy_seed()
    for i, tasso in enumerate(tassi_costanti):
        print(f"\n--- ESECUZIONE DELLA SIMULAZIONE CON TASSO DI ARRIVO {tasso} ---")

        rng = np.random.default_rng(seed=numpy_seed+i+100)

        metrics = Metrics()
        lambda_fn= lambda t, rate=tasso: rate

        simulator = Simulator(config_module=config, metrics=metrics, rng=rng, lambda_function=lambda_fn)
        simulator.run()
        metrics.print_summary()

        print("\n--- Esecuzione baseline Terminata ---")

        # --- ESECUZIONE DELLA SIMULAZIONE MIGLIORATA (PRIORITÀ PER WORKER) ---
        print("\n--- SCENARIO MIGLIORATO (ABSTRACT PRIORITY) ---")

        rng_prio = np.random.default_rng(seed=numpy_seed+i+100)   #stesso seed di baseline
        metrics_prio = MetricsWithPriority(config)
        simulator_prio = SimulatorWithPriority(config, metrics_prio, rng_prio, lambda_function=lambda_fn)
        simulator_prio.run()

        metrics_prio.print_summary()
        print(".............................................")
        metrics.print_summary()

        print("\n--- Esecuzione migliorativa Terminata ---")

        # Dopo la simulazione:
        # --- ANALISI DEI RISULTATI ---
        run_prefix = f"lambda_{tasso}"
        output_folder = f"output/plots_{run_prefix}"
        export_summary(metrics_prio, output_dir="output", label="con_priorita", by_priority=True)
        export_summary(metrics, output_dir="output", label="senza_priorita", by_priority=False)
        plotter = Plotter(metrics,metrics_prio,config)
        plotter.plot_queue_history(output_dir=output_folder,filename=f"{run_prefix}_queue_history.png")
        plotter.plot_response_time_trend(output_dir=output_folder,filename=f"{run_prefix}_response_time_trend.png")
        plotter.plot_pod_history(output_dir=output_folder,filename=f"{run_prefix}_pod_history.png")
        plotter.plot_loss_by_type(output_dir=output_folder,filename=f"{run_prefix}_loss_by_type.png")
        plotter.plot_comparison_dashboard(output_dir=output_folder,filename=f"{run_prefix}_comparison_dashboard.png")
        plotter.plot_wait_time_trend(output_dir=output_folder,filename=f"{run_prefix}_wait_time_trend.png")


def steady_state_simulation_baseline():
    """
    Esegue una singola simulazione a orizzonte lungo per l'analisi di regime permanente
    e calcola gli intervalli di confidenza usando il metodo Batch Means.
    """
    print("\n--- BASELINE: AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")

    # Simulazione senza priorità
    print(f"Esecuzione simulazione per {config.STEADY_SIMULATION_TIME} secondi...")
    rng = np.random.default_rng(seed=config.LEHMER_SEED)
    metrics = Metrics()
    simulator = Simulator(config, metrics, rng)
    simulator.run()
    print("Simulazione Baseline a orizzonte infinito terminata.")

    # Analisi Steady-State
    print("\n--- Analisi Batch Means per il Tempo di Risposta ---")
    analyzer = SteadyStateAnalyzer(metrics, config)

    # Prendi tutti i tempi di risposta grezzi
    all_response_times_data = metrics.get_all_response_times_with_timestamps()

    # Calcola e stampa i risultati
    analysis_results = analyzer.calculate_batch_means_ci(
        metric_data=all_response_times_data,
        warmup_period=config.WARM_UP_TO_STEADY,
        num_batches=config.NUM_BATCHES,
        confidence_level=config.CONFIDENCE_LEVEL
    )

    if analysis_results:
        analyzer.print_ci_results(analysis_results, "Tempo di Risposta")
        analyzer.plot_confidence_interval(analysis_results, "BASELINE: Tempo di Risposta Medio (Steady State)", "ouput/steady_state", "steady_state_baseline")

def steady_state_simulation_priority():
    """
    Esegue una singola simulazione a orizzonte lungo per l'analisi di regime permanente
    e calcola gli intervalli di confidenza usando il metodo Batch Means.
    """
    print("\n--- PRIORITY: AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")

    # Simulazione (es. solo quella con priorità)
    print(f"Esecuzione simulazione per {config.STEADY_SIMULATION_TIME} secondi...")
    rng = np.random.default_rng(seed=config.LEHMER_SEED)
    metrics_prio = MetricsWithPriority(config) # Usiamo la classe di metriche più completa
    simulator_prio = SimulatorWithPriority(config, metrics_prio, rng)
    simulator_prio.run()
    print("Simulazione Priority a orizzonte infinito terminata.")

    # Analisi Steady-State
    print("\n--- Analisi Batch Means per il Tempo di Risposta ---")
    analyzer = SteadyStateAnalyzer(metrics_prio, config)

    # Prendi tutti i tempi di risposta grezzi
    all_response_times_data = metrics_prio.get_all_response_times_with_timestamps()

    # Calcola e stampa i risultati
    analysis_results = analyzer.calculate_batch_means_ci(
        metric_data=all_response_times_data,
        warmup_period=config.WARM_UP_TO_STEADY,
        num_batches=config.NUM_BATCHES,
        confidence_level=config.CONFIDENCE_LEVEL
    )

    if analysis_results:
        analyzer.print_ci_results(analysis_results, "Tempo di Risposta")
        analyzer.plot_confidence_interval(analysis_results, "PRIORITY: Tempo di Risposta Medio (Steady State)","ouput/steady_state", "steady_state_priority")


if __name__ == "__main__":
    main()

