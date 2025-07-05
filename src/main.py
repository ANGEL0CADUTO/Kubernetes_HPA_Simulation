import numpy as np
from src import config

from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
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
    run_steady_state_experiment()

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


def run_steady_state_experiment():
    """
    Esegue entrambe le simulazioni a orizzonte infinito e genera i grafici di
    confronto steady-state, suddividendo le richieste per tipo.
    """
    print("\n--- AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")

    output_dir = "plots/steady_state"

    # --- ESECUZIONE BASELINE ---
    print("\n--- Esecuzione Scenario Baseline ---")
    rng_baseline = np.random.default_rng(config.LEHMER_SEED)
    metrics_baseline = Metrics()
    simulator_baseline = Simulator(config, metrics_baseline, rng_baseline)
    simulator_baseline.run()

    # --- ESECUZIONE PRIORITÀ ---
    print("\n--- Esecuzione Scenario con Priorità ---")
    rng_prio = np.random.default_rng(config.LEHMER_SEED)
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(config, metrics_prio, rng_prio)
    simulator_prio.run()

    # --- PLOTTING FINALE CONFRONTO ---
    print("\n--- Generazione Report Steady-State ---")
    steady_plotter = SteadyStatePlotter(metrics_baseline, metrics_prio, config)

    # Creiamo gli analizzatori da passare al plotter
    analyzer_baseline = SteadyStateAnalyzer(metrics_baseline, config)
    analyzer_prio = SteadyStateAnalyzer(metrics_prio, config)

    # Chiamiamo la nuova funzione per le perdite per tipo
    steady_plotter.plot_steady_state_loss_by_type_ci(analyzer_baseline, analyzer_prio, config.WARM_UP_TO_STEADY,
                                                     config.NUM_BATCHES, output_dir)

    # Possiamo ancora chiamare la funzione per i tempi
    steady_plotter.plot_steady_state_times_by_type(analyzer_baseline, analyzer_prio, config.WARM_UP_TO_STEADY,
                                                    config.NUM_BATCHES, output_dir)
    print("\n--- Fine dell'analisi Steady-State ---")


if __name__ == "__main__":
    main()

