import numpy as np
from src import config
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
# ----- MODIFICA QUI ----
from analysis.data_report import *
from analysis.plotter import CSVPlotter

# -------------------------
from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority

csv1 = "output/non_prioritized_summary.csv",
csv2 = "output/prioritized_summary.csv",
label1 = "Senza Priorità",
label2 = "Con Priorità"


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
    metrics.print_summary()

    print("\n--- Esecuzione baseline Terminata ---")

    # --- ESECUZIONE DELLA SIMULAZIONE MIGLIORATA (PRIORITÀ PER WORKER) ---
    print("\n--- SCENARIO MIGLIORATO (ABSTRACT PRIORITY) ---")

    # rng_prio = np.random.default_rng(seed=numpy_seed) #da controllare gestione del RNG!!! capire qual è il modo migliore
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(config, metrics_prio, rng)
    simulator_prio.run()

    metrics_prio.print_summary()

    print("\n--- Esecuzione migliorativa Terminata ---")

    # Dopo la simulazione:
    export_summary(metrics, output_dir="output", label=label1, by_priority=False)
    export_summary(metrics_prio, output_dir="output", label=label2, by_priority=True)

    # Plot confronto
    plotter = CSVPlotter(output_dir="plots")  # Output in /plots

    plotter.compare_bar(csv1, csv2, label1, label2, metric="avg_response_time")

    plotter.compare_lines(csv1, csv2, label1, label2, metric="avg_wait_time")

    # Visualizzazione individuale
    plotter.plot_single_metric(csv1, label1, metric="max_response_time", kind="bar")

    plotter.plot_single_metric(csv2, label2, metric="max_response_time", kind="line")


if __name__ == "__main__":
    main()

    # elisa è stata qui:)
