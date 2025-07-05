import numpy as np
from src import config


from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
# ----- MODIFICA QUI ----
from analysis.data_report import *
from analysis.plotter import Plotter

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

    rng_prio = np.random.default_rng(seed=numpy_seed)   #stesso seed di baseline
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(config, metrics_prio, rng_prio)
    simulator_prio.run()

    metrics_prio.print_summary()
    print(".............................................")
    metrics.print_summary()

    print("\n--- Esecuzione migliorativa Terminata ---")

    # Dopo la simulazione:
    # --- ANALISI DEI RISULTATI ---
    export_summary(metrics_prio, output_dir="output", label="con_priorita", by_priority=True)
    export_summary(metrics, output_dir="output", label="senza_priorita", by_priority=False)
    plotter = Plotter(metrics,metrics_prio,config)
    plotter.generate_comprehensive_report()



if __name__ == "__main__":
    main()

