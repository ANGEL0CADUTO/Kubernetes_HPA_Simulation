import numpy as np
from src import config
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
# ----- MODIFICA QUI -----
from src.analysis.plotter import plot_pod_history, plot_queue_history, plot_response_time_trend, \
    plot_response_time_histogram
# -------------------------
from src.simulation.simulator import Simulator


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

    # --- MODIFICA QUI ---
    # Genera i grafici di andamento temporale
    plot_pod_history(metrics, config)
    plot_queue_history(metrics)
    plot_response_time_trend(metrics)  # Chiama la nuova funzione

    # Genera il grafico con gli istogrammi
    plot_response_time_histogram(metrics)  # Nome funzione cambiato per chiarezza
    # --------------------

    print("\n--- Esecuzione Terminata ---")


if __name__ == "__main__":
    main()

    # elisa è stata qui :)
