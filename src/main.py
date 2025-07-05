# main.py - VERSIONE MERGED

import numpy as np
from src import config
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
from analysis.data_report import export_summary
from analysis.plotter import Plotter
from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority
import os # Importa il modulo os per creare le directory

def main():
    """
    Funzione principale che orchestra l'intero processo,
    eseguendo le simulazioni per diversi tassi di arrivo.
    """
    print("--- Inizio Progetto di Simulazione E-commerce ---")

    # Definiamo i tassi di arrivo da testare
    # Puoi aggiungere funzioni lambda più complesse qui se vuoi
    arrival_scenarios = {
        "tasso_70": lambda t: 70,
        "tasso_85": lambda t: 85,
        "tasso_89": lambda t: 89,
        # Esempio di tasso variabile:
        # "ciclo_giornaliero": lambda t: 50 + 40 * np.sin(2 * np.pi * t / (config.SIMULATION_TIME / 2))
    }

    # Il generatore Lehmer ci fornisce una base di seed riproducibile
    lehmer_rng = LehmerRNG(seed=config.LEHMER_SEED)

    # Eseguiamo un ciclo per ogni scenario di tasso di arrivo
    for scenario_name, lambda_fn in arrival_scenarios.items():
        print(f"\n{'='*20} ESECUZIONE SCENARIO: {scenario_name.upper()} {'='*20}")

        # Generiamo un set di seed UNICO per questo scenario, ma derivato dal Lehmer
        # per garantire che se rieseguiamo tutto, i risultati siano identici.
        # Chiamiamo _next_seed() per assicurarci che ogni ciclo for abbia seed diversi.
        lehmer_rng._next_seed()
        base_seed_for_scenario = lehmer_rng._next_seed()

        # Da questo singolo seed di scenario, deriviamo i 3 seed per i nostri RNG
        scenario_rng_gen = LehmerRNG(seed=base_seed_for_scenario)
        seeds = scenario_rng_gen.get_numpy_seeds(count=3)
        arrival_seed, choice_seed, service_seed = seeds[0], seeds[1], seeds[2]

        # --- ESECUZIONE BASELINE (per questo tasso di arrivo) ---
        print(f"\n--- {scenario_name}: SCENARIO BASELINE (FIFO) ---")
        metrics = Metrics()

        arrival_rng_base = np.random.default_rng(seed=arrival_seed)
        choice_rng_base = np.random.default_rng(seed=choice_seed)
        service_rng_base = np.random.default_rng(seed=service_seed)

        simulator = Simulator(
            config_module=config,
            metrics=metrics,
            arrival_rng=arrival_rng_base,
            choice_rng=choice_rng_base,
            service_rng=service_rng_base,
            lambda_function=lambda_fn  # Passiamo la funzione lambda
        )
        simulator.run()
        metrics.print_summary()
        print("\n--- Esecuzione baseline terminata ---")

        # --- ESECUZIONE MIGLIORATA (per questo tasso di arrivo) ---
        print(f"\n--- {scenario_name}: SCENARIO MIGLIORATO (PRIORITY) ---")
        metrics_prio = MetricsWithPriority(config)

        arrival_rng_prio = np.random.default_rng(seed=arrival_seed)
        choice_rng_prio = np.random.default_rng(seed=choice_seed)
        service_rng_prio = np.random.default_rng(seed=service_seed)

        simulator_prio = SimulatorWithPriority(
            config_module=config,
            metrics=metrics_prio,
            arrival_rng=arrival_rng_prio,
            choice_rng=choice_rng_prio,
            service_rng=service_rng_prio,
            lambda_function=lambda_fn # Passiamo la stessa funzione lambda
        )
        simulator_prio.run()
        metrics_prio.print_summary()
        print("\n--- Esecuzione migliorativa terminata ---")

        # --- ANALISI DEI RISULTATI (per questo tasso di arrivo) ---
        print(f"\n--- Generazione report per lo scenario: {scenario_name} ---")

        # Creiamo una cartella di output specifica per questo scenario
        output_folder = f"output/plots_{scenario_name}"
        os.makedirs(output_folder, exist_ok=True)

        # Esportiamo i dati
        export_summary(metrics_prio, output_dir=output_folder, label=f"{scenario_name}_con_priorita", by_priority=True)
        export_summary(metrics, output_dir=output_folder, label=f"{scenario_name}_senza_priorita", by_priority=False)

        # Generiamo i grafici salvandoli nella cartella dedicata
        plotter = Plotter(metrics, metrics_prio, config)
        plotter.generate_comprehensive_report(output_dir=output_folder, run_prefix=scenario_name)

    print("\nTutte le simulazioni sono terminate.")


if __name__ == "__main__":
    main()