import numpy as np
from src import config

from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
from analysis.data_report import export_summary
from analysis.data_report import *
from analysis.plotter import Plotter
from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority
import os # Importa il modulo os per creare le directory

csv1 = "output/non_prioritized_summary.csv",
csv2 = "output/prioritized_summary.csv",
label1 = "Senza Priorità",
label2 = "Con Priorità"

#tassi di arrivo dinamici
tassi_costanti=[70,85,89] # stabile, vicino l'instabilità e instabile si posso modificare

def call_steady_simulations():
    """
    Funzione principale che orchestra l'intero processo,
    eseguendo le simulazioni per diversi tassi di arrivo.
    """
    print("--- Inizio Simulazione Steady-State ---")
    run_steady_state_experiment()
    #simple_simulation()

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