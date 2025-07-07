import numpy as np
from src import config

from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG
from src.utils.metrics import Metrics
from analysis.data_report import export_summary
from analysis.data_report import *
from analysis.validation_plotter import  ValidationPlotter
from analysis.plotter import Plotter
from src.simulation.simulator import Simulator
from src.utils.metrics_with_priority import MetricsWithPriority
import os # Importa il modulo os per creare le directory

csv1 = "output/non_prioritized_summary.csv",
csv2 = "output/prioritized_summary.csv",
label1 = "Senza Priorità",
label2 = "Con Priorità"

def main():
    """
    Funzione principale che orchestra l'intero processo.
    """
    print("--- Inizio Progetto di Simulazione E-commerce ---")

    arrival_scenarios = {
        "tasso_70": lambda t: 70,
        "tasso_85": lambda t: 85,
        "tasso_100": lambda t: 100,
    }

    # Dizionario per raccogliere i risultati di tutte le run
    all_run_metrics = {}

    lehmer_rng = LehmerRNG(seed=config.LEHMER_SEED)


    for scenario_name, lambda_fn in arrival_scenarios.items():
        print(f"\n{'='*20} ESECUZIONE SCENARIO: {scenario_name.upper()} {'='*20}")

        lehmer_rng._next_seed()
        base_seed_for_scenario = lehmer_rng._next_seed()
        scenario_rng_gen = LehmerRNG(seed=base_seed_for_scenario)
        seeds = scenario_rng_gen.get_numpy_seeds(count=3)
        arrival_seed, choice_seed, service_seed = seeds[0], seeds[1], seeds[2]

        # --- ESECUZIONE BASELINE ---
        print(f"\n--- {scenario_name}: SCENARIO BASELINE (FIFO) ---")
        metrics_base = Metrics(config_module=config)
        arrival_rng_base = np.random.default_rng(seed=arrival_seed)
        choice_rng_base = np.random.default_rng(seed=choice_seed)
        service_rng_base = np.random.default_rng(seed=service_seed)
        simulator_base = Simulator(
            config_module=config, metrics=metrics_base, arrival_rng=arrival_rng_base,
            choice_rng=choice_rng_base, service_rng=service_rng_base, lambda_function=lambda_fn
        )
        simulator_base.run(simulation_duration=config.SIMULATION_TIME)
        metrics_base.print_summary()
        print("\n--- Esecuzione baseline terminata ---")

        # --- ESECUZIONE MIGLIORATA ---
        print(f"\n--- {scenario_name}: SCENARIO MIGLIORATO (PRIORITY) ---")
        metrics_prio = MetricsWithPriority(config)
        arrival_rng_prio = np.random.default_rng(seed=arrival_seed)
        choice_rng_prio = np.random.default_rng(seed=choice_seed)
        service_rng_prio = np.random.default_rng(seed=service_seed)
        simulator_prio = SimulatorWithPriority(
            config_module=config, metrics=metrics_prio, arrival_rng=arrival_rng_prio,
            choice_rng=choice_rng_prio, service_rng=service_rng_prio, lambda_function=lambda_fn
        )
        simulator_prio.run(simulation_duration=config.SIMULATION_TIME)
        metrics_prio.print_summary()
        print("\n--- Esecuzione migliorativa terminata ---")

        # --- ANALISI DEI RISULTATI per questo scenario ---
        print(f"\n--- Generazione report per lo scenario: {scenario_name} ---")
        output_folder = f"output/plots_{scenario_name}"
        os.makedirs(output_folder, exist_ok=True)
        #export_summary(metrics_prio, output_dir=output_folder, label=f"{scenario_name}_con_priorita", by_priority=True)
        #export_summary(metrics_base, output_dir=output_folder, label=f"{scenario_name}_senza_priorita", by_priority=False)
        plotter = Plotter(metrics_base, metrics_prio, config)
        plotter.generate_comprehensive_report(output_dir=output_folder, run_prefix=scenario_name)

        # --- AGGIUNTA: SALVA I RISULTATI PER L'ANALISI FINALE ---
        all_run_metrics[scenario_name] = {'baseline': metrics_base, 'priority': metrics_prio}
        # --------------------------------------------------------


    if all_run_metrics:
        validation_plotter = ValidationPlotter(all_run_metrics, config)
        validation_plotter.generate_validation_report()
    # ---------------------------------------------------

    if config.STEADY_ENABLED:
        print("--- Inizio Simulazione Steady-State ---")
        run_steady_state_experiment()
        print("--- Fine Simulazione Steady-State ---")

    print("\nTutte le simulazioni sono terminate.")




def run_steady_state_experiment():
    """
    Esegue entrambe le simulazioni a orizzonte infinito e genera i grafici di
    confronto steady-state.
    """
    print("\n--- AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")

    output_dir = "plots/steady_state"

    # Usiamo un tasso di arrivo fisso per l'analisi, es. 70
    steady_lambda_fn = lambda t: 85

    # Creiamo i 3 generatori RNG necessari, usando il seed di base per riproducibilità
    base_seed = config.LEHMER_SEED
    lehmer_rng = LehmerRNG(seed=base_seed)
    seeds = lehmer_rng.get_numpy_seeds(count=3)
    arrival_seed, choice_seed, service_seed = seeds[0], seeds[1], seeds[2]

    # --- ESECUZIONE BASELINE ---
    print("\n--- Esecuzione Scenario Baseline (Steady-State) ---")
    metrics_baseline = Metrics(config_module=config)
    simulator_baseline = Simulator(
        config_module=config,
        metrics=metrics_baseline,
        arrival_rng=np.random.default_rng(arrival_seed),
        choice_rng=np.random.default_rng(choice_seed),
        service_rng=np.random.default_rng(service_seed),
        lambda_function=steady_lambda_fn
    )
    simulator_baseline.run(simulation_duration=config.STEADY_SIMULATION_TIME)

    # --- ESECUZIONE PRIORITÀ ---
    print("\n--- Esecuzione Scenario con Priorità (Steady-State) ---")
    metrics_prio = MetricsWithPriority(config)
    simulator_prio = SimulatorWithPriority(
        config_module=config,
        metrics=metrics_prio,
        arrival_rng=np.random.default_rng(arrival_seed),
        choice_rng=np.random.default_rng(choice_seed),
        service_rng=np.random.default_rng(service_seed),
        lambda_function=steady_lambda_fn
    )
    simulator_prio.run(simulation_duration=config.STEADY_SIMULATION_TIME)

    # --- ANALISI E PLOTTING FINALE ---
    print("\n--- Generazione Report Steady-State ---")

    # 1. Istanziamo gli oggetti necessari
    analyzer_baseline = SteadyStateAnalyzer(metrics_baseline, config)
    analyzer_prio = SteadyStateAnalyzer(metrics_prio, config)
    steady_plotter = SteadyStatePlotter(metrics_baseline, metrics_prio, config)

    # 2. Facciamo UN'UNICA chiamata al metodo orchestratore
    steady_plotter.generate_steady_state_report(
        analyzer_baseline=analyzer_baseline,
        analyzer_prio=analyzer_prio,
        warmup=config.WARM_UP_TO_STEADY,
        batches=config.NUM_BATCHES,
        output_dir=output_dir
    )

    print("\n--- Fine dell'analisi Steady-State ---")


if __name__ == "__main__":
    main()