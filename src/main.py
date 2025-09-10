import os

from analysis.plotter import Plotter
from src import config
from src.analysis.new_plotter import newPlotter
from src.simulation.simulator import Simulator
from src.simulation.simulator_wfq import SimulatorWFQ
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG as RNGManager  # Usiamo un alias per chiarezza
from src.utils.metrics import Metrics # Importa la classe Metrics
from src.utils.metrics_with_priority import MetricsWithPriority # Importa la classe MetricsWithPriority
# from src.utils.acs import batch_means, compute_batch_size # Non più usati direttamente qui
import matplotlib.pyplot as plt


def main():
    """
    Funzione principale che orchestra l'intero processo di simulazione.
    """
    print("--- Inizio Progetto di Simulazione E-commerce (Versione con Rigore Metodologico) ---")

    NUM_REPLICATIONS = 2

    arrival_scenarios = {
        # "tasso_85": lambda t: 85,
        "tasso_170": lambda t: 170,
        #"tasso_100": lambda t: 255, # Temporaneamente escluso per test rapidi
    }

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    all_results = {scenario_name: {} for scenario_name in arrival_scenarios}

    # --- CICLO ESTERNO: REPLICHE ---
    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        # LA CHIAMATA ORA E' QUI: nel ciclo esterno, prima del ciclo interno.
        # Generiamo UN SOLO set di stream e UN SOLO seed per l'INTERA replica.
        replication_streams, rep_seed = rng_manager.get_replication_streams()
        print(f"--- Replica {i+1} utilizzerà il SEED master per tutti gli scenari: {rep_seed} ---")

        # --- CICLO INTERNO: SCENARI ---
        for scenario_name, lambda_fn in arrival_scenarios.items():
            print(f"\n--- ESECUZIONE SCENARIO: {scenario_name.upper()} ---")

            # NON generiamo più un nuovo stream qui, ma usiamo quello definito sopra.
            # Questo garantisce che 'replication_streams' sia identico per
            # 'tasso_70' e 'tasso_85' all'interno della stessa replica 'i'.

            # --- ESECUZIONE BASELINE ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO BASELINE (FIFO) ---")

            simulator_base = Simulator(
                config, Metrics,
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_base.run(simulation_duration=config.SIMULATION_TIME)
            metrics_base = simulator_base.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

            # --- ESECUZIONE MIGLIORATA ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO MIGLIORATO (PRIORITY) ---")

            simulator_prio = SimulatorWithPriority(config,
                                                   MetricsWithPriority,
                                                   arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                                                   service_rng=replication_streams['service'], lambda_function=lambda_fn)
            simulator_prio.run(simulation_duration=config.SIMULATION_TIME)
            metrics_prio = simulator_prio.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

            # Salva i risultati di questa replica/scenario
            all_results[scenario_name][i] = {
                'baseline': metrics_base,
                'priority': metrics_prio,
                'seed': rep_seed # Il seed salvato sarà lo stesso per tutti gli scenari di questa replica
            }

            # --- Generazione report dettagliati per QUESTA specifica run (singola replica/scenario) ---
            print(f"\n--- Generazione report completo per {scenario_name}, Replica {i+1} ---")
            output_folder_single_run = f"output/single_run_plots/{scenario_name}/replica_{i+1}"

            # Determina i parametri di carico per questa specifica run (scenario)
            current_base_load = lambda_fn(0)
            current_peak_load = lambda_fn(0)

            current_peak_start = getattr(config, 'PEAK_START_TIME', 0)
            current_peak_end = getattr(config, 'PEAK_END_TIME', 0)

            single_run_plotter = Plotter(metrics_base, metrics_prio, config)
            single_run_plotter.generate_comprehensive_report(
                output_dir=output_folder_single_run,
                run_prefix=f"{scenario_name}_repl{i+1}",
                peak_start=current_peak_start,
                peak_end=current_peak_end,
                base_load=current_base_load,
                peak_load=current_peak_load
            )

    print(f"\n\n{'='*30} FINE DI TUTTE LE REPLICHE E SCENARI {'='*30}")

    # --- CHIAMATA AL METODO generate_replication_reports DOPO TUTTE LE SIMULAZIONI ---
    # Ora che 'all_results' è completamente popolato, possiamo generare i report complessivi
    # che confrontano le repliche e gli scenari.

    # Per inizializzare il Plotter per i report aggregati, prendiamo le metriche
    # della prima replica del primo scenario. Queste servono solo a soddisfare il costruttore
    # ma i metodi di report aggregati useranno 'all_results' direttamente.
    first_scenario_key = next(iter(all_results))
    first_replica_key = next(iter(all_results[first_scenario_key]))
    initial_metrics_for_plotter = all_results[first_scenario_key][first_replica_key]['baseline']
    initial_metrics_prio_for_plotter = all_results[first_scenario_key][first_replica_key]['priority']

    overall_plotter = Plotter(initial_metrics_for_plotter, initial_metrics_prio_for_plotter, config)

    # Chiamata al metodo per generare i report di replica aggregati
    overall_plotter.generate_replication_reports(
        all_results=all_results,
        num_replications=NUM_REPLICATIONS,
        arrival_scenarios=arrival_scenarios, # Passa il dizionario degli scenari di arrivo
        output_dir='output/aggregated_replication_reports' # Directory dedicata per i report aggregati
    )

    print("\n--- Generazione di tutti i report di simulazione completata ---")

    return all_results, NUM_REPLICATIONS


def run_steady_state_experiment(rng_manager: RNGManager):
    """
    Esegue la simulazione a orizzonte infinito per l'analisi di regime permanente.
    """
    print("\n--- AVVIO ESPERIMENTO STEADY-STATE A ORIZZONTE INFINITO ---")
    output_dir = "plots/steady_state"
    os.makedirs(output_dir, exist_ok=True)

    steady_lambda_fn = lambda t: 170
    print(f"--- Tasso di arrivo (lambda) per Steady-State: {steady_lambda_fn(0)} richieste/secondo ---")

    steady_streams_dict, steady_rep_seed = rng_manager.get_replication_streams()
    print(f"--- La run Steady-State utilizzerà il SEED master: {steady_rep_seed} ---")


    print("\n--- Esecuzione Scenario Baseline (Steady-State) ---")
    simulator_baseline = Simulator(config, Metrics,
                                   arrival_rng=steady_streams_dict['arrivals'],
                                   choice_rng=steady_streams_dict['choice'],
                                   service_rng=steady_streams_dict['service'],
                                   lambda_function=steady_lambda_fn
                                   )
    simulator_baseline.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_baseline = simulator_baseline.metrics_agg

    print("\n--- Esecuzione Scenario con Priorità (Steady-State) ---")
    simulator_prio = SimulatorWithPriority(
        config, MetricsWithPriority,
        arrival_rng=steady_streams_dict['arrivals'],
        choice_rng=steady_streams_dict['choice'],
        service_rng=steady_streams_dict['service'],
        lambda_function=steady_lambda_fn
    )
    #DISATTIVATO PER GRAFICI
    simulator_prio.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_prio = simulator_prio.metrics_agg

    print("\n--- Esecuzione Scenario WFQ (Steady-State) ---")
    simulator_wfq=SimulatorWFQ(
        config, MetricsWithPriority,
        arrival_rng=steady_streams_dict['arrivals'],
        choice_rng=steady_streams_dict['choice'],
        service_rng=steady_streams_dict['service'],
        lambda_function=steady_lambda_fn
    )

    #DISATTIVATO PER GRAFICI
    simulator_wfq.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_wfq = simulator_wfq.metrics_agg

    # --- CALCOLO WARM-UP E ANALISI STEADY-STATE ---
    print("\n--- Calcolo Warm-up e Analisi Steady-State per Tempi di Risposta e Throughput ---")

    # Inizializza gli analizzatori con i dati completi (inclusi warm-up)
    analyzer_baseline = SteadyStateAnalyzer(metrics_baseline, config)
    analyzer_prio = SteadyStateAnalyzer(metrics_prio, config)
    analyzer_wfq = SteadyStateAnalyzer(metrics_wfq, config)

    # Estrai i campioni COMPLETI con i timestamp per la stima del warm-up.
    # Useremo questi per convertire l'indice del warm-up in tempo.
    all_response_data_baseline = analyzer_baseline.extract_full_response_data()
    all_response_data_prio = analyzer_prio.extract_full_response_data()
    all_response_data_wfq = analyzer_wfq.extract_full_response_data()

    # Estrai i soli valori per l'analisi del warm-up (senza timestamp)
    all_response_times_values_baseline = analyzer_baseline.extract_response_times_values()
    all_response_times_values_prio = analyzer_prio.extract_response_times_values()
    all_response_times_values_wfq = analyzer_wfq.extract_response_times_values()


    # Stima la durata del warm-up per tutti gli scenari (ORA restituisce TEMPI in secondi)
    estimated_warmup_duration_baseline = analyzer_baseline.estimate_warmup(
        all_response_times_values_baseline, all_response_data_baseline
    )
    estimated_warmup_duration_prio = analyzer_prio.estimate_warmup(
        all_response_times_values_prio, all_response_data_prio
    )
    estimated_warmup_duration_wfq = analyzer_wfq.estimate_warmup(
        all_response_times_values_wfq, all_response_data_wfq
    )

    print(f"Warm-up stimato per Baseline: {estimated_warmup_duration_baseline:.2f} secondi.")
    print(f"Warm-up stimato per Priorità: {estimated_warmup_duration_prio:.2f} secondi.")
    print(f"Warm-up stimato per WFQ: {estimated_warmup_duration_wfq:.2f} secondi.")


    # --- Analisi dei Tempi di Risposta (Overall) ---
    print("\n--- Esecuzione Batch Means per i Tempi di Risposta Complessivi ---")

    # Estrai i campioni DOPO aver rimosso il warm-up per i valori dei tempi di risposta.
    samples_baseline_post_warmup_values = [v for t, v in all_response_data_baseline if t >= estimated_warmup_duration_baseline]
    samples_prio_post_warmup_values = [v for t, v in all_response_data_prio if t >= estimated_warmup_duration_prio]
    samples_wfq_post_warmup_values = [v for t, v in all_response_data_wfq if t >= estimated_warmup_duration_wfq]

    # Utilizza la funzione steady_state_analysis dell'analizzatore
    results_rt_baseline = analyzer_baseline.steady_state_analysis(
        samples_baseline_post_warmup_values, confidence=config.CONFIDENCE_LEVEL
    )
    analyzer_baseline.print_ci_results(results_rt_baseline, "Baseline Overall Response Time")

    results_rt_prio = analyzer_prio.steady_state_analysis(
        samples_prio_post_warmup_values, confidence=config.CONFIDENCE_LEVEL
    )
    analyzer_prio.print_ci_results(results_rt_prio, "Priority Overall Response Time")

    results_rt_wfq = analyzer_wfq.steady_state_analysis(
        samples_wfq_post_warmup_values, confidence=config.CONFIDENCE_LEVEL
    )
    analyzer_wfq.print_ci_results(results_rt_wfq, "WFQ Overall Response Time")


    # --- Analisi del Throughput ---
    print("\n--- Esecuzione Batch Means per il Throughput ---")

    # Ottieni tutti i timestamp di completamento per il calcolo del throughput
    all_completion_timestamps_baseline = [t for t, _ in all_response_data_baseline]
    all_completion_timestamps_prio = [t for t, _ in all_response_data_prio]
    all_completion_timestamps_wfq = [t for t, _ in all_response_data_wfq]

    results_throughput_baseline = analyzer_baseline.calculate_throughput_ci(
        all_completion_timestamps_baseline, estimated_warmup_duration_baseline
    )
    analyzer_baseline.print_ci_results(results_throughput_baseline, "Baseline Throughput")

    results_throughput_prio = analyzer_prio.calculate_throughput_ci(
        all_completion_timestamps_prio, estimated_warmup_duration_prio
    )
    analyzer_prio.print_ci_results(results_throughput_prio, "Priority Throughput")

    results_throughput_wfq = analyzer_wfq.calculate_throughput_ci(
        all_completion_timestamps_wfq, estimated_warmup_duration_wfq
    )
    analyzer_wfq.print_ci_results(results_throughput_wfq, "WFQ Throughput")


    # 1. IMPOSTA LO STILE
    plt.style.use('./style/plot_style.mplstyle')
    print("Stile 'plot_style.mplstyle' caricato per steady state.")


    print("\n--- Generazione Report Steady-State ---")
    # Passa i dizionari completi dei risultati per maggiore flessibilità
    steady_plotter = SteadyStatePlotter(metrics_baseline, metrics_prio, metrics_wfq, config)

    # MODIFIED: Rimosso gli argomenti analyzer_*
    steady_plotter.generate_steady_state_report(
        response_time_results={
            "baseline": results_rt_baseline,
            "priority": results_rt_prio,
            "wfq": results_rt_wfq
        },
        throughput_results={
            "baseline": results_throughput_baseline,
            "priority": results_throughput_prio,
            "wfq": results_throughput_wfq
        },
        warmup={
            "baseline" : estimated_warmup_duration_baseline,
            "priority" : estimated_warmup_duration_prio,
            "wfq": estimated_warmup_duration_wfq
        },
        output_dir=output_dir
    )
    print("\n--- Fine dell'analisi Steady-State ---")


def print_final_debug_summary(all_results: dict):
    """
    Stampa un riepilogo di debug con le metriche chiave per ogni singola esecuzione.
    Include controlli di consistenza e metriche di performance di alto livello.
    """
    print("\n" + "="*45 + " RIASSUNTO DI DEBUG FINALE " + "="*45)

    for scenario_name, replications in all_results.items():
        print(f"\n--- SCENARIO: {scenario_name.upper()} ---")
        for i, rep_data in replications.items():
            metrics_base = rep_data['baseline']
            metrics_prio = rep_data['priority']
            seed = rep_data['seed']

            # --- Calcoli per il Baseline ---
            total_b = metrics_base.total_requests_generated
            served_b = metrics_base.total_requests_served
            lost_b = sum(metrics_base.requests_timed_out_data.values())
            queue_remaining_b = total_b - served_b - lost_b
            consistency_check_b = "OK" if queue_remaining_b >= 0 else "ERRORE"

            # --- FIX: Convertiamo esplicitamente la media in float ---
            # Aggiungiamo anche un controllo nel caso in cui non ci siano state richieste servite
            if metrics_base.global_response_times_welford.mean is not None:

                avg_resp_time_b = metrics_base.global_response_times_welford.mean
            else:
                avg_resp_time_b = 0.0

            # --- Calcoli per il Prioritario ---
            total_p = len(metrics_prio.request_generation_timestamps)
            served_p = sum(metrics_prio.requests_completed_by_priority.values())
            lost_p = sum(metrics_prio.requests_timed_out_by_priority.values())
            queue_remaining_p = total_p - served_p - lost_p
            consistency_check_p = "OK" if queue_remaining_p >= 0 else "ERRORE"

            # --- FIX: Convertiamo esplicitamente la media in float ---
            if metrics_prio.global_welford_response.mean is not None:
                avg_resp_time_p =metrics_prio.global_welford_response.mean
            else:
                avg_resp_time_p = 0.0

            print(f"  Replica {i+1} (Seed: {seed}):")
            print(f"    [Baseline]   | "
                  f"Generati: {total_b:<5} | Serviti: {served_b:<5} | Persi: {lost_b:<5} | In Coda: {queue_remaining_b:<4} | "
                  f"Check: {consistency_check_b:<7} | E[T]: {avg_resp_time_b:.4f}s")

            print(f"    [Prioritario]| "
                  f"Generati: {total_p:<5} | Serviti: {served_p:<5} | Persi: {lost_p:<5} | In Coda: {queue_remaining_p:<4} | "
                  f"Check: {consistency_check_p:<7} | E[T]: {avg_resp_time_p:.4f}s")


if __name__ == "__main__":

    # 1. Esegui tutte le simulazioni e ottieni i risultati
    all_results, num_replications = main()

    # 2. Se le simulazioni hanno prodotto risultati, procedi con l'analisi aggregata
    if all_results:
        # Inizializziamo un Plotter. I dati passati all'init sono irrilevanti
        # per il metodo di plotting aggregato, che riceve tutto ciò di cui ha bisogno.
        final_plotter = newPlotter(None, None, config)

        # 1. IMPOSTA LO STILE UNA VOLTA PER TUTTE
        plt.style.use('./style/plot_style.mplstyle')
        print("Stile 'plot_style.mplstyle' caricato globalmente.")

        # DISATTIVATO PERCHè SE NE OCCUPA BLACK FRIDAY
        # Chiamiamo il metodo corretto per generare i grafici delle tracce
        # final_plotter.plot_replication_traces_per_scenario(all_results)

        print_final_debug_summary(all_results)

    # 3. Esegui l'analisi steady-state se è abilitata nel config
    #    (Questa parte è separata e non è stata toccata)


    if config.STEADY_ENABLED:
        rng_manager_for_steady_state = RNGManager(master_seed=config.LEHMER_SEED)
        run_steady_state_experiment(rng_manager_for_steady_state)
    else:
        print("\n--- Analisi Steady-State disabilitata in config.py. Per abilitarla, impostare STEADY_ENABLED = True. ---")

    print("\n--- Processo di Simulazione e Analisi Completato ---")