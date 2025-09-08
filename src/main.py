
import os

from analysis.plotter import Plotter
from src import config

from src.simulation.simulator import Simulator
from src.simulation.simulator_wfq import SimulatorWFQ
from src.simulation.simulator_with_priority import SimulatorWithPriority
from src.steady_state_analysis.steady_state_analyzer import SteadyStateAnalyzer
from src.steady_state_analysis.steady_state_plotter import SteadyStatePlotter
from src.utils.lehmer_rng import LehmerRNG as RNGManager  # Usiamo un alias per chiarezza
from src.utils.metrics import Metrics # Importa la classe Metrics
from src.utils.metrics_with_priority import MetricsWithPriority # Importa la classe MetricsWithPriority
from src.utils.acs import batch_means, compute_batch_size


def main():
    """
    Funzione principale che orchestra l'intero processo di simulazione.
    """
    print("--- Inizio Progetto di Simulazione E-commerce (Versione con Rigore Metodologico) ---")

    NUM_REPLICATIONS = 2 # Definisci qui il numero di repliche

    # Definisci i tuoi scenari di arrivo
    arrival_scenarios = {
        "tasso_70": lambda t: 85,
        "tasso_85": lambda t: 170,
        #"tasso_100": lambda t: 255, # Temporaneamente escluso per test rapidi
    }

    rng_manager = RNGManager(master_seed=config.LEHMER_SEED)
    all_results = {scenario_name: {} for scenario_name in arrival_scenarios}

    # --- CICLO ESTERNO: REPLICHE ---
    for i in range(NUM_REPLICATIONS):
        print(f"\n{'='*30} INIZIO REPLICA {i + 1}/{NUM_REPLICATIONS} {'='*30}")

        # Generiamo UN SOLO set di stream e UN SOLO seed per l'INTERA replica.
        replication_streams, rep_seed = rng_manager.get_replication_streams()
        print(f"--- Replica {i+1} utilizzerà il SEED master per tutti gli scenari: {rep_seed} ---")

        # --- CICLO INTERNO: SCENARI ---
        for scenario_name, lambda_fn in arrival_scenarios.items():
            print(f"\n--- ESECUZIONE SCENARIO: {scenario_name.upper()} ---")

            # --- ESECUZIONE BASELINE ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO BASELINE (FIFO) ---")
            # NON creare qui l'istanza delle metriche. Passa la CLASSE Metrics.
            simulator_base = Simulator(
                config, Metrics, # <-- MODIFICA QUI: passa la CLASSE Metrics
                arrival_rng=replication_streams['arrivals'], choice_rng=replication_streams['choice'],
                service_rng=replication_streams['service'], lambda_function=lambda_fn
            )
            simulator_base.run(simulation_duration=config.SIMULATION_TIME)
            metrics_base = simulator_base.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

            # --- ESECUZIONE MIGLIORATA ---
            print(f"\n--- {scenario_name} (Replica {i+1}): SCENARIO MIGLIORATO (PRIORITY) ---")
            # NON creare qui l'istanza delle metriche. Passa la CLASSE MetricsWithPriority.
            simulator_prio = SimulatorWithPriority(config,
                                                   MetricsWithPriority, # <-- MODIFICA QUI: passa la CLASSE MetricsWithPriority
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
            current_base_load = lambda_fn(0) # Valuta lambda a t=0 per ottenere il tasso costante
            current_peak_load = lambda_fn(0)
            # Prendi peak_start e peak_end dalla configurazione, se esistono, altrimenti 0
            current_peak_start = getattr(config, 'PEAK_START_TIME', 0)
            current_peak_end = getattr(config, 'PEAK_END_TIME', 0)

            # Inizializza un Plotter per questa singola run e genera il report completo
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

    steady_lambda_fn = lambda t: 85
    steady_streams_dict, steady_rep_seed = rng_manager.get_replication_streams()
    print(f"--- La run Steady-State utilizzerà il SEED master: {steady_rep_seed} ---")


    print("\n--- Esecuzione Scenario Baseline (Steady-State) ---")
    # NON creare qui l'istanza delle metriche. Passa la CLASSE Metrics.
    simulator_baseline = Simulator(config, Metrics, # <-- MODIFICA QUI: passa la CLASSE Metrics
                                   arrival_rng=steady_streams_dict['arrivals'],
                                   choice_rng=steady_streams_dict['choice'],
                                   service_rng=steady_streams_dict['service'],
                                   lambda_function=steady_lambda_fn
                                   )
    simulator_baseline.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_baseline = simulator_baseline.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

    print("\n--- Esecuzione Scenario con Priorità (Steady-State) ---")
    # NON creare qui l'istanza delle metriche. Passa la CLASSE MetricsWithPriority.
    simulator_prio = SimulatorWithPriority(
        config, MetricsWithPriority, # <-- MODIFICA QUI: passa la CLASSE MetricsWithPriority
        arrival_rng=steady_streams_dict['arrivals'],
        choice_rng=steady_streams_dict['choice'],
        service_rng=steady_streams_dict['service'],
        lambda_function=steady_lambda_fn
    )
    simulator_prio.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_prio = simulator_prio.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

    # Anche per WFQ, passa la CLASSE MetricsWithPriority e recupera l'istanza
    # assumendo che SimulatorWFQ abbia una struttura simile.
    simulator_wfq=SimulatorWFQ(
        config, MetricsWithPriority, # <-- MODIFICA QUI: passa la CLASSE MetricsWithPriority
        arrival_rng=steady_streams_dict['arrivals'],
        choice_rng=steady_streams_dict['choice'],
        service_rng=steady_streams_dict['service'],
        lambda_function=steady_lambda_fn
    )
    simulator_wfq.run(simulation_duration=config.STEADY_SIMULATION_TIME)
    metrics_wfq = simulator_wfq.metrics_agg # <-- RECUPERA LE METRICHE AGGREGATE DAL SIMULATORE

    # --- CALCOLO WARM-UP E BATCH MEANS ---
    print("\n--- Calcolo Warm-up e Batch Means per i Tempi di Risposta Complessivi ---")

    # Inizializza gli analizzatori con i dati completi (inclusi warm-up)
    analyzer_baseline_full_data = SteadyStateAnalyzer(metrics_baseline, config)
    analyzer_prio_full_data = SteadyStateAnalyzer(metrics_prio, config)
    analyzer_wfq_full_data = SteadyStateAnalyzer(metrics_wfq, config)

    # Estrai i campioni COMPLETI con i timestamp per la stima del warm-up.
    # Useremo questi per convertire l'indice del warm-up in tempo.
    all_response_data_baseline = analyzer_baseline_full_data.extract_full_response_data()
    all_response_data_prio = analyzer_prio_full_data.extract_full_response_data()
    all_response_data_wfq = analyzer_wfq_full_data.extract_full_response_data()

    # Estrai i soli valori per l'analisi del warm-up (senza timestamp)
    all_response_times_values_baseline = analyzer_baseline_full_data.extract_response_times_values()
    all_response_times_values_prio = analyzer_prio_full_data.extract_response_times_values()
    all_response_times_values_wfq = analyzer_wfq_full_data.extract_response_times_values()


    # Stima la durata del warm-up per entrambi gli scenari (ORA restituisce TEMPI in secondi)
    estimated_warmup_duration_baseline = analyzer_baseline_full_data.estimate_warmup(
        all_response_times_values_baseline, all_response_data_baseline
    )
    estimated_warmup_duration_prio = analyzer_prio_full_data.estimate_warmup(
        all_response_times_values_prio, all_response_data_prio
    )
    estimated_warmup_duration_wfq = analyzer_wfq_full_data.estimate_warmup(
        all_response_times_values_wfq, all_response_data_wfq
    )

    # Nota il cambiamento nell'output della stampa: "secondi" invece di "periodi/s"
    print(f"Warm-up stimato per Baseline: {estimated_warmup_duration_baseline:.2f} secondi.")
    print(f"Warm-up stimato per Priorità: {estimated_warmup_duration_prio:.2f} secondi.")
    print(f"Warm-up stimato per WFQ: {estimated_warmup_duration_wfq:.2f} secondi.")


    # Estrai i campioni DOPO aver rimosso il warm-up.
    # CRITICAL CHANGE: ora filtriamo i TIMESTAMP di completamento,
    # non più solo i valori del tempo di risposta.
    # `calculate_throughput_ci` ha bisogno dei timestamp di completamento.

    # Per il calcolo del Batch Means per il tempo di risposta OVERALL
    samples_baseline_post_warmup_values = [v for t, v in metrics_baseline.get_all_response_times_with_timestamps() if t >= estimated_warmup_duration_baseline]
    samples_prio_post_warmup_values = [v for t, v in metrics_prio.get_all_response_times_with_timestamps() if t >= estimated_warmup_duration_prio]
    samples_wfq_post_warmup_values = [v for t, v in metrics_wfq.get_all_response_times_with_timestamps() if t >= estimated_warmup_duration_wfq]

    # Calcolo b, k_ottimale e rho1 per baseline usando i dati POST-WARMUP
    b_base, k_base_optimal, rho_base = compute_batch_size(
        samples_baseline_post_warmup_values, k_initial_target=config.BATCH_K, threshold=config.BATCH_THRESHOLD
    )

    mean_base, ci95_base, half_width_base = None, (None, None), None
    # Verifica che b_base e k_base_optimal siano numeri validi prima di usare
    if b_base is None or k_base_optimal is None or b_base <= 0 or k_base_optimal <= 1: # k_optimal deve essere almeno 2 per batch_means
        print("ATTENZIONE: Impossibile determinare dimensioni/numero batch validi per Baseline. Impostazione valori di fallback.")
    else:
        batch_means_results_base = batch_means(samples_baseline_post_warmup_values, b_base, k_base_optimal, confidence=config.CONFIDENCE_LEVEL)
        if batch_means_results_base: # Verifica che batch_means abbia prodotto risultati
            mean_base = batch_means_results_base['mean']
            ci95_base = batch_means_results_base['ci']
            half_width_base = batch_means_results_base['half_width']
    print(f"Baseline (Overall Response Time): b={b_base}, k={k_base_optimal}, rho1={rho_base:.3f}, media={mean_base:.4f}, IC95={ci95_base}")


    # Calcolo b, k_ottimale e rho1 per priorità usando i dati POST-WARMUP
    b_prio, k_prio_optimal, rho_prio = compute_batch_size(
        samples_prio_post_warmup_values, k_initial_target=config.BATCH_K, threshold=config.BATCH_THRESHOLD
    )

    mean_prio, ci95_prio, half_width_prio = None, (None, None), None
    if b_prio is None or k_prio_optimal is None or b_prio <= 0 or k_prio_optimal <= 1:
        print("ATTENZIONE: Impossibile determinare dimensioni/numero batch validi per Priorità. Impostazione valori di fallback.")
    else:
        batch_means_results_prio = batch_means(samples_prio_post_warmup_values, b_prio, k_prio_optimal, confidence=config.CONFIDENCE_LEVEL)
        if batch_means_results_prio:
            mean_prio = batch_means_results_prio['mean']
            ci95_prio = batch_means_results_prio['ci']
            half_width_prio = batch_means_results_prio['half_width']
    print(f"Priorità (Overall Response Time): b={b_prio}, k={k_prio_optimal}, rho1={rho_prio:.3f}, media={mean_prio:.4f}, IC95={ci95_prio}")

    # Calcolo b, k_ottimale e rho1 per WFQ usando i dati POST-WARMUP
    b_wfq, k_wfq_optimal, rho_wfq = compute_batch_size( # Nuovo b, k per WFQ se necessario
        samples_wfq_post_warmup_values, k_initial_target=config.BATCH_K, threshold=config.BATCH_THRESHOLD
    )

    mean_wfq, ci95_wfq, half_width_wfq = None, (None, None), None
    if b_wfq is None or k_wfq_optimal is None or b_wfq <= 0 or k_wfq_optimal <= 1:
        print("ATTENZIONE: Impossibile determinare dimensioni/numero batch validi per WFQ. Impostazione valori di fallback.")
    else:
        batch_means_results_wfq = batch_means(samples_wfq_post_warmup_values, b_wfq, k_wfq_optimal, confidence=config.CONFIDENCE_LEVEL)
        if batch_means_results_wfq:
            mean_wfq = batch_means_results_wfq['mean']
            ci95_wfq = batch_means_results_wfq['ci']
            half_width_wfq = batch_means_results_wfq['half_width']
    print(f"WFQ (Overall Response Time): b={b_wfq}, k={k_wfq_optimal}, rho1={rho_wfq:.3f}, media={mean_wfq:.4f}, IC95={ci95_wfq}")


    print("\n--- Generazione Report Steady-State ---")
    steady_plotter = SteadyStatePlotter(metrics_baseline, metrics_prio, metrics_wfq,config) # Passa metrics_wfq qui


    # La funzione generate_steady_state_report deve essere aggiornata nel Plotter
    # per accettare gli analizzatori e i valori di warmup corretti.
    steady_plotter.generate_steady_state_report(
        analyzer_baseline=analyzer_baseline_full_data,
        analyzer_prio=analyzer_prio_full_data,
        analyzer_wfq=analyzer_wfq_full_data, # Assicurati che Plotter sia aggiornato per gestirlo
        warmup={
            "baseline" : estimated_warmup_duration_baseline,
            "priority" : estimated_warmup_duration_prio,
            "wfq": estimated_warmup_duration_wfq
        },
        batches={ # Questi batches si riferiscono al calcolo del tempo di risposta complessivo
            "baseline" : (mean_base, ci95_base, b_base, k_base_optimal),
            "priority": (mean_prio, ci95_prio, b_prio, k_prio_optimal),
            "wfq": (mean_wfq, ci95_wfq, b_wfq, k_wfq_optimal) # Utilizza b_wfq e k_wfq_optimal qui
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
        final_plotter = Plotter(None, None, config)

        # Chiamiamo il metodo corretto per generare i grafici delle tracce
        final_plotter.plot_replication_traces_per_scenario(all_results, num_replications)
        # AGGIUNGI QUESTA CHIAMATA
        print_final_debug_summary(all_results)

    # 3. Esegui l'analisi steady-state se è abilitata nel config
    #    (Questa parte è separata e non è stata toccata)
    if config.STEADY_ENABLED:
        rng_manager_for_steady_state = RNGManager(master_seed=config.LEHMER_SEED)
        run_steady_state_experiment(rng_manager_for_steady_state)
    else:
        print("\n--- Analisi Steady-State disabilitata in config.py ---")

    print("\n--- Processo di Simulazione e Analisi Completato ---")