import os
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


from src.utils.acs import batch_means, compute_batch_size, ljung_box_test


class SimulationMode(Enum):
    TERMINATING = "terminating"
    STEADY_STATE = "steady_state"


class SteadyStateAnalyzer:
    """
    Classe per l'analisi dello stato stazionario (steady-state) o replicata (terminating)
    dei risultati di simulazione. Implementa tecniche come Batch Means per la stima
    di intervalli di confidenza e metodi per la stima del periodo di warm-up.

    È compatibile con le classi Metrics e MetricsWithPriority.
    """
    def __init__(self, metrics, config):
        """
        Inizializza l'analizzatore con le metriche della simulazione e la configurazione.

        Args:
            metrics: Istanza della classe Metrics o MetricsWithPriority contenente i dati della simulazione.
            config: Oggetto di configurazione con parametri come CONFIDENCE_LEVEL, WARMUP_METHOD, ecc.
        """
        self.metrics = metrics
        self.config = config
        self.mode = getattr(config, "SIMULATION_MODE", SimulationMode.STEADY_STATE)
        self.alpha = 1 - getattr(config, "CONFIDENCE_LEVEL", 0.95)
        self.rel_precision = getattr(config, "REL_PRECISION", None)
        self.abs_precision = getattr(config, "ABS_PRECISION", None)
        self.warmup_method = getattr(config, "WARMUP_METHOD", "MSER5")
        self.confidence_level = getattr(config, "CONFIDENCE_LEVEL", 0.95)

    def extract_response_times_values(self) -> list[float]:
        """
        Estrae solo i valori dei tempi di risposta (senza timestamp)
        per l'analisi del warm-up o per altre analisi statistiche.

        Returns:
            list[float]: Lista dei valori dei tempi di risposta.
        """
        all_responses = self.metrics.get_all_response_times_with_timestamps()
        return [resp_time for _, resp_time in all_responses]

    def extract_full_response_data(self) -> list[tuple[float, float]]:
        """
        Estrae sia i timestamp che i valori dei tempi di risposta.
        Utile per convertire gli indici di warm-up in durate temporali.

        Returns:
            list[tuple[float, float]]: Lista di tuple (timestamp, tempo_risposta).
        """
        return self.metrics.get_all_response_times_with_timestamps()


    def estimate_warmup(self, values_for_warmup_analysis: list[float], full_data_with_timestamps: list[tuple[float, float]]) -> float:
        """
        Stima la durata del warm-up in SECONDI utilizzando il metodo configurato (Welch o MSER5).

        Args:
            values_for_warmup_analysis (list[float]): Solo i valori (es. tempi di risposta)
                                                      utilizzati direttamente dal metodo di warm-up.
            full_data_with_timestamps (list[tuple[float, float]]): I dati completi (timestamp, valore)
                                                                    per convertire l'indice di warm-up in tempo.

        Returns:
            float: Il tempo di warm-up stimato in secondi.
        """
        if not values_for_warmup_analysis or len(values_for_warmup_analysis) < 2:
            print("  DEBUG(Analyzer): Dati insufficienti per stimare il warm-up. Ritorno 0s.")
            return 0.0

        warmup_index = 0
        if self.warmup_method == "WELCH":
            warmup_index = self._welch(np.array(values_for_warmup_analysis))
            print(f"  DEBUG(Analyzer): Metodo Welch ha stimato un indice di warm-up: {warmup_index}")
        elif self.warmup_method == "MSER5":
            warmup_index = self._mser5(np.array(values_for_warmup_analysis))
            print(f"  DEBUG(Analyzer): Metodo MSER5 ha stimato un indice di warm-up: {warmup_index}")
        else:
            print(f"  DEBUG(Analyzer): Metodo di warm-up '{self.warmup_method}' non riconosciuto. Non viene applicato un warm-up o viene impostato a 0.")
            warmup_index = 0 # Nessun warmup specifico se il metodo non è valido

        estimated_warmup_time = 0.0
        if warmup_index == 0:
            # Se l'indice è 0, il metodo ha indicato che il sistema è già in stato stazionario
            # o non ci sono abbastanza dati per stimare un warm-up.
            estimated_warmup_time = 0.0
            print("  DEBUG(Analyzer): Warm-up index è 0, tempo di warm-up impostato a 0s.")
        elif warmup_index >= len(full_data_with_timestamps):
            # Se l'indice di warm-up è fuori dal range dei dati, potrebbe indicare che il
            # sistema non si è stabilizzato entro la durata della simulazione.
            # Assegniamo un valore conservativo.
            estimated_warmup_time = self.config.STEADY_SIMULATION_TIME * 0.8
            print(f"  WARNING(Analyzer): Warmup_index {warmup_index} fuori range ({len(full_data_with_timestamps)}). "
                  f"Il sistema non sembra stabilizzarsi. Assegno un warmup conservativo di {estimated_warmup_time:.2f}s "
                  f"(80% della durata totale della simulazione).")
        else:
            # Il tempo di warm-up è il timestamp dell'evento che corrisponde all'indice stimato.
            estimated_warmup_time = full_data_with_timestamps[warmup_index][0]

        # Per evitare che il warmup sia quasi l'intera simulazione,
        # lo limitiamo ad un massimo dell'80% della durata totale. Questo garantisce
        # che rimanga una porzione significativa di dati per l'analisi dello stato stazionario.
        if estimated_warmup_time >= self.config.STEADY_SIMULATION_TIME * 0.95:
            original_estimated_time = estimated_warmup_time
            estimated_warmup_time = self.config.STEADY_SIMULATION_TIME * 0.8
            print(f"  WARNING(Analyzer): Tempo di warm-up stimato ({original_estimated_time:.2f}s) troppo lungo "
                  f"(> 95% della simulazione). Ridotto conservativamente a {estimated_warmup_time:.2f}s "
                  f"(80% della durata totale della simulazione) per lasciare dati per lo steady-state.")

        print(f"  DEBUG(Analyzer): Tempo di warm-up finale stimato: {estimated_warmup_time:.2f}s.")
        return estimated_warmup_time

    def _welch(self, x: np.ndarray) -> int:
        """
        Implementazione euristica del metodo di Welch per la stima del warm-up.
        Cerca il punto in cui la media mobile della serie si stabilizza.

        Args:
            x (np.ndarray): Array dei valori da analizzare (e.g., tempi di risposta).

        Returns:
            int: L'indice del punto di inizio dello stato stazionario.
        """
        n = len(x)
        if n < 50:
            print("  DEBUG(Analyzer): Dati insufficienti (n < 50) per il metodo Welch, ritorno 0.")
            return 0

        # La dimensione della finestra di smoothing è un parametro critico.
        # Una finestra euristica comune è circa il 10% della lunghezza della serie,
        # con un minimo per garantire un smoothing significativo.
        window = max(5, int(min(50, n) // 10))
        if window < 1 or n < window * 2:
            print(f"  DEBUG(Analyzer): Impossibile creare finestra valida per Welch con n={n}, window={window}. Ritorno 0.")
            return 0

        # Calcola la media mobile. 'mode='valid'' assicura che il risultato
        # contenga solo punti dove la finestra è completamente all'interno dei dati.
        smoothed = np.convolve(x, np.ones(window)/window, mode='valid')
        if len(smoothed) < 2:
            print("  DEBUG(Analyzer): Dati smoothed insufficienti per il metodo Welch per calcolare le differenze, ritorno 0.")
            return 0

        # Calcola la deviazione standard della serie smoothed.
        std_smoothed = np.std(smoothed)
        if std_smoothed == 0:
            # Se la serie smoothed è costante, non c'è transitorio evidente.
            print("  DEBUG(Analyzer): Serie smoothed costante, nessun transitorio rilevato con Welch, ritorno 0.")
            return 0

        # Le differenze assolute tra elementi consecutivi della serie smoothed.
        diffs = np.abs(np.diff(smoothed))


        threshold_welch = std_smoothed / 10.0


        valid_cutoffs = np.where(diffs < threshold_welch)[0]
        cutoff = valid_cutoffs[0] if len(valid_cutoffs) > 0 else 0

        # Consideriamo che un warmup di 0 indichi che non è stato trovato un punto di cutoff significativo.
        return cutoff if cutoff > 0 else 0

    def _mser5(self, x: np.ndarray) -> int:
        """
        Implementazione euristica del metodo MSER-5 (Mean Square Error for the Sample Mean)
        per la stima del warm-up. Cerca di minimizzare la varianza della media campionaria
        della porzione di dati considerata in stato stazionario.

        Args:
            x (np.ndarray): Array dei valori da analizzare.

        Returns:
            int: L'indice del punto di inizio dello stato stazionario.
        """
        n = len(x)
        if n < 200: # MSER5 generalmente ha bisogno di più dati per essere efficace
            print("  DEBUG(Analyzer): Dati insufficienti (n < 200) per il metodo MSER5, ritorno 0.")
            return 0

        # Dimensione della finestra per la media mobile. È un parametro euristico,
        # spesso circa n/20.
        window = max(5, n // 20)
        if window < 1 or n < window * 2:
            print(f"  DEBUG(Analyzer): Impossibile creare finestra valida per MSER5 con n={n}, window={window}. Ritorno 0.")
            return 0

        best_t0, best_var = 0, float("inf")


        for t0 in range(0, n - window): # Garantisce almeno `window` elementi rimanenti
            resid = x[t0:]

            if len(resid) < window: # Assicurati che resid abbia almeno la lunghezza della finestra
                break

            # Calcola la media mobile sulla porzione 'resid'.
            convolved_resid = np.convolve(resid, np.ones(window)/window, mode="valid")

            if len(convolved_resid) < 2: # Abbiamo bisogno di almeno due punti per calcolare la varianza
                continue

            # Calcola la varianza campionaria delle medie mobili.
            v = np.var(convolved_resid, ddof=1) # ddof=1 per varianza campionaria
            if v < best_var:
                best_var, best_t0 = v, t0

        # Se best_t0 è 0 e best_var è ancora inf, significa che non sono stati trovati dati validi
        if best_var == float("inf"):
            print("  DEBUG(Analyzer): MSER5 non ha trovato una varianza minimizzata, ritorno 0.")
            return 0

        return best_t0


    def steady_state_analysis(self, values: list[float]):
        """

        Esegue l'intera pipeline di analisi Batch Means per una data serie di osservazioni.
        1. Cerca la configurazione ottimale (b, k) usando `compute_batch_size`.
        2. Calcola l'intervallo di confidenza usando `batch_means`.
        3. Esegue test diagnostici aggiuntivi (Ljung-Box).
        4. Controlla il raggiungimento della precisione desiderata.

        Args:
            values (list[float]): La serie di osservazioni a regime (già filtrata dal warm-up).

        Returns:
            dict | None: Dizionario con tutti i risultati dell'analisi (media, CI, b, k, etc.),
                         o None se l'analisi fallisce in uno degli step.
        """
        if not values or len(values) < self.config.BATCH_K: # Richiesta minima per avere dati
            print(f"  DEBUG(Analyzer): Dati insufficienti ({len(values)}) per analisi Batch Means.")
            return None

        # Step 1: Cerca (b, k) usando la logica robusta.
        # Usiamo un target iniziale più aggressivo (più batch) per dare più spazio alla ricerca.
        b, k, rho1 = compute_batch_size(
            data=values,
            k_initial_target=256, # Partiamo da un target alto
            threshold=self.config.BATCH_THRESHOLD,
        )

        if b is None or k is None:
            print("  WARNING(Analyzer): compute_batch_size non ha trovato una configurazione (b, k) valida.")
            return None

        # Step 2: Calcola l'intervallo di confidenza con (b, k) trovati.
        results = batch_means(values, b, k, self.confidence_level)

        if results is None:
            print("  WARNING(Analyzer): batch_means ha restituito None. Impossibile calcolare il CI.")
            return None

        # Step 3: Esegui test diagnostici.
        # Le medie dei batch per il test di Ljung-Box.
        batch_means_for_test = [np.mean(values[i*b:(i+1)*b]) for i in range(k)]

        # Il test richiede h < n. Usiamo min(10, k-1) come lag.
        lags = min(10, k - 1) if k > 1 else 0
        pval = None
        if lags > 0:
            pval = ljung_box_test(batch_means_for_test, h=lags)

        independence_ok = (pval is not None) and (pval > self.alpha)

        # Step 4: Controlla la precisione.
        precision_met = True
        mean = results['mean']
        half_width = results['half_width']

        if self.abs_precision is not None and half_width > self.abs_precision:
            precision_met = False
        if self.rel_precision is not None:
            if mean != 0 and (half_width / abs(mean)) > self.rel_precision:
                precision_met = False
            elif mean == 0 and half_width > 0: # Caso limite
                precision_met = False

        # Aggiorna il dizionario dei risultati con le informazioni aggiuntive
        results.update({
            "ljung_box_pvalue": pval,
            "independence_ok": independence_ok,
            "precision_met": precision_met,
            "rho1": rho1
        })

        return results

    def terminating_analysis(self, replications: list[float], confidence: float = 0.95) -> dict | None:
        """
        Esegue l'analisi statistica per simulazioni terminating replicate.

        Args:
            replications (list[float]): Lista delle medie ottenute da ciascuna replica.
            confidence (float): Livello di confidenza desiderato.

        Returns:
            dict | None: Dizionario con media, CI, semi-ampiezza, livello di confidenza e numero di repliche,
                         o None se ci sono meno di 2 repliche.
        """
        n = len(replications)
        if n < 2:
            print("  DEBUG(Analyzer): Meno di 2 replicazioni per l'analisi terminating, ritorno None.")
            return None

        mean = np.mean(replications)
        s2 = np.var(replications, ddof=1) # Varianza campionaria
        dof = n - 1

        if dof <= 0 or np.isnan(s2):
            half_width = np.nan
            ci_lower, ci_upper = np.nan, np.nan
        else:
            tval = t.ppf((1 + confidence)/2, df=dof)
            half_width = tval * np.sqrt(s2 / n)
            ci_lower = mean - half_width
            ci_upper = mean + half_width

        return {
            "mean": mean,
            "ci": (ci_lower, ci_upper),
            "half_width": half_width,
            "confidence_level": confidence,
            "replications": n
        }


    def plot_confidence_interval(self, results: dict, title: str, output_dir: str, filename: str):
        """
        Genera un grafico a barra di errore per visualizzare l'intervallo di confidenza.

        Args:
            results (dict): Dizionario contenente i risultati dell'analisi CI.
            title (str): Titolo del grafico.
            output_dir (str): Directory di output per il salvataggio del grafico.
            filename (str): Nome del file del grafico (es. "response_time_ci.png").
        """
        if not results or np.isnan(results.get("mean", np.nan)) or np.isnan(results.get("half_width", np.nan)):
            print(f"  WARNING(Analyzer): Impossibile generare plot per '{title}', risultati non validi (NaN o mancanti).")
            return

        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained") # Added layout="constrained"
        ax.errorbar(x=[0], y=[results["mean"]], yerr=results["half_width"], fmt='o', color='b',
                    capsize=10, markersize=8, elinewidth=3, label=f'CI {results["confidence_level"]:.0%}')
        ax.set_title(title, pad=20)
        ax.set_ylabel('Valore Medio')
        ax.set_xticks([])
        ax.grid(True, axis='y', alpha=0.7)
        ax.set_xlim([-1, 1])

        # Prepare CI text, handling None p-value
        ljung_box_pvalue_str = f"{results['ljung_box_pvalue']:.4f}" if results.get('ljung_box_pvalue') is not None else "N/A"
        ci_text = (
            f"Media stimata: {results['mean']:.4f}\n"
            f"CI al {results['confidence_level']:.0%}: [{results['ci'][0]:.4f}, {results['ci'][1]:.4f}]\n"
            f"Semi-ampiezza CI: {results['half_width']:.4f}\n"
            f"Batch size: {results.get('batch_size','N/A')}, "
            f"Numero Batch: {results.get('num_batches','N/A')}\n"
            f"Ljung–Box p-value: {ljung_box_pvalue_str}\n"
            f"Indipendenza Batch: {'OK' if results.get('independence_ok', False) else 'NO'}\n"
            f"Precisione Desiderata: {'Raggiunta' if results.get('precision_met', False) else 'NON Raggiunta'}"
        )
        ax.text(0.02, 0.98, ci_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.6, edgecolor='gray'))

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)


    def print_ci_results(self, results: dict | None, metric_name: str):
        """
        Stampa i risultati dell'analisi dell'intervallo di confidenza.
        """
        if not results:
            print(f"Nessun risultato valido per '{metric_name}'")
            return
        print(f"\n--- Risultati Analisi Batch Means per '{metric_name}' ---")
        print(f"  - Media: {results['mean']:.6f}")
        print(f"  - CI {results['confidence_level']:.0%}: ({results['ci'][0]:.6f}, {results['ci'][1]:.6f})")
        print(f"  - Semi-ampiezza CI (Half-Width): {results['half_width']:.6f}")
        print(f"  - Batch size: {results.get('batch_size','N/A')}, #Batch: {results.get('num_batches','N/A')}")
        if 'ljung_box_pvalue' in results:
            p = results['ljung_box_pvalue']

            if p is None:
                print(f"  - Ljung–Box p-value: N/A (Dati insufficienti per il test)")
            else:
                print(f"  - Ljung–Box p-value: {p:.4f} ({'OK' if results['independence_ok'] else '-> ATTENZIONE autocorrelazione residua'})")
        if 'precision_met' in results:
            print(f"  - Precisione desiderata: {'RAGGIUNTA' if results['precision_met'] else 'NON RAGGIUNTA'}")
        print(f"----------------------------------------------------")

    def calculate_throughput_ci(self, completion_timestamps: list[float], warmup_period: float) -> dict | None:
        """
        # REVISED (ADAPTIVE LOGIC): This method now uses an adaptive approach to determine the
        # number of throughput samples to generate, making the choice more robust than a fixed magic number.
        # The logic is now based on the density of the original completion events. It aims to create
        # one throughput sample for every `EVENTS_PER_SAMPLE` original events, ensuring that the
        # granularity of the analysis adapts to the simulation's output. The number of samples is
        # bounded (MIN/MAX_SAMPLES) to guarantee both statistical validity and computational efficiency.

        Args:
            completion_timestamps (list[float]): Timestamp degli eventi completati (già ordinati).
            warmup_period (float): Tempo di warm-up da escludere.

        Returns:
            dict | None: Dizionario con mean, ci, half_width, ecc., o None se i dati sono insufficienti.
        """
        steady_state_timestamps = [t for t in completion_timestamps if t >= warmup_period]
        num_steady_events = len(steady_state_timestamps)

        if num_steady_events < 2:
            print("  WARNING(Analyzer): Meno di 2 timestamp in steady-state per calcolare il throughput. Ritorno None.")
            return None

        ss_start_time = steady_state_timestamps[0]
        ss_end_time = steady_state_timestamps[-1]
        total_ss_duration = ss_end_time - ss_start_time

        if total_ss_duration <= 0.01: # Richiede un intervallo di tempo minimo.
            print(f"  WARNING(Analyzer): Durata totale dello steady-state ({total_ss_duration:.2f}s) troppo breve per il throughput. Ritorno None.")
            return None


        EVENTS_PER_SAMPLE = 50    # Vogliamo che ogni nostro campione rappresenti circa 50 eventi reali.
        MIN_SAMPLES = 50          # Minimo numero di campioni per un'analisi Batch Means affidabile.
        MAX_SAMPLES = 800         # Massimo per garantire performance computazionali veloci.

        # Calcola il numero di campioni target basato sulla densità degli eventi.
        target_samples = int(num_steady_events / EVENTS_PER_SAMPLE)

        # Applica i limiti (bounding).
        num_temporal_batches = max(MIN_SAMPLES, min(target_samples, MAX_SAMPLES))

        # Se il numero di eventi originali è molto basso, potremmo non raggiungere MIN_SAMPLES.
        # In tal caso, usiamo un numero di campioni inferiore, ma solo se è ancora ragionevole.
        if num_steady_events < MIN_SAMPLES:
            if num_steady_events < 10: # Se ci sono meno di 10 eventi in totale, l'analisi non ha senso.
                print(f"  WARNING(Analyzer): Numero di eventi in steady-state ({num_steady_events}) troppo basso per l'analisi del throughput.")
                return None
            num_temporal_batches = num_steady_events # Usa un campione per ogni evento.

        temporal_batch_duration = total_ss_duration / num_temporal_batches

        throughput_samples = []
        current_event_idx = 0

        for i in range(num_temporal_batches):
            batch_start_time = ss_start_time + i * temporal_batch_duration
            batch_end_time = ss_start_time + (i + 1) * temporal_batch_duration
            events_in_batch = 0

            while current_event_idx < num_steady_events and steady_state_timestamps[current_event_idx] < batch_end_time:
                if steady_state_timestamps[current_event_idx] >= batch_start_time:
                    events_in_batch += 1
                current_event_idx += 1

            throughput_samples.append(events_in_batch / temporal_batch_duration)

        # Applica l'analisi Batch Means standard ai campioni di throughput generati.
        throughput_results = self.steady_state_analysis(throughput_samples)

        if throughput_results:
            throughput_results["total_steady_state_events"] = num_steady_events
            throughput_results["total_steady_state_duration"] = total_ss_duration

        return throughput_results