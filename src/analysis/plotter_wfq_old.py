# File: analysis/plotter_wfq.py (CON INDENTAZIONE CORRETTA)

import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
from src import config
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
plt.style.use('ggplot')

class PlotterWFQ:
    def __init__(self, metrics_base, metrics_prio, metrics_wfq, config_module):
        self.metrics_base = metrics_base
        self.metrics_prio = metrics_prio
        self.metrics_wfq = metrics_wfq
        self.config = config_module

    def _save_plot(self, output_dir, filename, fig):
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_path = os.path.join(output_dir, filename)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico finale salvato in: {save_path}")

    def generate_final_dashboards(self, output_dir, run_prefix, peak_start, peak_end, base_load, peak_load):

        sim_time = self.config.SIMULATION_TIME
        TIME_WINDOW_STR = '10s'
        TIME_WINDOW_SEC = 10

        # --- FUNZIONI HELPER ---
        def get_time_based_moving_average(history):
            if not history or len(history) < 2: return pd.Series(dtype=np.float64)
            times, values = zip(*sorted(history, key=lambda x: x[0]))
            s = pd.Series(values, index=pd.to_datetime(times, unit='s'))
            return s.resample('1s').mean().rolling(window=TIME_WINDOW_STR, min_periods=1).mean()

        def plot_series(ax, series, **kwargs):
            if series is not None and not series.empty:
                time_in_seconds = (series.index - pd.to_datetime(0, unit='s')).total_seconds()
                ax.plot(time_in_seconds, series.values, **kwargs)
        def _extract_seconds(data):
            ts_list = []
            for ts, _ in data:
                if isinstance(ts, pd.Timestamp):
                    ts_list.append(ts.timestamp())  # converte datetime in epoch seconds
                else:
                    ts_list.append(float(ts))
            return ts_list


        # --- PREPARAZIONE DATI ---
        base_ma = get_time_based_moving_average(self.metrics_base.get_all_response_times_with_timestamps())
        prio_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_prio.response_times_history_by_prio.items()}
        wfq_ma_data = {p: get_time_based_moving_average(hist) for p, hist in self.metrics_wfq.response_times_history_by_prio.items()}

        # --- Grafico 1: Protezione QoS per Priorità HIGH ---
        fig1, ax1 = plt.subplots(figsize=(20, 8)); ax1.set_title(f"Protezione QoS per Priorità HIGH - {run_prefix}", fontsize=18)
        ax1.set_xlabel("Tempo (s)"); ax1.set_ylabel("Tempo Risposta Medio (s)"); ax1.grid(True); ax1.set_ylim(bottom=0, top=20.0)
        plot_series(ax1, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        plot_series(ax1, prio_ma_data.get(config.Priority.HIGH), color='darkred', lw=2, label='Priorità Strette (HIGH)')
        plot_series(ax1, wfq_ma_data.get(config.Priority.HIGH), color='limegreen', lw=2, label='WFQ (HIGH)')
        ax_load1 = ax1.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]
        load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load1.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load1.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax_load1.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig1.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_1_QoS_HIGH.png", fig1)

        # --- Grafico 2: Analisi Starvation per Priorità LOW ---
        fig2, ax2 = plt.subplots(figsize=(20, 8)); ax2.set_title(f"Analisi Starvation per Priorità LOW - {run_prefix}", fontsize=18)
        ax2.set_xlabel("Tempo (s)"); ax2.set_ylabel("Tempo Risposta Medio (s)"); ax2.grid(True); ax2.set_ylim(bottom=0, top=20.0)
        plot_series(ax2, base_ma, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)')
        plot_series(ax2, prio_ma_data.get(config.Priority.LOW), color='darkred', lw=2, label='Priorità Strette (LOW)')
        plot_series(ax2, wfq_ma_data.get(config.Priority.LOW), color='limegreen', lw=2, label='WFQ (LOW)')
        ax_load2 = ax2.twinx(); ax_load2.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load2.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax2.get_legend_handles_labels(); lines2, labels2 = ax_load2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig2.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_2_Starvation_LOW.png", fig2)

        # --- Grafico 3: Performance Interna del WFQ ---
        fig3, ax3 = plt.subplots(figsize=(20, 8)); ax3.set_title(f"Performance Interna del DWFQ - {run_prefix}", fontsize=18)
        ax3.set_xlabel("Tempo (s)"); ax3.set_ylabel("Tempo Risposta Medio (s)"); ax3.grid(True); ax3.set_ylim(bottom=0, top=20.0)
        colors = {config.Priority.HIGH: 'green', config.Priority.MEDIUM: 'orange', config.Priority.LOW: 'purple'}
        for prio in config.Priority:
            plot_series(ax3, wfq_ma_data.get(prio), color=colors.get(prio, 'black'), lw=2.5, label=f'DWFQ - {prio.name}')
        ax_load3 = ax3.twinx(); ax_load3.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico')
        ax_load3.set_ylabel("Carico (req/s)", color='gray'); lines, labels = ax3.get_legend_handles_labels(); lines2, labels2 = ax_load3.get_legend_handles_labels()
        ax3.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12); fig3.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_3_WFQ_Internal_Performance.png", fig3)



        # --- DEBUG TIMESTAMPS G4 ---
        def _ts_to_seconds(ts):
            """Converte un timestamp (Timestamp/np.datetime64/float/int) in secondi (float)."""
            if isinstance(ts, pd.Timestamp):
                return ts.timestamp()
            if isinstance(ts, np.datetime64):
                return pd.Timestamp(ts).timestamp()
            try:
                return float(ts)
            except Exception:
                return np.nan

        def _describe_pairs(name, pairs, sim_time):
            print(f"\n=== DEBUG {name} ===")
            if not pairs:
                print("Vuoto.")
                return [], None, None

            # estrai ts e valori
            ts_raw = [p[0] for p in pairs]
            ts_sec = np.array([_ts_to_seconds(t) for t in ts_raw], dtype=float)

            # stima scala (s/ms/ns) per capire cosa sta succedendo
            finite = np.isfinite(ts_sec)
            if not finite.any():
                print("Tutti i timestamp non numerici/NaN.")
                return [], None, None

            min_sec = float(np.min(ts_sec[finite]))
            max_sec = float(np.max(ts_sec[finite]))
            span = max_sec - min_sec
            mag = max(abs(min_sec), abs(max_sec))
            scale = "ns" if mag > 1e12 else ("ms" if mag > 1e10 else "s")

            print(f"n={len(ts_sec)} | type(ts[0])={type(ts_raw[0])}")
            print(f"min_ts_sec={min_sec:.3f}  max_ts_sec={max_sec:.3f}  span={span:.3f}  (scala stimata: {scale})")

            # mostra qualche esempio
            head_raw = ts_raw[:5]
            tail_raw = ts_raw[-5:]
            head_sec = ts_sec[:5]
            tail_sec = ts_sec[-5:]
            print("prime 5 ts (raw):", head_raw)
            print("prime 5 ts (sec):", head_sec)
            print("ultime 5 ts (raw):", tail_raw)
            print("ultime 5 ts (sec):", tail_sec)

            # copertura rispetto ai bin [0, sim_time] e [t0, t0+sim_time]
            t0 = min_sec
            in_0_sim = int(np.sum((ts_sec >= 0) & (ts_sec <= sim_time)))
            in_t0_window = int(np.sum((ts_sec >= t0) & (ts_sec <= t0 + sim_time)))
            print(f"copertura [0, {sim_time}]: {in_0_sim} eventi")
            print(f"copertura [t0={t0:.3f}, t0+sim_time={t0+sim_time:.3f}]: {in_t0_window} eventi")

            return ts_sec.tolist(), min_sec, max_sec

        # Pairs grezzi per HIGH
        base_high_pairs = []
        for rt, prio in self.config.REQUEST_TYPE_TO_PRIORITY.items():
            if prio == self.config.Priority.HIGH:
                base_high_pairs.extend(self.metrics_base.response_times_history.get(rt, []))

        prio_high_pairs = self.metrics_prio.response_times_history_by_prio.get(config.Priority.HIGH, [])
        wfq_high_pairs  = self.metrics_wfq.response_times_history_by_prio.get(config.Priority.HIGH, [])

        # Stampa diagnostica
        base_ts_abs, base_min, base_max = _describe_pairs("BASE HIGH",  base_high_pairs, sim_time)
        prio_ts_abs, prio_min, prio_max = _describe_pairs("PRIO HIGH",  prio_high_pairs, sim_time)
        wfq_ts_abs,  wfq_min,  wfq_max  = _describe_pairs("WFQ  HIGH",  wfq_high_pairs,  sim_time)

        # (solo stampa) range bin attuali
        print(f"\n=== DEBUG BINS ===")
        print(f"sim_time={sim_time}, THROUGHPUT_WINDOW_SEC={TIME_WINDOW_SEC}")
        print(f"primi bin previsti: 0, {TIME_WINDOW_SEC}, {2*TIME_WINDOW_SEC}, ...")

        fig4, ax4 = plt.subplots(figsize=(20, 8))
        ax4.set_title(f"Throughput (Richieste Servite/sec) per Priorità HIGH - {run_prefix}", fontsize=18)
        ax4.set_xlabel("Tempo (s)"); ax4.set_ylabel("Throughput (req/s)"); ax4.grid(True)
        ax4.set_ylim(bottom=0,top=85.0)

        # Usiamo una finestra in secondi per l'istogramma
        THROUGHPUT_WINDOW_SEC = 10

        high_prio_traffic_share = sum(prob for rt, prob in self.config.TRAFFIC_PROFILE.items() if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.HIGH)
        peak_load_high = peak_load * high_prio_traffic_share


        # Raccolta dati (questa parte era già corretta)
        base_ts_high = []
        for rt, prio in self.config.REQUEST_TYPE_TO_PRIORITY.items():
            if prio == self.config.Priority.HIGH:
                base_ts_high.extend(_extract_seconds(self.metrics_base.response_times_history.get(rt, [])))

        prio_ts_high = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.HIGH, []))
        wfq_ts_high = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.HIGH, []))

        # Creiamo i bin (intervalli di tempo) per l'istogramma
        bins = np.arange(0, sim_time + THROUGHPUT_WINDOW_SEC, THROUGHPUT_WINDOW_SEC)


        # Calcolo e Plotting con il metodo robusto
        if base_ts_high:
            counts, _ = np.histogram(base_ts_high, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax4.plot(bins[:-1], throughput, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO) - Throughput HIGH', drawstyle='steps-post')

        if prio_ts_high:
            counts, _ = np.histogram(prio_ts_high, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            print("DEBUG histogram counts (prio):", counts[:20])
            ax4.plot(bins[:-1], throughput, color='darkred', lw=2, label='Priorità Strette - Throughput HIGH', drawstyle='steps-post')

        if wfq_ts_high:
            counts, _ = np.histogram(wfq_ts_high, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax4.plot(bins[:-1], throughput, color='limegreen', lw=2.5, label='DWFQ - Throughput HIGH', drawstyle='steps-post')

        ax4.axhline(y=peak_load_high, color='gray', linestyle=':', lw=2, alpha=0.9, label=f'Tasso Arrivo Teorico HIGH (~{peak_load_high:.1f} req/s)')

        ax4.legend(loc='upper left', fontsize=12); fig4.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_4_Throughput_HIGH.png", fig4)


        # --- Grafico 5: Throughput MEDIUM ---
        fig5, ax5 = plt.subplots(figsize=(20, 8))
        ax5.set_title(f"Throughput (Richieste Servite/sec) per Priorità MEDIUM - {run_prefix}", fontsize=18)
        ax5.set_xlabel("Tempo (s)"); ax5.set_ylabel("Throughput (req/s)"); ax5.grid(True)
        ax5.set_ylim(bottom=0, top=85.0)

        medium_prio_traffic_share = sum(prob for rt, prob in self.config.TRAFFIC_PROFILE.items()
                                        if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.MEDIUM)
        peak_load_medium = peak_load * medium_prio_traffic_share

        base_ts_medium = []
        for rt, prio in self.config.REQUEST_TYPE_TO_PRIORITY.items():
            if prio == self.config.Priority.MEDIUM:
                base_ts_medium.extend(_extract_seconds(self.metrics_base.response_times_history.get(rt, [])))

        prio_ts_medium = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.MEDIUM, []))
        wfq_ts_medium  = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.MEDIUM, []))

        if base_ts_medium:
            counts, _ = np.histogram(base_ts_medium, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax5.plot(bins[:-1], throughput, color='royalblue', linestyle='--', lw=2.5,
                     label='Baseline (FIFO) - Throughput MEDIUM', drawstyle='steps-post')

        if prio_ts_medium:
            counts, _ = np.histogram(prio_ts_medium, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax5.plot(bins[:-1], throughput, color='darkorange', lw=2,
                     label='Priorità Strette - Throughput MEDIUM', drawstyle='steps-post')

        if wfq_ts_medium:
            counts, _ = np.histogram(wfq_ts_medium, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax5.plot(bins[:-1], throughput, color='limegreen', lw=2.5,
                     label='DWFQ - Throughput MEDIUM', drawstyle='steps-post')

        ax5.axhline(y=peak_load_medium, color='gray', linestyle=':', lw=2, alpha=0.9,
                    label=f'Tasso Arrivo Teorico MEDIUM (~{peak_load_medium:.1f} req/s)')

        ax5.legend(loc='upper left', fontsize=12); fig5.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_5_Throughput_MEDIUM.png", fig5)


        # --- Grafico 6: Throughput LOW ---
        fig6, ax6 = plt.subplots(figsize=(20, 8))
        ax6.set_title(f"Throughput (Richieste Servite/sec) per Priorità LOW - {run_prefix}", fontsize=18)
        ax6.set_xlabel("Tempo (s)"); ax6.set_ylabel("Throughput (req/s)"); ax6.grid(True)
        ax6.set_ylim(bottom=0, top=85.0)

        low_prio_traffic_share = sum(prob for rt, prob in self.config.TRAFFIC_PROFILE.items()
                                     if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == self.config.Priority.LOW)
        peak_load_low = peak_load * low_prio_traffic_share

        base_ts_low = []
        for rt, prio in self.config.REQUEST_TYPE_TO_PRIORITY.items():
            if prio == self.config.Priority.LOW:
                base_ts_low.extend(_extract_seconds(self.metrics_base.response_times_history.get(rt, [])))

        prio_ts_low = _extract_seconds(self.metrics_prio.response_times_history_by_prio.get(config.Priority.LOW, []))
        wfq_ts_low  = _extract_seconds(self.metrics_wfq.response_times_history_by_prio.get(config.Priority.LOW, []))

        if base_ts_low:
            counts, _ = np.histogram(base_ts_low, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax6.plot(bins[:-1], throughput, color='royalblue', linestyle='--', lw=2.5,
                     label='Baseline (FIFO) - Throughput LOW', drawstyle='steps-post')

        if prio_ts_low:
            counts, _ = np.histogram(prio_ts_low, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax6.plot(bins[:-1], throughput, color='darkred', lw=2,
                     label='Priorità Strette - Throughput LOW', drawstyle='steps-post')

        if wfq_ts_low:
            counts, _ = np.histogram(wfq_ts_low, bins=bins)
            throughput = counts / THROUGHPUT_WINDOW_SEC
            ax6.plot(bins[:-1], throughput, color='limegreen', lw=2.5,
                     label='DWFQ - Throughput LOW', drawstyle='steps-post')

        ax6.axhline(y=peak_load_low, color='gray', linestyle=':', lw=2, alpha=0.9,
                    label=f'Tasso Arrivo Teorico LOW (~{peak_load_low:.1f} req/s)')

        ax6.legend(loc='upper left', fontsize=12); fig6.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_6_Throughput_LOW.png", fig6)

        # --- SEZIONE NUOVA: GRAFICI DI TIMEOUT PER PRIORITA' ---

        def calculate_rate_histogram(timestamps):
            if not timestamps: return None, None
            bins = np.arange(0, sim_time + TIME_WINDOW_SEC, TIME_WINDOW_SEC)
            counts, _ = np.histogram(timestamps, bins=bins)
            rate = counts / TIME_WINDOW_SEC
            times = bins[:-1]
            return times, rate

        # --- Grafico 7: Timeout per Priorità HIGH ---
        fig7, ax7 = plt.subplots(figsize=(20, 8))
        ax7.set_title(f"Tasso di Timeout (req/s) per Priorità HIGH - {run_prefix}", fontsize=18)
        ax7.set_xlabel("Tempo (s)"); ax7.set_ylabel("Tasso di Timeout (req/s)"); ax7.grid(True); ax7.set_ylim(bottom=0)

        base_ts_high = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH]
        prio_ts_high = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH]
        wfq_ts_high = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.HIGH]

        t, r = calculate_rate_histogram(base_ts_high); ax7.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(prio_ts_high); ax7.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_high); ax7.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None

        ax_load7 = ax7.twinx(); load_times = [0, peak_start, peak_start, peak_end, peak_end, sim_time]
        load_values = [base_load, base_load, peak_load, peak_load, base_load, base_load]
        ax_load7.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load7.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax7.get_legend_handles_labels(); lines2, labels2 = ax_load7.get_legend_handles_labels()
        ax7.legend(lines + lines2, loc='upper left', fontsize=12); fig7.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_7_Timeout_Rate_HIGH.png", fig7)

        # --- Grafico 8: Timeout per Priorità MEDIUM ---
        fig8, ax8 = plt.subplots(figsize=(20, 8))
        ax8.set_title(f"Tasso di Timeout (req/s) per Priorità MEDIUM - {run_prefix}", fontsize=18)
        ax8.set_xlabel("Tempo (s)"); ax8.set_ylabel("Tasso di Timeout (req/s)"); ax8.grid(True); ax8.set_ylim(bottom=0)

        base_ts_medium = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.MEDIUM]
        prio_ts_medium = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.MEDIUM]
        wfq_ts_medium = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.MEDIUM]

        t, r = calculate_rate_histogram(base_ts_medium); ax8.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(prio_ts_medium); ax8.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_medium); ax8.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None

        ax_load8 = ax8.twinx(); ax_load8.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load8.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax8.get_legend_handles_labels(); lines2, labels2 = ax_load8.get_legend_handles_labels()
        ax8.legend(lines + lines2, loc='upper left', fontsize=12); fig8.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_8_Timeout_Rate_MEDIUM.png", fig8)

        # --- Grafico 9: Timeout per Priorità LOW ---
        fig9, ax9 = plt.subplots(figsize=(20, 8))
        ax9.set_title(f"Tasso di Timeout (req/s) per Priorità LOW - {run_prefix}", fontsize=18)
        ax9.set_xlabel("Tempo (s)"); ax9.set_ylabel("Tasso di Timeout (req/s)"); ax9.grid(True); ax9.set_ylim(bottom=0)

        base_ts_low = [ts for ts, rt in self.metrics_base.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.LOW]
        prio_ts_low = [ts for ts, rt in self.metrics_prio.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.LOW]
        wfq_ts_low = [ts for ts, rt in self.metrics_wfq.timeout_history if self.config.REQUEST_TYPE_TO_PRIORITY.get(rt) == config.Priority.LOW]

        t, r = calculate_rate_histogram(base_ts_low); ax9.plot(t, r, color='royalblue', linestyle='--', lw=2.5, label='Baseline (FIFO)', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(prio_ts_low); ax9.plot(t, r, color='darkred', lw=2, label='Priorità Strette', drawstyle='steps-post') if t is not None else None
        t, r = calculate_rate_histogram(wfq_ts_low); ax9.plot(t, r, color='limegreen', lw=2.5, label='DWFQ', drawstyle='steps-post') if t is not None else None

        ax_load9 = ax9.twinx(); ax_load9.plot(load_times, load_values, color='gray', linestyle=':', lw=2, alpha=0.7, label='Carico'); ax_load9.set_ylabel("Carico (req/s)", color='gray')
        lines, labels = ax9.get_legend_handles_labels(); lines2, labels2 = ax_load9.get_legend_handles_labels()
        ax9.legend(lines + lines2, loc='upper left', fontsize=12); fig9.tight_layout()
        self._save_plot(output_dir, f"{run_prefix}_9_Timeout_Rate_LOW.png", fig9)

