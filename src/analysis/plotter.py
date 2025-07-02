import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle
import numpy as np

matplotlib.use('QT5Agg')  # For compatibility with headless environments

class CSVPlotter:
    def __init__(self, file1, label1, file2=None, label2=None):
        """
        file1, file2: percorsi ai file .xlsx generati da data_report.py
        label1, label2: etichette leggibili per i due scenari (es. 'Con Priorità', 'Senza Priorità')
        """
        self.file1 = file1
        self.file2 = file2
        self.label1 = label1
        self.label2 = label2

        self.df1 = pd.read_excel(file1, sheet_name=None)
        self.df2 = pd.read_excel(file2, sheet_name=None) if file2 else None

        # Configurazione tema coerente
        self._setup_theme()

    def _setup_theme(self):
        """Configura un tema coerente per tutti i grafici"""
        # Palette di colori coerente
        self.colors = {
            'primary': '#2E86AB',      # Blu principale
            'secondary': '#A23B72',    # Magenta/Rosa
            'tertiary': '#F18F01',     # Arancione
            'quaternary': '#C73E1D',   # Rosso
            'background': '#F8F9FA',   # Grigio molto chiaro
            'grid': '#E9ECEF',         # Grigio per le griglie
            'text': '#212529'          # Grigio scuro per il testo
        }

        # Configurazione matplotlib
        plt.rcParams.update({
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': 'white',
            'axes.edgecolor': self.colors['grid'],
            'axes.linewidth': 1.2,
            'axes.labelcolor': self.colors['text'],
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'legend.title_fontsize': 12,
            'grid.color': self.colors['grid'],
            'grid.linewidth': 0.8,
            'grid.alpha': 0.7,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        })

        # Configurazione seaborn
        sns.set_palette([self.colors['primary'], self.colors['secondary'],
                         self.colors['tertiary'], self.colors['quaternary']])

    def _apply_consistent_styling(self, ax, title, xlabel, ylabel):
        """Applica styling coerente a tutti i grafici"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20, color=self.colors['text'])
        ax.set_xlabel(xlabel, fontsize=12, fontweight='medium', color=self.colors['text'])
        ax.set_ylabel(ylabel, fontsize=12, fontweight='medium', color=self.colors['text'])

        # Griglia personalizzata
        ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.7, color=self.colors['grid'])
        ax.set_axisbelow(True)

        # Bordi degli assi
        for spine in ax.spines.values():
            spine.set_color(self.colors['grid'])
            spine.set_linewidth(1.2)

    def _load(self, sheet_name):
        df_a = self.df1[sheet_name].copy()
        if self.df2:
            df_b = self.df2[sheet_name].copy()
            return df_a, df_b
        return df_a, None

    def plot_avg_response_times(self, by='type', figsize=(10, 6)):
        """
        Confronta tempo medio di risposta per tipo o priorità.
        """
        df1, df2 = self._load("aggregated")
        metric_filter = {
            'type': 'Avg Response Time (type)',
            'priority': 'Avg Response Time'
        }[by]

        data = []
        scenarios = []

        for df, label in [(df1, self.label1), (df2, self.label2) if df2 is not None else (None, None)]:
            if df is None: continue
            scenarios.append(label)
            for _, row in df.iterrows():
                if pd.isna(row['Value']) or metric_filter not in row['Metric']:
                    continue
                group = row['Metric'].split(' - ')[0]
                data.append({'Group': group, 'Tempo Medio': row['Value'], 'Scenario': label})

        plot_df = pd.DataFrame(data)

        # Creazione figura con dimensioni coerenti
        fig, ax = plt.subplots(figsize=figsize)

        # Determina la palette in base al numero di scenari
        if len(scenarios) == 1:
            palette = [self.colors['primary']]
        else:
            palette = [self.colors['primary'], self.colors['secondary']]

        # Barplot con colori coerenti
        bars = sns.barplot(data=plot_df, x='Group', y='Tempo Medio', hue='Scenario', ax=ax,
                           palette=palette)

        # Styling coerente
        self._apply_consistent_styling(ax,
                                       f"Tempo Medio di Risposta per {by.capitalize()}",
                                       by.capitalize(),
                                       "Tempo (s)")

        # Legenda migliorata
        legend = ax.legend(title="Scenario", frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        # Annotazioni sui valori delle barre
        for container in bars.containers:
            bars.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

        plt.tight_layout()
        makesure_dir_exists("plots")
        plt.savefig(os.path.join("plots", f"avg_response_times_{by}.png"),
                    dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        plt.show()

    def plot_system_evolution(self, figsize=(14, 8)):
        """
        Mostra l'evoluzione di pod e coda nel tempo per il primo file.
        """
        df = self.df1["system_snapshots"]
        fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot pod count - linea continua senza marcatori
        axs[0].plot(df['timestamp'], df['pod_count'],
                    label='Numero Pod', color=self.colors['primary'], linewidth=2.5)
        axs[0].fill_between(df['timestamp'], df['pod_count'], alpha=0.3, color=self.colors['primary'])

        # Plot queue length - linea continua senza marcatori
        axs[1].plot(df['timestamp'], df['queue_length'],
                    label='Lunghezza Coda', color=self.colors['secondary'], linewidth=2.5)
        axs[1].fill_between(df['timestamp'], df['queue_length'], alpha=0.3, color=self.colors['secondary'])

        # Styling per ogni subplot
        self._apply_consistent_styling(axs[0], "", "", "Numero Pod")
        self._apply_consistent_styling(axs[1], "", "Timestamp", "Lunghezza Coda")

        # Titolo principale
        fig.suptitle(f"Evoluzione del Sistema - {self.label1}",
                     fontsize=16, fontweight='bold', y=0.98, color=self.colors['text'])

        # Legende
        for ax in axs:
            legend = ax.legend(frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)

        # Rotazione etichette x
        plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        makesure_dir_exists("plots")
        plt.savefig(os.path.join("plots", "system_evolution.png"),
                    dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        plt.show()

    def plot_response_histogram(self, figsize=(12, 7), bins=50):
        """
        Istogramma dei tempi di risposta grezzi.
        """
        df1 = self.df1['raw_response'].copy()
        df1['Scenario'] = self.label1

        if self.df2:
            df2 = self.df2['raw_response'].copy()
            df2['Scenario'] = self.label2
            df = pd.concat([df1, df2])
        else:
            df = df1

        fig, ax = plt.subplots(figsize=figsize)

        # Determina la palette in base al numero di scenari
        scenarios = df['Scenario'].unique()
        if len(scenarios) == 1:
            palette = [self.colors['primary']]
        else:
            palette = [self.colors['primary'], self.colors['secondary']]

        # Istogramma con trasparenza e colori coerenti
        sns.histplot(data=df, x='response_time', hue='Scenario',
                     element="step", stat="density", common_norm=False,
                     bins=bins, alpha=0.7, ax=ax, palette=palette)

        # Linee delle medie - linee continue
        for i, scenario in enumerate(scenarios):
            mean_val = df[df['Scenario'] == scenario]['response_time'].mean()
            color = palette[i] if i < len(palette) else self.colors['primary']
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Media {scenario}: {mean_val:.2f}s')

        self._apply_consistent_styling(ax,
                                       "Distribuzione Tempi di Risposta",
                                       "Tempo di Risposta (s)",
                                       "Densità")

        # Legenda migliorata
        legend = ax.legend(frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        plt.tight_layout()
        makesure_dir_exists("plots")
        plt.savefig(os.path.join("plots", "response_histogram.png"),
                    dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        plt.show()

    def plot_wait_histogram(self, figsize=(12, 7), bins=50, save_to=None):
        """
        Istogramma dei tempi di attesa grezzi.
        """
        df1 = self.df1['raw_wait'].copy()
        df1['Scenario'] = self.label1

        if self.df2:
            df2 = self.df2['raw_wait'].copy()
            df2['Scenario'] = self.label2
            df = pd.concat([df1, df2])
        else:
            df = df1

        fig, ax = plt.subplots(figsize=figsize)

        # Determina la palette in base al numero di scenari
        scenarios = df['Scenario'].unique()
        if len(scenarios) == 1:
            palette = [self.colors['primary']]
        else:
            palette = [self.colors['primary'], self.colors['secondary']]

        # Istogramma con trasparenza e colori coerenti
        sns.histplot(data=df, x='wait_time', hue='Scenario',
                     element="step", stat="density", common_norm=False,
                     bins=bins, alpha=0.7, ax=ax, palette=palette)

        # Linee delle medie - linee continue
        for i, scenario in enumerate(scenarios):
            mean_val = df[df['Scenario'] == scenario]['wait_time'].mean()
            color = palette[i] if i < len(palette) else self.colors['primary']
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.8,
                       label=f'Media {scenario}: {mean_val:.2f}s')

        self._apply_consistent_styling(ax,
                                       "Distribuzione Tempi di Attesa",
                                       "Tempo di Attesa (s)",
                                       "Densità")

        # Legenda migliorata
        legend = ax.legend(frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        plt.tight_layout()
        save_path = save_to or os.path.join("plots", "wait_histogram.png")
        makesure_dir_exists("plots")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        plt.show()

    def plot_comparison_summary(self, figsize=(16, 10)):
        """
        Crea un dashboard di confronto completo con metriche multiple
        """
        if not self.df2:
            print("Confronto richiede due scenari")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Tempi medi di risposta
        df1, df2 = self._load("aggregated")
        data = []
        for df, label in [(df1, self.label1), (df2, self.label2)]:
            for _, row in df.iterrows():
                if pd.isna(row['Value']) or 'Avg Response Time' not in row['Metric']:
                    continue
                group = row['Metric'].split(' - ')[0] if ' - ' in row['Metric'] else 'Overall'
                data.append({'Group': group, 'Tempo': row['Value'], 'Scenario': label})

        plot_df = pd.DataFrame(data)
        sns.barplot(data=plot_df, x='Group', y='Tempo', hue='Scenario', ax=ax1,
                    palette=[self.colors['primary'], self.colors['secondary']])
        self._apply_consistent_styling(ax1, "Tempi Medi di Risposta", "Categoria", "Tempo (s)")

        # 2. Evoluzione pod (solo primo scenario) - linea continua
        df_sys = self.df1["system_snapshots"]
        ax2.plot(df_sys['timestamp'], df_sys['pod_count'],
                 color=self.colors['primary'], linewidth=2.5)
        self._apply_consistent_styling(ax2, "Evoluzione Pod", "Tempo", "Numero Pod")

        # 3. Distribuzione tempi risposta
        df1_resp = self.df1['raw_response'].copy()
        df2_resp = self.df2['raw_response'].copy()
        ax3.hist(df1_resp['response_time'], bins=30, alpha=0.7,
                 label=self.label1, color=self.colors['primary'], density=True)
        ax3.hist(df2_resp['response_time'], bins=30, alpha=0.7,
                 label=self.label2, color=self.colors['secondary'], density=True)
        self._apply_consistent_styling(ax3, "Distribuzione Tempi Risposta", "Tempo (s)", "Densità")

        # 4. Metriche comparative
        metrics = ['Throughput', 'Avg Wait Time', 'Max Queue Length']
        values1, values2 = [], []

        for metric in metrics:
            val1 = df1[df1['Metric'].str.contains(metric, na=False)]['Value'].iloc[0] if len(df1[df1['Metric'].str.contains(metric, na=False)]) > 0 else 0
            val2 = df2[df2['Metric'].str.contains(metric, na=False)]['Value'].iloc[0] if len(df2[df2['Metric'].str.contains(metric, na=False)]) > 0 else 0
            values1.append(val1)
            values2.append(val2)

        x = np.arange(len(metrics))
        width = 0.35
        ax4.bar(x - width/2, values1, width, label=self.label1, color=self.colors['primary'])
        ax4.bar(x + width/2, values2, width, label=self.label2, color=self.colors['secondary'])
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics, rotation=45, ha='right')
        self._apply_consistent_styling(ax4, "Confronto Metriche", "Metrica", "Valore")

        # Legende per tutti i subplot
        for ax in [ax1, ax3, ax4]:
            legend = ax.legend(frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.9)

        fig.suptitle("Dashboard Confronto Scenari", fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        makesure_dir_exists("plots")
        plt.savefig(os.path.join("plots", "comparison_dashboard.png"),
                    dpi=300, bbox_inches='tight', facecolor=self.colors['background'])
        plt.show()


def makesure_dir_exists(directory):
    """
    Assicura che la directory esista, altrimenti la crea.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
