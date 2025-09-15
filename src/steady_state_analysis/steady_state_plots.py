import matplotlib.pyplot as plt
import numpy as np
import os

class CumulativeResponsePlotter:
    """
    Classe per generare grafici dei tempi di risposta cumulativi sia per numero di batch sia per numero totale di jobs.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _plot_cumulative(self,
                         batch_mean_values: list[float],
                         overall_mean: float,
                         x_values: np.ndarray,
                         x_label: str,
                         y_label: str,
                         title: str,
                         filename: str,
                         label_values: str):
        """
        Genera un grafico dei valori cumulativi (medie cumulative dei batch).
        """

        if not batch_mean_values or x_values is None or len(x_values) != len(batch_mean_values):
            fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
            ax.text(
                0.5, 0.5,
                "Dati insufficienti o parametri batch non validi.\nImpossibile generare il grafico.",
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, color='red'
            )
            ax.set_title(title, fontsize=20, weight='bold')
            save_path = os.path.join(self.output_dir, filename + '.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Grafico non generato a causa di dati insufficienti o parametri non validi: {filename}")
            return

        # Medie cumulative
        cumulative_mean_values = np.cumsum(batch_mean_values) / np.arange(1, len(batch_mean_values) + 1)

        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        ax.plot(x_values, cumulative_mean_values, label=label_values, linestyle='-', linewidth=2, color='blue')

        # Linea media globale
        if overall_mean is not None:
            ax.axhline(y=overall_mean, color='yellow', linestyle='--', linewidth=2,
                       label=f'Media Globale: {overall_mean:.4f}s')
        ax.set_yscale('log')

        ax.set_title(title, fontsize=20, weight='bold')
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.grid(True, which='both', linestyle=':', alpha=0.7)
        ax.legend(loc='lower right', fontsize=14, shadow=True, fancybox=True)
        ax.tick_params(axis='both', which='major', labelsize=14)

        save_path = os.path.join(self.output_dir, filename + '.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafico cumulativo salvato in: {save_path}")

    def plot_by_batch(self, batch_mean_values: list[float], overall_mean: float, system_label: str):
        """
        Genera grafico del tempo di risposta cumulativo in base al numero di batch.
        """
        title = f"Tempo di Risposta Medio Cumulativo per Batch ({system_label})"
        filename = f"cumulative_response_per_batch_{system_label.lower().replace(' ', '_')}"
        x_label = "Numero di Batch"
        y_label = "Tempo di Risposta Medio Cumulativo (s)"
        label_values = "Media Cumulativa dei Batch"

        x_values = np.arange(1, len(batch_mean_values) + 1)  # Batch indices
        self._plot_cumulative(batch_mean_values, overall_mean, x_values, x_label, y_label, title, filename, label_values)

    def plot_by_jobs(self, batch_mean_values: list[float], batch_size: int, overall_mean: float, system_label: str):
        """
        Genera grafico del tempo di risposta cumulativo in base al numero totale di jobs (k*b).
        """
        title = f"Tempo di Risposta Medio Cumulativo per Jobs ({system_label})"
        filename = f"cumulative_response_per_jobs_{system_label.lower().replace(' ', '_')}"
        x_label = "Numero Totale di Jobs (k * batch_size)"
        y_label = "Tempo di Risposta Medio Cumulativo (s)"
        label_values = "Media Cumulativa dei Batch"

        x_values = np.arange(1, len(batch_mean_values) + 1) * batch_size  # Total jobs
        self._plot_cumulative(batch_mean_values, overall_mean, x_values, x_label, y_label, title, filename, label_values)
