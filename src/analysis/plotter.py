
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')  # Backend per interattività con finestre Qt


class CSVPlotter:
    def __init__(self, output_dir="plots", dpi=300, style="seaborn-v0_8"):
        self.output_dir = output_dir
        self.dpi = dpi
        self.style = style
        plt.style.use(self.style)
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_align_data(self, csv1, csv2, metric):
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)

        df1 = df1[df1["category"] != "TOTAL"]
        df2 = df2[df2["category"] != "TOTAL"]

        all_categories = sorted(set(df1["category"]).union(set(df2["category"])))

        values1 = [
            float(df1[df1["category"] == cat][metric].values[0]) if cat in df1["category"].values else 0.0
            for cat in all_categories
        ]
        values2 = [
            float(df2[df2["category"] == cat][metric].values[0]) if cat in df2["category"].values else 0.0
            for cat in all_categories
        ]

        return all_categories, values1, values2

    def compare_bar(self, csv1, csv2, label1, label2, metric: str, show=True):
        categories, values1, values2 = self._load_and_align_data(csv1, csv2, metric)
        x = range(len(categories))

        plt.figure(figsize=(10, 6))
        plt.bar([i - 0.2 for i in x], values1, width=0.4, label=label1, alpha=0.85)
        plt.bar([i + 0.2 for i in x], values2, width=0.4, label=label2, alpha=0.85)

        plt.xticks(ticks=x, labels=categories, rotation=45)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Confronto {metric.replace('_', ' ').title()} per Categoria")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        path = os.path.join(self.output_dir, f"comparison_bar_{metric}.png")
        plt.savefig(path, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()

    def compare_lines(self, csv1, csv2, label1, label2, metric: str, show=True):
        df1 = pd.read_csv(csv1)
        df2 = pd.read_csv(csv2)

        df1 = df1[df1["category"] != "TOTAL"]
        df2 = df2[df2["category"] != "TOTAL"]

        categories = sorted(set(df1["category"]).union(set(df2["category"])))
        x = range(len(categories))

        y1 = [df1[df1["category"] == cat][metric].values[0] if cat in df1["category"].values else 0 for cat in categories]
        y2 = [df2[df2["category"] == cat][metric].values[0] if cat in df2["category"].values else 0 for cat in categories]

        plt.figure(figsize=(10, 6))
        plt.plot(x, y1, marker='o', label=label1, linewidth=2)
        plt.plot(x, y2, marker='o', label=label2, linewidth=2, linestyle='--')

        plt.xticks(ticks=x, labels=categories, rotation=45)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Andamento {metric.replace('_', ' ').title()} per Categoria")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        path = os.path.join(self.output_dir, f"comparison_line_{metric}.png")
        plt.savefig(path, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()

    def plot_single_metric(self, csv, label, metric: str, kind="bar", show=True):
        df = pd.read_csv(csv)
        df = df[df["category"] != "TOTAL"]

        x = range(len(df))
        y = df[metric].fillna(0).values
        categories = df["category"].tolist()

        plt.figure(figsize=(10, 6))

        if kind == "bar":
            plt.bar(x, y, color='#1f77b4', alpha=0.85, label=label)
        elif kind == "line":
            plt.plot(x, y, marker='o', linestyle='-', color='#ff7f0e', linewidth=2, label=label)

        plt.xticks(ticks=x, labels=categories, rotation=45)
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"{metric.replace('_', ' ').title()} - {label}")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout(rect=[0, 0.05, 1, 1])

        path = os.path.join(self.output_dir, f"{label.lower()}_{metric}.png")
        plt.savefig(path, dpi=self.dpi)
        if show:
            plt.show()
        plt.close()
