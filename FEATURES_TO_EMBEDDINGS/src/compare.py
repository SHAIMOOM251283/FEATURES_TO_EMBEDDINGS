import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Comparison:

    def __init__(self):
        data_dir = "data"
        self.data_dir = PROJECT_ROOT / data_dir
        output_dir = "outputs"
        self.output_dir = PROJECT_ROOT / output_dir
        labels_name = "label_map.csv"
        self.labels_path = self.data_dir / labels_name

        summary_name = "comparison.csv"
        self.summary_path = self.output_dir / summary_name
        per_class_name = "comparison_per_class.csv"
        self.per_class_path = self.output_dir / per_class_name
        chart_name = "comparison_summary.png"
        self.chart_path = self.output_dir / chart_name
        heatmap_name = "comparison_heatmap.png"
        self.heatmap_path = self.output_dir / heatmap_name

        # Each entry names the file to read, the key inside it, and how the
        # model should appear in the final table.
        self.sources = [
            {"file": "ml_results.json", "key": "logistic_regression",
             "label": "Engineered features\n+ LogReg",
             "input": "22 hand-made numbers", "representation": "fixed, human-chosen"},
            {"file": "ml_results.json", "key": "random_forest",
             "label": "Engineered features\n+ Random Forest",
             "input": "22 hand-made numbers", "representation": "fixed, human-chosen"},
            {"file": "tfidf_results.json", "key": "tfidf_logistic_regression",
             "label": "TF-IDF\n+ LogReg",
             "input": "raw words", "representation": "fixed, computed"},
            {"file": "dl_results.json", "key": "neural_network",
             "label": "Embeddings\n(470K params)",
             "input": "raw words", "representation": "learned"},
        ]

        self.label_names = []
        self.rows = []
        self.per_class = None
        self.baseline = None
        self.sweep = None

    def load_labels(self):
        """Read the category names in label order."""
        labels = pd.read_csv(self.labels_path).sort_values("label")
        self.label_names = list(labels["category"])
        return self.label_names

    def read_json(self, name):
        """Read one results file, or return None when a track has not run."""
        path = self.output_dir / name
        if not path.exists():
            print(f"  missing {name} — run that track first")
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def collect(self):
        """Gather headline metrics from every track that has produced results."""
        for source in self.sources:
            payload = self.read_json(source["file"])
            if payload is None or source["key"] not in payload:
                continue
            entry = payload[source["key"]]
            self.rows.append({
                "model": source["label"].replace("\n", " "),
                "input": source["input"],
                "representation": source["representation"],
                "accuracy": entry["accuracy"],
                "macro_f1": entry["macro_f1"],
                "train_seconds": entry["train_seconds"],
            })

        ml = self.read_json("ml_results.json")
        if ml and "majority_baseline" in ml:
            self.baseline = ml["majority_baseline"]["accuracy"]

        self.sweep = self.read_json("dl_sweep_results.json")
        if self.sweep:
            selected = self.sweep["selected"]
            test = self.sweep["test"]
            self.rows.append({
                "model": f"Embeddings, tuned ({selected})",
                "input": "raw words",
                "representation": "learned",
                "accuracy": test["accuracy"],
                "macro_f1": test["macro_f1"],
                "train_seconds": test["train_seconds"],
            })

        return self.rows

    def build_per_class(self):
        """Assemble a per-category F1 table across every track."""
        columns = {}
        for source in self.sources:
            payload = self.read_json(source["file"])
            if payload is None or source["key"] not in payload:
                continue
            scores = payload[source["key"]]["per_class"]
            columns[source["label"].replace("\n", " ")] = {
                name: scores[name]["f1-score"]
                for name in self.label_names if name in scores
            }

        sweep = self.read_json("dl_sweep_results.json")
        if sweep and "per_class" in sweep.get("test", {}):
            scores = sweep["test"]["per_class"]
            label = f"Embeddings, tuned\n({sweep['selected']})".replace("\n", " ")
            columns[label] = {name: scores[name]["f1-score"]
                              for name in self.label_names if name in scores}

        self.per_class = pd.DataFrame(columns)
        support = None
        ml = self.read_json("ml_results.json")
        if ml:
            scores = ml["logistic_regression"]["per_class"]
            support = pd.Series({name: int(scores[name]["support"])
                                 for name in self.label_names if name in scores})
        if support is not None:
            self.per_class.insert(0, "test books", support)

        return self.per_class

    def report(self):
        """Print the headline table and the per-category breakdown."""
        summary = pd.DataFrame(self.rows).sort_values("macro_f1")

        print("\n=== headline results (101 shared test books) ===")
        if self.baseline is not None:
            print(f"majority baseline accuracy: {self.baseline:.4f}\n")
        print(summary[["model", "input", "representation",
                       "accuracy", "macro_f1", "train_seconds"]].to_string(index=False))

        print("\n=== per-category F1 ===")
        print(self.per_class.round(2).to_string())

        return summary

    def plot_summary(self):
        """Chart accuracy and macro F1 for every model."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.rows).sort_values("macro_f1")

        positions = np.arange(len(frame))
        height = 0.38

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.barh(positions + height / 2, frame["accuracy"], height,
                  label="accuracy", color="#2a6f97")
        axis.barh(positions - height / 2, frame["macro_f1"], height,
                  label="macro F1", color="#d4a017")

        if self.baseline is not None:
            axis.axvline(self.baseline, color="#888888", linestyle="--",
                         linewidth=1.2, label="majority baseline")

        axis.set_yticks(positions)
        axis.set_yticklabels(frame["model"], fontsize=9)
        axis.set_xlabel("score")
        axis.set_title("Same books, same split, four representations")
        axis.legend(loc="lower right", fontsize=9)
        figure.tight_layout()
        figure.savefig(self.chart_path, dpi=150)
        plt.close(figure)
        print(f"\nsaved chart to {self.chart_path.name}")

    def plot_heatmap(self):
        """Show which categories each approach could and could not see."""
        frame = self.per_class.drop(columns=["test books"], errors="ignore")

        figure, axis = plt.subplots(figsize=(9, 6))
        image = axis.imshow(frame.to_numpy(), cmap="YlGnBu",
                            vmin=0.0, vmax=1.0, aspect="auto")

        axis.set_xticks(range(len(frame.columns)))
        axis.set_xticklabels(frame.columns, fontsize=8, rotation=20, ha="right")
        axis.set_yticks(range(len(frame.index)))
        axis.set_yticklabels(frame.index, fontsize=9)

        for row in range(len(frame.index)):
            for column in range(len(frame.columns)):
                value = frame.iat[row, column]
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          fontsize=8,
                          color="white" if value > 0.55 else "#222222")

        axis.set_title("Per-category F1 — where each representation succeeded")
        figure.colorbar(image, ax=axis, shrink=0.8, label="F1")
        figure.tight_layout()
        figure.savefig(self.heatmap_path, dpi=150)
        plt.close(figure)
        print(f"saved heatmap to {self.heatmap_path.name}")

    def save(self):
        """Write both tables to CSV."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.summary_path, index=False)
        self.per_class.round(4).to_csv(self.per_class_path)
        print(f"saved {self.summary_path.name} and {self.per_class_path.name}")

    def run(self):
        """Read every track's results, tabulate, chart, save."""
        self.load_labels()
        self.collect()
        self.build_per_class()
        self.report()
        self.plot_summary()
        self.plot_heatmap()
        self.save()
        return self.rows


if __name__ == "__main__":
    Comparison().run()