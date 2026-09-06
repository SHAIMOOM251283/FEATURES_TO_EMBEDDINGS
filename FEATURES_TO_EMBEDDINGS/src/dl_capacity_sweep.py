import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from dl_track import DLTrack

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class CapacitySweep:

    def __init__(self):
        output_dir = "outputs"
        self.output_dir = PROJECT_ROOT / output_dir
        results_name = "dl_sweep_results.json"
        self.results_path = self.output_dir / results_name
        plot_name = "dl_sweep_curve.png"
        self.plot_path = self.output_dir / plot_name

        # Each setting shrinks the model in one of two ways: fewer dimensions
        # per word, or fewer words in the vocabulary. Both reduce parameters.
        self.settings = [
            {"name": "tiny",      "embed_dim": 8,  "min_word_count": 10, "dropout": 0.5},
            {"name": "small",     "embed_dim": 16, "min_word_count": 8,  "dropout": 0.5},
            {"name": "medium",    "embed_dim": 24, "min_word_count": 5,  "dropout": 0.45},
            {"name": "large",     "embed_dim": 32, "min_word_count": 4,  "dropout": 0.45},
            {"name": "wide",      "embed_dim": 48, "min_word_count": 3,  "dropout": 0.4},
            {"name": "original",  "embed_dim": 64, "min_word_count": 2,  "dropout": 0.4},
        ]

        self.rows = []
        self.results = {}

    def build_track(self, setting):
        """Create a DLTrack with one capacity setting applied."""
        track = DLTrack()
        track.embed_dim = setting["embed_dim"]
        track.hidden_dim = setting["embed_dim"]
        track.min_word_count = setting["min_word_count"]
        track.dropout = setting["dropout"]
        # Smaller models take more epochs to converge, so the budget must be
        # generous enough that every configuration stops on its own. A fixed
        # low ceiling would measure training budget instead of capacity.
        track.max_epochs = 600
        track.patience = 25
        return track

    def run_one(self, setting):
        """Train one configuration and score it on validation."""
        print(f"\n--- {setting['name']} "
              f"(embed {setting['embed_dim']}, min count {setting['min_word_count']}) ---")

        track = self.build_track(setting)
        track.set_seeds()
        track.load()
        track.build_vocabulary()
        track.build_tensors()
        track.build_model()
        track.train()

        _, val_accuracy, val_predictions = track.score("val")
        val_labels = track.tensors["val"][1].numpy()
        val_macro_f1 = f1_score(val_labels, val_predictions,
                                average="macro", zero_division=0)

        row = {
            "name": setting["name"],
            "embed_dim": setting["embed_dim"],
            "min_word_count": setting["min_word_count"],
            "vocabulary": len(track.vocabulary),
            "parameters": sum(p.numel() for p in track.model.parameters()),
            "epochs": len(track.history["train_loss"]),
            "final_train_loss": round(track.history["train_loss"][-1], 4),
            "best_val_loss": round(min(track.history["val_loss"]), 4),
            "val_accuracy": round(float(val_accuracy), 4),
            "val_macro_f1": round(float(val_macro_f1), 4),
            "train_seconds": round(track.train_seconds, 2),
        }
        self.rows.append(row)
        return track, row

    def sweep(self):
        """Train every configuration, keeping the best by validation accuracy."""
        best_track = None
        best_row = None

        for setting in self.settings:
            track, row = self.run_one(setting)
            if best_row is None or row["val_accuracy"] > best_row["val_accuracy"]:
                best_track, best_row = track, row

        self.best_track = best_track
        self.best_row = best_row
        return self.rows

    def score_winner(self):
        """Score only the validation winner on the held-out test set."""
        track = self.best_track
        _, accuracy, predictions = track.score("test")
        labels = track.tensors["test"][1].numpy()
        macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

        self.results = {
            "settings": self.rows,
            "selected": self.best_row["name"],
            "selected_on": "validation accuracy",
            "test": {
                "accuracy": round(float(accuracy), 4),
                "macro_f1": round(float(macro_f1), 4),
                "parameters": self.best_row["parameters"],
                "vocabulary": self.best_row["vocabulary"],
                "train_seconds": self.best_row["train_seconds"],
                "per_class": classification_report(
                    labels, predictions,
                    target_names=track.label_names,
                    output_dict=True,
                    zero_division=0,
                ),
            },
        }

        print(f"\nselected '{self.best_row['name']}' on validation accuracy "
              f"({self.best_row['val_accuracy']:.4f})")
        print(f"test accuracy  {accuracy:.4f}")
        print(f"test macro f1  {macro_f1:.4f}")
        return self.results

    def report(self):
        """Print every configuration, so the whole curve is visible."""
        table = pd.DataFrame(self.rows)[
            ["name", "embed_dim", "vocabulary", "parameters",
             "epochs", "final_train_loss", "best_val_loss", "val_accuracy"]
        ]
        print("\nall configurations:")
        print(table.to_string(index=False))

    def plot(self):
        """Chart validation accuracy and the train/validation loss gap."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(self.rows).sort_values("parameters")

        figure, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

        left.plot(frame["parameters"], frame["val_accuracy"],
                  marker="o", color="#2a6f97")
        left.set_xscale("log")
        left.set_xlabel("parameters (log scale)")
        left.set_ylabel("validation accuracy")
        left.set_title("Does shrinking the model help?")
        for _, row in frame.iterrows():
            left.annotate(row["name"], (row["parameters"], row["val_accuracy"]),
                          textcoords="offset points", xytext=(0, 8), fontsize=8)

        gap = frame["best_val_loss"] - frame["final_train_loss"]
        right.plot(frame["parameters"], gap, marker="o", color="#d4a017")
        right.set_xscale("log")
        right.set_xlabel("parameters (log scale)")
        right.set_ylabel("validation loss minus train loss")
        right.set_title("How much did it memorise?")

        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=150)
        plt.close(figure)
        print(f"\nsaved plot to {self.plot_path.name}")

    def save(self):
        """Write every configuration and the selected result to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"saved results to {self.results_path.name}")

    def run(self):
        """Sweep every capacity, select on validation, score once on test."""
        self.sweep()
        self.report()
        self.score_winner()
        self.plot()
        self.save()
        return self.results


if __name__ == "__main__":
    CapacitySweep().run()