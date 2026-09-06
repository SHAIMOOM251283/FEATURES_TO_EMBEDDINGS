import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TfidfTrack:

    def __init__(self):
        data_dir = "data"
        self.data_dir = PROJECT_ROOT / data_dir
        output_dir = "outputs"
        self.output_dir = PROJECT_ROOT / output_dir
        input_name = "books_clean.csv"
        self.input_path = self.data_dir / input_name
        labels_name = "label_map.csv"
        self.labels_path = self.data_dir / labels_name
        results_name = "tfidf_results.json"
        self.results_path = self.output_dir / results_name
        plot_name = "tfidf_top_terms.png"
        self.plot_path = self.output_dir / plot_name

        self.random_state = 42
        self.min_df = 2
        self.max_df = 0.6
        self.ngram_range = (1, 2)
        self.sublinear_tf = True
        self.stop_words = "english"
        self.max_iter = 2000

        self.frame = None
        self.label_names = []
        self.vectorizer = None
        self.model = None
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.results = {}

    def load(self):
        """Read the shared clean dataset and the label mapping."""
        self.frame = pd.read_csv(self.input_path)
        labels = pd.read_csv(self.labels_path).sort_values("label")
        self.label_names = list(labels["category"])
        print(f"loaded {len(self.frame)} rows, {len(self.label_names)} categories")
        return self.frame

    def vectorize(self):
        """Fit the vectoriser on training text only, then transform both splits.

        Fitting on the full dataset would let test vocabulary and document
        frequencies influence the representation, which is the same leak the
        DL track's vocabulary avoids.
        """
        train = self.frame[self.frame["split"] == "train"]
        test = self.frame[self.frame["split"] == "test"]

        self.vectorizer = TfidfVectorizer(
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            sublinear_tf=self.sublinear_tf,
            stop_words=self.stop_words,
        )

        self.train_x = self.vectorizer.fit_transform(train["description"].fillna(""))
        self.test_x = self.vectorizer.transform(test["description"].fillna(""))
        self.train_y = train["label"].to_numpy()
        self.test_y = test["label"].to_numpy()

        print(f"vocabulary: {len(self.vectorizer.vocabulary_)} terms "
              f"(unigrams and bigrams, min_df {self.min_df})")
        print(f"train {self.train_x.shape}, test {self.test_x.shape}")
        return self.train_x

    def train(self):
        """Fit logistic regression on the sparse term matrix."""
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=self.random_state,
        )

        started = time.time()
        self.model.fit(self.train_x, self.train_y)
        self.train_seconds = time.time() - started

        print(f"trained in {self.train_seconds:.3f}s")
        return self.model

    def evaluate(self):
        """Score the model on the shared held-out test set."""
        predictions = self.model.predict(self.test_x)
        accuracy = accuracy_score(self.test_y, predictions)
        macro_f1 = f1_score(self.test_y, predictions, average="macro")

        self.results["tfidf_logistic_regression"] = {
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "train_seconds": round(self.train_seconds, 3),
            "feature_count": len(self.vectorizer.vocabulary_),
            "per_class": classification_report(
                self.test_y, predictions,
                target_names=self.label_names,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\ntfidf_logistic_regression")
        print(f"  accuracy  {accuracy:.4f}")
        print(f"  macro f1  {macro_f1:.4f}")
        return self.results

    def top_terms_for(self, label_index, count=8):
        """Return the terms pushing hardest toward one category."""
        terms = np.array(self.vectorizer.get_feature_names_out())
        weights = self.model.coef_[label_index]
        order = np.argsort(weights)[::-1][:count]
        return list(zip(terms[order], weights[order]))

    def report_top_terms(self):
        """Print the words the model relies on, per category."""
        print("\nstrongest terms per category:")
        for index, name in enumerate(self.label_names):
            terms = [term for term, _ in self.top_terms_for(index, count=6)]
            print(f"  {name:<20} {', '.join(terms)}")

    def plot_top_terms(self):
        """Chart the strongest terms for a few readable categories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        chosen = [name for name in
                  ["Food and Drink", "Historical Fiction", "Romance", "Fantasy"]
                  if name in self.label_names]

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        for axis, name in zip(axes.flat, chosen):
            index = self.label_names.index(name)
            pairs = self.top_terms_for(index, count=8)[::-1]
            terms = [term for term, _ in pairs]
            values = [value for _, value in pairs]
            axis.barh(terms, values, color="#2a6f97")
            axis.set_title(name, fontsize=11)
            axis.tick_params(labelsize=9)

        figure.suptitle("Terms the fixed representation relies on", fontsize=13)
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=150)
        plt.close(figure)
        print(f"\nsaved plot to {self.plot_path.name}")

    def save(self):
        """Write metrics to JSON for the final comparison stage."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"saved results to {self.results_path.name}")

    def run(self):
        """Load, vectorise, train, evaluate, inspect, save."""
        self.load()
        self.vectorize()
        self.train()
        self.evaluate()
        self.report_top_terms()
        self.plot_top_terms()
        self.save()
        return self.results


if __name__ == "__main__":
    TfidfTrack().run()
