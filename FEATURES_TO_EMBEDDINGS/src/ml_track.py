import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class MLTrack:

    def __init__(self):
        data_dir = "data"
        self.data_dir = PROJECT_ROOT / data_dir
        output_dir = "outputs"
        self.output_dir = PROJECT_ROOT / output_dir
        input_name = "books_clean.csv"
        self.input_path = self.data_dir / input_name
        labels_name = "label_map.csv"
        self.labels_path = self.data_dir / labels_name
        results_name = "ml_results.json"
        self.results_path = self.output_dir / results_name
        plot_name = "ml_feature_importance.png"
        self.plot_path = self.output_dir / plot_name

        # Words chosen by hand as plausible category giveaways. Every one of
        # these is a guess a human made before seeing any result.
        self.keyword_flags = {
            "kw_recipe": r"\b(recipe|recipes|cook|cooking|kitchen|chef)\b",
            "kw_love": r"\b(love|heart|romance|kiss|wedding)\b",
            "kw_murder": r"\b(murder|detective|killer|crime|mystery)\b",
            "kw_magic": r"\b(magic|dragon|wizard|kingdom|sword)\b",
            "kw_war": r"\b(war|century|history|historical|empire)\b",
            "kw_child": r"\b(child|children|kid|kids|young|school)\b",
            "kw_comic": r"\b(comic|volume|series|graphic|manga)\b",
        }
        self.random_state = 42
        self.n_estimators = 300

        self.frame = None
        self.label_names = []
        self.feature_names = []
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.models = {}
        self.results = {}

    def load(self):
        """Read the shared clean dataset and the label mapping."""
        self.frame = pd.read_csv(self.input_path)
        labels = pd.read_csv(self.labels_path).sort_values("label")
        self.label_names = list(labels["category"])
        print(f"loaded {len(self.frame)} rows, {len(self.label_names)} categories")
        return self.frame

    def count_pattern(self, text, pattern):
        """Count case-insensitive regex matches in one string."""
        return len(re.findall(pattern, text, flags=re.IGNORECASE))

    def build_features(self):
        """Turn each book into a row of hand-chosen numbers."""
        frame = self.frame
        description = frame["description"].fillna("")
        title = frame["title"].fillna("")

        features = pd.DataFrame(index=frame.index)
        features["desc_words"] = description.str.split().str.len()
        features["desc_chars"] = description.str.len()
        features["desc_sentences"] = description.str.count(r"[.!?]") + 1
        features["avg_word_len"] = features["desc_chars"] / features["desc_words"]
        features["avg_sentence_len"] = features["desc_words"] / features["desc_sentences"]
        features["title_words"] = title.str.split().str.len()
        features["title_chars"] = title.str.len()
        features["digit_count"] = description.str.count(r"\d")
        features["comma_count"] = description.str.count(",")
        features["exclaim_count"] = description.str.count("!")
        features["question_count"] = description.str.count(r"\?")
        features["quote_count"] = description.str.count(r'["\u201c\u201d]')
        features["capital_words"] = description.str.count(r"\b[A-Z][a-z]+")
        features["has_subtitle"] = title.str.contains(":").astype(int)
        features["has_series"] = title.str.contains(r"#\d").astype(int)

        for name, pattern in self.keyword_flags.items():
            features[name] = description.apply(
                lambda text, p=pattern: self.count_pattern(text, p)
            )

        features = features.fillna(0.0)
        self.feature_names = list(features.columns)
        print(f"built {len(self.feature_names)} hand-engineered features")

        self.frame = pd.concat([frame, features], axis=1)
        return features

    def split(self):
        """Separate train and test using the shared split column."""
        train = self.frame[self.frame["split"] == "train"]
        test = self.frame[self.frame["split"] == "test"]

        self.train_x = train[self.feature_names].to_numpy()
        self.train_y = train["label"].to_numpy()
        self.test_x = test[self.feature_names].to_numpy()
        self.test_y = test["label"].to_numpy()

        print(f"train {self.train_x.shape}, test {self.test_x.shape}")
        return self.train_x, self.test_x

    def train_logistic_regression(self):
        """Fit the linear baseline on standardised features."""
        scaler = StandardScaler().fit(self.train_x)
        model = LogisticRegression(
            max_iter=2000,
            random_state=self.random_state,
        )

        started = time.time()
        model.fit(scaler.transform(self.train_x), self.train_y)
        elapsed = time.time() - started

        self.models["logistic_regression"] = (model, scaler)
        self.evaluate("logistic_regression", model, scaler, elapsed)
        return model

    def train_random_forest(self):
        """Fit the tree ensemble — no scaling needed."""
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

        started = time.time()
        model.fit(self.train_x, self.train_y)
        elapsed = time.time() - started

        self.models["random_forest"] = (model, None)
        self.evaluate("random_forest", model, None, elapsed)
        return model

    def evaluate(self, name, model, scaler, train_seconds):
        """Score one model on the held-out test set."""
        test_x = scaler.transform(self.test_x) if scaler else self.test_x
        predictions = model.predict(test_x)

        accuracy = accuracy_score(self.test_y, predictions)
        macro_f1 = f1_score(self.test_y, predictions, average="macro")

        self.results[name] = {
            "accuracy": round(float(accuracy), 4),
            "macro_f1": round(float(macro_f1), 4),
            "train_seconds": round(train_seconds, 3),
            "feature_count": len(self.feature_names),
            "per_class": classification_report(
                self.test_y, predictions,
                target_names=self.label_names,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\n{name}")
        print(f"  accuracy  {accuracy:.4f}")
        print(f"  macro f1  {macro_f1:.4f}")
        print(f"  trained   {train_seconds:.3f}s")
        return self.results[name]

    def plot_feature_importance(self):
        """Show which hand-made features the forest actually leaned on."""
        model, _ = self.models["random_forest"]
        ranked = (pd.Series(model.feature_importances_, index=self.feature_names)
                  .sort_values(ascending=True))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.barh(ranked.index, ranked.values, color="#2a6f97")
        axis.set_xlabel("importance")
        axis.set_title("Which hand-engineered features mattered")
        figure.tight_layout()
        figure.savefig(self.plot_path, dpi=150)
        plt.close(figure)

        print(f"\nsaved plot to {self.plot_path.name}")
        print("top features:")
        for name, value in ranked.tail(6)[::-1].items():
            print(f"  {name:<18} {value:.4f}")
        return ranked

    def baseline_accuracy(self):
        """Accuracy from always guessing the most common training class."""
        counts = pd.Series(self.train_y).value_counts()
        majority = counts.index[0]
        accuracy = (self.test_y == majority).mean()
        self.results["majority_baseline"] = {
            "accuracy": round(float(accuracy), 4),
            "class": self.label_names[majority],
        }
        print(f"\nmajority baseline ({self.label_names[majority]}): {accuracy:.4f}")
        return accuracy

    def save(self):
        """Write metrics to JSON for the final comparison stage."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"saved results to {self.results_path.name}")

    def run(self):
        """Load, engineer, train both models, evaluate, save."""
        self.load()
        self.build_features()
        self.split()
        self.baseline_accuracy()
        self.train_logistic_regression()
        self.train_random_forest()
        self.plot_feature_importance()
        self.save()
        return self.results


if __name__ == "__main__":
    MLTrack().run()