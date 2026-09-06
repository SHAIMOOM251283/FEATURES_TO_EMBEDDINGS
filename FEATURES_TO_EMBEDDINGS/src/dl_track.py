import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DLTrack:

    def __init__(self):
        data_dir = "data"
        self.data_dir = PROJECT_ROOT / data_dir
        output_dir = "outputs"
        self.output_dir = PROJECT_ROOT / output_dir
        input_name = "books_clean.csv"
        self.input_path = self.data_dir / input_name
        labels_name = "label_map.csv"
        self.labels_path = self.data_dir / labels_name
        results_name = "dl_results.json"
        self.results_path = self.output_dir / results_name
        curve_name = "dl_training_curve.png"
        self.curve_path = self.output_dir / curve_name
        space_name = "dl_embedding_space.png"
        self.space_path = self.output_dir / space_name

        self.seed = 42
        self.min_word_count = 2
        self.max_length = 250
        self.embed_dim = 64
        self.hidden_dim = 64
        self.dropout = 0.4
        self.batch_size = 32
        self.learning_rate = 0.002
        self.weight_decay = 1e-4
        self.max_epochs = 60
        self.patience = 8
        self.validation_fraction = 0.15

        self.pad_token = "<pad>"
        self.unknown_token = "<unk>"

        self.frame = None
        self.label_names = []
        self.vocabulary = {}
        self.model = None
        self.history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        self.results = {}
        self.tensors = {}

    def set_seeds(self):
        """Fix every source of randomness so the run reproduces."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)

    def load(self):
        """Read the shared clean dataset and the label mapping."""
        self.frame = pd.read_csv(self.input_path)
        labels = pd.read_csv(self.labels_path).sort_values("label")
        self.label_names = list(labels["category"])
        print(f"loaded {len(self.frame)} rows, {len(self.label_names)} categories")
        return self.frame

    def tokenize(self, text):
        """Lowercase and split into word tokens."""
        return re.findall(r"[a-z']+", str(text).lower())

    def build_vocabulary(self):
        """Build the word index from training descriptions only.

        Using test text here would leak information the model is not
        supposed to have, and would quietly inflate the final score.
        """
        train = self.frame[self.frame["split"] == "train"]
        counter = Counter()
        for description in train["description"]:
            counter.update(self.tokenize(description))

        kept = [word for word, count in counter.most_common()
                if count >= self.min_word_count]

        self.vocabulary = {self.pad_token: 0, self.unknown_token: 1}
        for word in kept:
            self.vocabulary[word] = len(self.vocabulary)

        print(f"vocabulary: {len(self.vocabulary)} words "
              f"(from {len(counter)} seen, min count {self.min_word_count})")
        return self.vocabulary

    def encode(self, text):
        """Turn one description into a fixed-length list of token ids."""
        unknown = self.vocabulary[self.unknown_token]
        ids = [self.vocabulary.get(word, unknown)
               for word in self.tokenize(text)][:self.max_length]
        padding = [0] * (self.max_length - len(ids))
        return ids + padding

    def build_tensors(self):
        """Encode every split into tensors, carving a validation slice."""
        train = self.frame[self.frame["split"] == "train"].reset_index(drop=True)
        test = self.frame[self.frame["split"] == "test"].reset_index(drop=True)

        shuffled = train.sample(frac=1.0, random_state=self.seed)
        cut = int(len(shuffled) * self.validation_fraction)
        validation = shuffled.iloc[:cut]
        fitting = shuffled.iloc[cut:]

        for name, part in [("train", fitting), ("val", validation), ("test", test)]:
            tokens = torch.tensor(
                [self.encode(text) for text in part["description"]],
                dtype=torch.long,
            )
            labels = torch.tensor(part["label"].to_numpy(), dtype=torch.long)
            self.tensors[name] = (tokens, labels)
            print(f"{name}: {tuple(tokens.shape)}")

        return self.tensors

    def build_model(self):
        """Assemble the layers, the class-weighted loss and the optimiser.

        The layers live in an nn.ModuleDict rather than a custom nn.Module
        subclass. The container still gives parameters(), train(), eval()
        and state_dict(), so the whole stage stays inside one class.
        """
        self.model = nn.ModuleDict({
            "embedding": nn.Embedding(len(self.vocabulary), self.embed_dim,
                                      padding_idx=0),
            "dropout": nn.Dropout(self.dropout),
            "hidden": nn.Linear(self.embed_dim, self.hidden_dim),
            "activation": nn.ReLU(),
            "output": nn.Linear(self.hidden_dim, len(self.label_names)),
        })

        parameters = sum(p.numel() for p in self.model.parameters())
        print(f"model: {parameters:,} parameters "
              f"({parameters * 4 / 1e6:.1f} MB at float32)")

        train_labels = self.tensors["train"][1]
        counts = torch.bincount(train_labels, minlength=len(self.label_names))
        weights = counts.sum() / (counts.clamp(min=1) * len(self.label_names))
        self.criterion = nn.CrossEntropyLoss(weight=weights.float())

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return self.model

    def pool(self, tokens):
        """Average the embeddings of real tokens, ignoring padding."""
        vectors = self.model["embedding"](tokens)
        mask = (tokens != 0).unsqueeze(-1).float()
        summed = (vectors * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, tokens):
        """Run one batch of token ids through to the class logits."""
        pooled = self.pool(tokens)
        hidden = self.model["activation"](
            self.model["hidden"](self.model["dropout"](pooled))
        )
        return self.model["output"](self.model["dropout"](hidden))

    def run_epoch(self, loader):
        """One pass over the training data."""
        self.model.train()
        total = 0.0
        for tokens, labels in loader:
            self.optimizer.zero_grad()
            loss = self.criterion(self.forward(tokens), labels)
            loss.backward()
            self.optimizer.step()
            total += loss.item() * len(labels)
        return total / len(loader.dataset)

    def score(self, split):
        """Return loss, accuracy and predictions for one split."""
        tokens, labels = self.tensors[split]
        self.model.eval()
        with torch.no_grad():
            logits = self.forward(tokens)
            loss = self.criterion(logits, labels).item()
            predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item()
        return loss, accuracy, predictions.numpy()

    def train(self):
        """Fit the model, stopping when validation loss stops improving."""
        tokens, labels = self.tensors["train"]
        loader = DataLoader(TensorDataset(tokens, labels),
                            batch_size=self.batch_size, shuffle=True)

        best_loss = float("inf")
        best_state = None
        waited = 0
        started = time.time()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.run_epoch(loader)
            val_loss, val_accuracy, _ = self.score("val")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            if epoch % 5 == 0 or epoch == 1:
                print(f"  epoch {epoch:>3}  train {train_loss:.4f}  "
                      f"val {val_loss:.4f}  val acc {val_accuracy:.4f}")

            if val_loss < best_loss - 1e-4:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                waited = 0
            else:
                waited += 1
                if waited >= self.patience:
                    print(f"  stopped early at epoch {epoch}")
                    break

        self.train_seconds = time.time() - started
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"  trained in {self.train_seconds:.1f}s "
              f"over {len(self.history['train_loss'])} epochs")
        return self.model

    def evaluate(self):
        """Score the trained model on the held-out test set."""
        _, accuracy, predictions = self.score("test")
        true_labels = self.tensors["test"][1].numpy()
        macro_f1 = f1_score(true_labels, predictions, average="macro")

        self.results["neural_network"] = {
            "accuracy": round(float(accuracy_score(true_labels, predictions)), 4),
            "macro_f1": round(float(macro_f1), 4),
            "train_seconds": round(self.train_seconds, 3),
            "epochs": len(self.history["train_loss"]),
            "vocabulary_size": len(self.vocabulary),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "per_class": classification_report(
                true_labels, predictions,
                target_names=self.label_names,
                output_dict=True,
                zero_division=0,
            ),
        }

        print(f"\nneural_network")
        print(f"  accuracy  {accuracy:.4f}")
        print(f"  macro f1  {macro_f1:.4f}")
        print(f"  trained   {self.train_seconds:.1f}s")
        return self.results

    def plot_training_curve(self):
        """Show the loss curves, so overfitting is visible rather than assumed."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epochs = range(1, len(self.history["train_loss"]) + 1)

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, self.history["train_loss"], label="train loss",
                  color="#2a6f97")
        axis.plot(epochs, self.history["val_loss"], label="validation loss",
                  color="#d4a017")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.set_title("Training and validation loss")
        axis.legend()
        figure.tight_layout()
        figure.savefig(self.curve_path, dpi=150)
        plt.close(figure)
        print(f"saved curve to {self.curve_path.name}")

    def plot_embedding_space(self):
        """Project pooled document vectors to 2D, coloured by true category.

        This is the picture of what the model taught itself. If the classes
        separate here, the embedding layer found structure in the words that
        no hand-written feature captured.
        """
        tokens, labels = self.tensors["test"]
        self.model.eval()
        with torch.no_grad():
            pooled = self.pool(tokens).numpy()

        coordinates = PCA(n_components=2, random_state=self.seed).fit_transform(pooled)
        labels = labels.numpy()

        figure, axis = plt.subplots(figsize=(9, 7))
        palette = plt.cm.tab10(np.linspace(0, 1, len(self.label_names)))
        for index, name in enumerate(self.label_names):
            mask = labels == index
            if mask.any():
                axis.scatter(coordinates[mask, 0], coordinates[mask, 1],
                             label=name, color=palette[index], s=45, alpha=0.85)
        axis.set_xlabel("component 1")
        axis.set_ylabel("component 2")
        axis.set_title("Learned embedding space (test books)")
        axis.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(self.space_path, dpi=150)
        plt.close(figure)
        print(f"saved embedding plot to {self.space_path.name}")

    def nearest_words(self, word, count=8):
        """List the words closest to one word in the learned space."""
        if word not in self.vocabulary:
            return []
        table = self.model['embedding'].weight.detach()
        vector = table[self.vocabulary[word]]
        similarity = torch.nn.functional.cosine_similarity(table, vector.unsqueeze(0))
        similarity[self.vocabulary[word]] = -1.0
        top = similarity.topk(count).indices.tolist()
        reverse = {index: term for term, index in self.vocabulary.items()}
        return [reverse[i] for i in top]

    def report_neighbours(self):
        """Print learned neighbours for a few probe words."""
        print("\nnearest words in the learned space:")
        for word in ["murder", "recipe", "war", "love", "dragon"]:
            neighbours = self.nearest_words(word)
            if neighbours:
                print(f"  {word:<8} -> {', '.join(neighbours)}")

    def save(self):
        """Write metrics to JSON for the final comparison stage."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.results_path, "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)
        print(f"saved results to {self.results_path.name}")

    def run(self):
        """Seed, load, encode, train, evaluate, visualise, save."""
        self.set_seeds()
        self.load()
        self.build_vocabulary()
        self.build_tensors()
        self.build_model()
        self.train()
        self.evaluate()
        self.plot_training_curve()
        self.plot_embedding_space()
        self.report_neighbours()
        self.save()
        return self.results


if __name__ == "__main__":
    DLTrack().run()