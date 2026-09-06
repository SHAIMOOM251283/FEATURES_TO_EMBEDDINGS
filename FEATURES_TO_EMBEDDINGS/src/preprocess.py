import csv
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Preprocessor:

    def __init__(self):
        data_dir = "data"
        self.data_dir = PROJECT_ROOT / data_dir
        input_name = "books_raw.csv"
        self.input_path = self.data_dir / input_name
        output_name = "books_clean.csv"
        self.output_path = self.data_dir / output_name
        labels_name = "label_map.csv"
        self.labels_path = self.data_dir / labels_name

        # 'Default' and 'Add a comment' are artefacts of how the demo site's
        # catalogue was generated. The books filed under them share nothing,
        # so they are noise for a category classifier.
        self.junk_categories = {"Default", "Add a comment"}
        self.top_n_categories = 10
        self.test_size = 0.2
        self.random_state = 42

        self.frame = None
        self.kept_categories = []
        self.label_map = {}

    def load(self):
        """Read the scraped CSV."""
        self.frame = pd.read_csv(self.input_path)
        print(f"loaded {len(self.frame)} rows from {self.input_path.name}")
        return self.frame

    def drop_junk_categories(self):
        """Remove the demo site's non-categories."""
        before = len(self.frame)
        self.frame = self.frame[~self.frame["category"].isin(self.junk_categories)]
        print(f"dropped junk categories: {before - len(self.frame)} rows removed")
        return self.frame

    def drop_empty_descriptions(self):
        """Remove books with no description — the DL track cannot use them."""
        before = len(self.frame)
        self.frame = self.frame[self.frame["description"].notna()]
        self.frame = self.frame[self.frame["description"].str.strip() != ""]
        print(f"dropped empty descriptions: {before - len(self.frame)} rows removed")
        return self.frame

    def keep_frequent_categories(self):
        """Keep only the most common categories, so each class has enough examples."""
        counts = self.frame["category"].value_counts()
        self.kept_categories = list(counts.head(self.top_n_categories).index)

        before = len(self.frame)
        self.frame = self.frame[self.frame["category"].isin(self.kept_categories)]
        smallest = self.frame["category"].value_counts().min()
        print(f"kept top {self.top_n_categories} categories: "
              f"{before - len(self.frame)} rows removed, "
              f"{len(self.frame)} remain, smallest class {smallest}")
        return self.frame

    def build_label_map(self):
        """Assign a stable integer id to each category, alphabetically."""
        ordered = sorted(self.kept_categories)
        self.label_map = {name: index for index, name in enumerate(ordered)}
        self.frame["label"] = self.frame["category"].map(self.label_map)
        return self.label_map

    def assign_split(self):
        """Mark each row as train or test, keeping class proportions equal."""
        train_index, test_index = train_test_split(
            self.frame.index,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.frame["label"],
        )
        self.frame["split"] = "train"
        self.frame.loc[test_index, "split"] = "test"
        return self.frame

    def save(self):
        """Write the clean dataset and the label mapping."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        columns = ["upc", "title", "category", "label", "split",
                   "price", "rating", "availability", "description", "url"]
        self.frame[columns].to_csv(self.output_path, index=False)
        print(f"saved {len(self.frame)} rows to {self.output_path.name}")

        with open(self.labels_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["label", "category"])
            for name, index in sorted(self.label_map.items(), key=lambda p: p[1]):
                writer.writerow([index, name])
        print(f"saved {len(self.label_map)} labels to {self.labels_path.name}")

    def report(self):
        """Print the per-class train and test counts, so thin classes show up."""
        table = (self.frame
                 .groupby(["category", "split"])
                 .size()
                 .unstack(fill_value=0)
                 .sort_values("train", ascending=False))

        print("\nclass distribution:")
        print(table.to_string())
        print(f"\ntrain rows: {(self.frame['split'] == 'train').sum()}")
        print(f"test rows:  {(self.frame['split'] == 'test').sum()}")
        print(f"smallest test class: {table['test'].min()}")

    def run(self):
        """Load, filter, label, split, save, then summarise."""
        self.load()
        self.drop_junk_categories()
        self.drop_empty_descriptions()
        self.keep_frequent_categories()
        self.build_label_map()
        self.assign_split()
        self.save()
        self.report()
        return self.frame


if __name__ == "__main__":
    Preprocessor().run()