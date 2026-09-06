# FEATURES_TO_EMBEDDINGS

One dataset, four representations, one honest comparison.

This project scrapes a single dataset of book descriptions and runs it down four
separate roads: hand engineered features, a word counting formula, a neural
network that learns its own representation, and a smaller version of that same
network. Every model is scored on the same 101 held out books, so the numbers
are directly comparable.

The point was never to prove that deep learning wins. It was to find out what
actually happens when the same problem is handed to different paradigms, and to
report the result whichever way it fell.

![One book, two minds](FEATURES_TO_EMBEDDINGS/images/00-one-book-two-minds.png)

## Headline result

| Approach                                 | Input                | Representation            | Accuracy        | Macro F1        | Training time   |
| ---------------------------------------- | -------------------- | ------------------------- | --------------- | --------------- | --------------- |
| Majority baseline                        | none                 | none                      | 0.218           |                 |                 |
| Embeddings, 470K parameters              | raw words            | learned                   | 0.416           | 0.301           | 4.27s           |
| Engineered features, Random Forest       | 22 hand made numbers | fixed, human chosen       | 0.446           | 0.386           | 0.33s           |
| Embeddings, tuned to 65K parameters      | raw words            | learned                   | 0.495           | 0.461           | 2.66s           |
| Engineered features, Logistic Regression | 22 hand made numbers | fixed, human chosen       | 0.515           | 0.480           | 0.008s          |
| **TF-IDF, Logistic Regression**    | **raw words**  | **fixed, computed** | **0.624** | **0.575** | **1.41s** |

The winner is a word counting formula that predates all of this. It reads the
same words the neural network reads, but its representation is computed by a
formula rather than learned by training, and on 403 practice books that turns
out to be the better trade.

![Accuracy and macro F1 for all five models](FEATURES_TO_EMBEDDINGS/outputs/comparison_summary.png)

## The illustrated walkthrough

The two tracks are explained step by step, with real arithmetic at miniature
scale, in the images below. The scale is deliberately reduced so each
calculation can be followed by hand. Three measurements instead of 22, four
numbers per word instead of 24, three genres instead of ten. Every image states
what the production code does instead.

### Part one: the machine learning track

A person writes the rules before any book is read. The book is reduced to a row
of numbers, and from that point on nothing knows it was ever a book.

**Step 1.** Three hand written rules count words, digits and keywords in one
real description.

![Three rulers counting one book](FEATURES_TO_EMBEDDINGS/images/ml-01-three-rulers.png)

**Step 2.** Nine multiplications and three additions produce one score per
genre. This is the entire model.

![Multiplying the counts by learned weights](FEATURES_TO_EMBEDDINGS/images/ml-02-multiply-and-add.png)

**Step 3.** Those scores become percentages, and the answer arrives sounding
confident. It is wrong.

![Turning scores into percentages](FEATURES_TO_EMBEDDINGS/images/ml-03-percentages.png)

**Step 4.** Nobody typed the weights in. They start at zero and are nudged by
being wrong, one book at a time.

![Where the weights came from](FEATURES_TO_EMBEDDINGS/images/ml-04-weights.png)

**Step 5.** The real marks across all ten genres, including one that scores
flat zero.

![The real report card](FEATURES_TO_EMBEDDINGS/images/ml-05-report-card.png)

**Step 6.** Why `recipe` worked, why the history words died, and why only 3 of
38 Fantasy books mention a dragon.

![Why the keyword hunches failed](FEATURES_TO_EMBEDDINGS/images/ml-06-hunches.png)

### Part two: the deep learning track

No rules are written and nothing is measured. Every word starts as random
numbers, and those numbers are shaped by being wrong.

**Step 1.** Every word is handed four random numbers that mean nothing at all.

![Every word starts as random numbers](FEATURES_TO_EMBEDDINGS/images/dl-01-random-numbers.png)

**Step 2.** The whole book is averaged down to four numbers, which land close
to zero because random values cancel out.

![Averaging the word numbers into one summary](FEATURES_TO_EMBEDDINGS/images/dl-02-average.png)

**Step 3.** The first guess comes out 34, 33, 33. That is a shrug rather than a
prediction.

![The first guess is a coin toss](FEATURES_TO_EMBEDDINGS/images/dl-03-first-guess.png)

**Step 4.** The correction reaches back and changes the words themselves. This
is the step the other track can never take.

![Being wrong rewrites the words](FEATURES_TO_EMBEDDINGS/images/dl-04-correction.png)

**Step 5.** After roughly fifty thousand corrections, words that predict the
same genre have drifted together, and the same book is answered correctly.

![Fifty thousand nudges later](FEATURES_TO_EMBEDDINGS/images/dl-05-after-training.png)

**Step 6.** The real marks, against the same book set the other track was
graded on. A dead heat overall, with opposite strengths.

![The Learner's real report card](FEATURES_TO_EMBEDDINGS/images/dl-06-report-card.png)

## What the data turned out to be

The source is books.toscrape.com, a site published by Zyte, formerly
Scrapinghub, as a scraping sandbox. It holds 1,000 books across 50 categories,
with structured fields and a free text description on every product page.

Two problems surfaced before any model was trained.

**The tabular fields carry no signal.** Price correlates with rating at 0.03 and
with availability at 0.01. Mean price is flat across all five star ratings, from
34.56 to 36.09. These values were generated randomly when the sandbox was built,
so the original plan to predict a price bracket was abandoned. It would have
produced a working pipeline that proved nothing.

**Two of the largest categories are not categories.** `Default` with 152 books
and `Add a comment` with 67 are artefacts of how the catalogue was generated.
The titles filed under them share nothing. Together they are 22 percent of the
dataset and the first and fourth largest classes, so a classifier would have
spent real capacity searching for a pattern that does not exist.

After removing those and keeping the ten most frequent remaining categories,
504 books remain: 403 for training and 101 for testing, stratified by category.

## Where each approach succeeded

![Per category F1 across all five models](FEATURES_TO_EMBEDDINGS/outputs/comparison_heatmap.png)

Three patterns are worth reading out of that grid.

**Food and Drink is flat across every model**, between 0.67 and 0.86. Cookbooks
are structurally distinct enough that even 22 crude measurements catch them. No
representation has an advantage.

**Young Adult is the widest spread**, from 0.73 for TF-IDF down to 0.00 for the
oversized network. This is the clearest single case for reading actual words.

**Historical Fiction is where the paradigms invert.** Both engineered feature
models scored exactly 0.00, never once identifying it. TF-IDF reached 0.29. The
tuned network reached 0.40, the best score any model achieved on that class. A
wartime novel is shaped like any other novel, so counting cannot find it. Only a
model that learns what words mean makes progress.

The mirror image is Fantasy, where the tuned network scores 0.15 against 0.42
for logistic regression. The two approaches fail in opposite directions.

## Why the hand written keyword rules mostly failed

Counted directly from the 403 training books:

| Rule                                                     | Fires correctly | Fires wrongly | Outcome                    |
| -------------------------------------------------------- | --------------- | ------------- | -------------------------- |
| `recipe` for Food and Drink                            | 23 of 24        | 0 of 379      | A perfect clue             |
| `war, century, history, empire` for Historical Fiction | 16 of 21        | 72 of 382     | Worse than useless         |
| `dragon` for Fantasy                                   | 3 of 38         |               | Ranked last of 22 features |

The `recipe` rule is why Food and Drink scored 0.86. The history words fire in
76 percent of the right books, which sounds promising, but they also fire in 72
books of other genres, roughly four and a half false alarms for every correct
one. Nonfiction alone fires 28 times.

The `dragon` rule reveals a plain mistake in intuition. Only 3 of 38 Fantasy
books mention one.

The lesson is that a hand written rule only works when the word belongs to one
genre alone, and that cannot be known in advance. Of the 22 features, the ones
that survived were the dullest: title length, description length, average word
size. Plain length beat every clever keyword guess.

![Feature importance across the 22 engineered features](FEATURES_TO_EMBEDDINGS/outputs/ml_feature_importance.png)

The flatness of that ranking is the finding. With 22 features an even split
would give each one 0.045. The strongest reaches only 0.079, so no single
measurement is decisive and the model is scraping together weak evidence from
everywhere. The keyword hunches sit at the bottom.

## The capacity experiment

The first neural network gave 470,538 parameters to 343 training books, roughly
1,400 parameters per book, and overfitted badly. Training loss fell to 0.71
while validation loss bottomed out near 1.75 and climbed.

![Training and validation loss for the 470K parameter model](FEATURES_TO_EMBEDDINGS/outputs/dl_training_curve.png)

The two curves fall together until roughly epoch 10, then separate. Everything
learned after that point was memorisation rather than generalisation.

What that failure looks like from the inside is clearer still. Projecting the
book vectors the oversized model produced, coloured by their true genre, gives
this:

![Learned embedding space of the 470K parameter model](FEATURES_TO_EMBEDDINGS/outputs/dl_embedding_space.png)

There are no clusters. Ten genres are scattered uniformly, with Nonfiction
spanning the full width and Young Adult everywhere at once. A model that had
learned what the words mean would separate them here. This one converged,
trained successfully, and arranged the books at random.

A sweep across six capacities settled the question. Selection was made on a
validation split carved out of the training data, never on the test set, and
only the validation winner was scored on test.

| Configuration    | Parameters       | Vocabulary      | Epochs        | Validation accuracy |
| ---------------- | ---------------- | --------------- | ------------- | ------------------- |
| tiny             | 10,410           | 1,281           | 26            | 0.050               |
| small            | 26,762           | 1,645           | 199           | 0.417               |
| **medium** | **65,338** | **2,687** | **148** | **0.467**     |
| large            | 111,210          | 3,432           | 133           | 0.400               |
| wide             | 220,138          | 4,527           | 91            | 0.450               |
| original         | 470,538          | 7,277           | 62            | 0.450               |

The shape is an inverted U. Below roughly 25,000 parameters the model cannot
represent the task at all, and `tiny` never left random guessing, with a loss
stuck at 2.30 against a ten class baseline of 2.303. Above roughly 220,000 it
memorises. The optimum sits near 160 parameters per training book.

![Validation accuracy and memorisation against parameter count](FEATURES_TO_EMBEDDINGS/outputs/dl_sweep_curve.png)

The right hand panel is the cleaner result. The gap between validation and
training loss climbs steadily with parameter count, from roughly zero at the
smallest configuration to 1.47 at the largest. That is overfitting isolated and
measured.

The tuned model at 65,338 parameters lifted test macro F1 from 0.301 to 0.461,
a 53 percent relative gain, and eliminated three of the four classes the
oversized model had scored zero on.

One methodological note belongs here. The first version of this sweep capped
training at 60 epochs, and four of the six configurations were still improving
when they hit that ceiling. Smaller models need more epochs to converge, so the
first run measured training budget rather than capacity, and produced the
opposite conclusion. The corrected sweep allows every configuration up to 600
epochs with early stopping, so each one halts on its own.

## Honest caveats

**TF-IDF benefits from series leakage.** Its strongest Fantasy terms are
`prince`, `magic`, `harry`, `seven`, `potter` and `harry potter`. Four of the top
six are Harry Potter. Sequential Art relies on `tohru`, a Fruits Basket
character. These are proper nouns from individual titles, and they help only
because the same series appears on both sides of the split. Genuine genre
vocabulary is present too, including `recipes`, `cookbook`, `detective`, `crime`
and `world war`, but 0.624 should be read as an optimistic ceiling.

![Strongest TF-IDF terms for four genres](FEATURES_TO_EMBEDDINGS/outputs/tfidf_top_terms.png)

**Test classes are small.** Historical Fiction has 5 test books, so one
misclassification moves its recall by 20 points. Differences of a few points
between models are noise. Macro F1 is reported alongside accuracy because
Nonfiction has 110 books and Historical Fiction has 26, a fourfold imbalance
that plain accuracy would flatter.

**The capacity selection is within noise.** Validation accuracy for `medium`,
`wide` and `original` was 0.467, 0.450 and 0.450 on a 60 book validation set.
`medium` won by one book. The defensible claim is that somewhere between 65,000
and 220,000 parameters beats 470,000, not that 65,338 is optimal.

**The dataset is synthetic.** Prices, ratings and stock counts were randomly
generated, and two categories are meaningless. Conclusions about representation
quality should transfer. Conclusions about book genres should not.

## Repository layout

```
FEATURES_TO_EMBEDDINGS/
    src/
        scraper.py             async Playwright extraction, 1,000 books in 113s
        preprocess.py          filtering, labelling, and the one shared split
        ml_track.py            22 engineered features, Logistic Regression and Random Forest
        tfidf_track.py         TF-IDF with unigrams and bigrams, Logistic Regression
        dl_track.py            trainable embeddings, mean pooling, dense classifier
        dl_capacity_sweep.py   six capacities, selected on validation
        compare.py             reads every result file, builds the tables and figures
    images/                    the 13 hand made explanatory visuals
    data/                      scraped and processed data
    outputs/                   figures and metrics written by the code
```

Two directories hold pictures, for two different reasons. `outputs/` is written
by the code, so re-running the pipeline recreates it. `images/` holds the
explanatory visuals, which nothing can regenerate.

Alongside the seven figures, `outputs/` holds the metrics each stage wrote:
`ml_results.json`, `tfidf_results.json`, `dl_results.json` and
`dl_sweep_results.json`, plus `comparison.csv` and `comparison_per_class.csv`
assembled from them. Every number quoted in this README comes from those files.

Models are built with scikit-learn and PyTorch, extraction with Playwright, and
figures with Matplotlib.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Two of those lines matter. `playwright install chromium` downloads the browser
binary, which pip does not, and skipping it is the most common first run
failure. The separate torch line fetches the CPU only build, because a plain
install pulls several gigabytes of NVIDIA CUDA runtime that a machine without a
GPU will never use.

Everything here runs on CPU. The full pipeline needs no GPU and peaks well under
2GB of memory.

## Running

```
python src/scraper.py
python src/preprocess.py
python src/ml_track.py
python src/tfidf_track.py
python src/dl_track.py
python src/dl_capacity_sweep.py
python src/compare.py
```

Stages must run in that order, since each reads what the previous one wrote.
Everything except the scrape finishes in seconds. All random seeds are fixed at
42, so results reproduce exactly, and the same numbers were confirmed on two
separate machines.

## What this project actually demonstrates

The finding is not that one paradigm beats another. It is that **representation
quality decides the outcome, and learning a representation is only worth it when
there is enough data to learn it from.** At 403 documents a formula from the
1970s beats a neural network. The crossover point would arrive with more data,
which this project did not test and does not claim.

A secondary finding, visible in the per category grid, is that aggregate scores
hide the interesting behaviour. Two models level on accuracy can be good at
entirely different things, and the class where the neural network finally earned
its keep is the class every other approach failed completely.