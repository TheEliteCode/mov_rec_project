# Movie Recommendation Engine
### Predicting how a user will rate a movie they've never seen

---

## The Problem

Imagine you're Netflix. You have 160,000 users and 48,000 movies. Each user has watched and rated a small fraction of your catalogue — maybe 30 movies out of 48,000. That's less than 0.1% of the possible combinations.

Your job: for every user-movie pair they haven't rated yet, predict what rating they *would* give. Get it right and you surface content people love. Get it wrong and they churn.

That's exactly what this project does.

Built on the **MovieLens dataset** — 10 million ratings from 162,541 real users across 48,213 movies — this project constructs a full recommendation pipeline from a naive global average all the way to an ensemble of collaborative filtering and gradient boosting, achieving a **Kaggle leaderboard score of 0.84 RMSE**.

---

## Results

| Model | Validation RMSE | Kaggle RMSE |
|-------|----------------|-------------|
| Global mean (dumb baseline) | 1.05 | 1.05 |
| Bias model (user + movie bias) | 0.8799 | 0.88 |
| Regularised bias (λ=15) | 0.8799 | 0.88 |
| SVD matrix factorization (K=100) | 0.8685 | 0.86 |
| LightGBM + features | 0.8570 | 0.85 |
| **Ensemble (42% SVD + 58% LightGBM)** | **0.8431** | **0.84** |

Every 0.01 improvement in RMSE on a dataset of 5 million predictions is meaningful. The Netflix Prize — the most famous recommender competition in history — ran for 3 years with hundreds of teams and achieved a winning RMSE of 0.856. This pipeline gets close to that benchmark.

---

## How It Works

### Stage 1 — The bias model

The simplest intelligent prediction: every rating is the sum of three components.

```
predicted_rating = global_mean + user_bias + movie_bias
```

- **Global mean (3.53)** — the average rating across all 10 million ratings
- **User bias** — how much a user rates above or below average (generous vs harsh raters)
- **Movie bias** — how much a movie is rated above or below average (great vs poor films)

Simple arithmetic, but it drops RMSE from 1.05 to 0.88 immediately.

Regularisation is applied to prevent movies with very few ratings from getting extreme bias estimates. A movie rated once by someone who gave it 5 stars should not be treated the same as a movie rated 30,000 times.

```python
# Regularised bias — shrinks uncertain estimates toward zero
movie_bias = (sum_of_ratings - global_mean * count) / (count + lambda)
```

### Stage 2 — Matrix factorization (SVD)

The bias model knows *how much* a user likes movies in general, and *how good* a movie is in general. It doesn't know *what kind* of movies a user likes.

SVD (Singular Value Decomposition) fixes this by compressing the entire 162k × 48k user-movie matrix into two smaller matrices — one representing users, one representing movies — each with K latent features.

```
User matrix    (162,541 × 100)
Movie matrix   (100 × 46,901)
```

Each of the 100 dimensions captures some hidden taste pattern the model discovers automatically — things like "how much does this user like slow-paced art films" or "how cerebral is this movie". We never name them; the algorithm figures them out from the rating patterns.

To predict a rating: take the dot product of the user's 100-dimensional vector and the movie's 100-dimensional vector, then add the biases.

```
prediction = global_mean + user_bias + movie_bias + dot(user_vector, movie_vector)
```

K=100 was found to be optimal through validation — enough to capture complex taste patterns without overfitting.

**Sparsity:** The user-movie matrix is 99.88% empty. We use a sparse matrix representation to avoid loading 61GB into RAM, storing only the 9 million non-zero values.

### Stage 3 — LightGBM with engineered features

Matrix factorization captures *who rates similarly*. LightGBM captures *what attributes predict ratings*. They make different kinds of errors, so combining them is powerful.

**Features engineered:**

| Feature group | Features | Why it helps |
|--------------|----------|-------------|
| User stats | bias, mean, std, count | Captures rating behaviour and experience |
| Movie stats | bias, mean, std, count | Captures movie quality and polarisation |
| SVD component | dot product of latent vectors | Collaborative taste signal |
| IMDB metadata | runtime, budget, director avg rating | Content quality signals |
| Genome scores | 50 tag relevance scores | Fine-grained content descriptors |
| Timestamp | rating year, month, days since release | Temporal context of ratings |
| User-genre affinity | avg rating per genre per user | Personal taste preferences |

**Why genome scores beat genre flags:**

Genre flags (is_action, is_drama etc.) showed near-zero importance in our feature importance analysis. A binary "Drama=1" flag doesn't tell you much. The genome scores replace this with 50 continuous relevance scores per movie — tags like "atmospheric", "thought-provoking", "cerebral", "bleak" — which are far more descriptive.

### Stage 4 — Ensemble

Neither model is perfect alone. SVD is strong on users with rich rating history where collaborative patterns are clear. LightGBM is strong on content signals and works well for cold-start movies with good metadata.

Blending them gives the best of both:

```python
final_prediction = 0.42 * svd_prediction + 0.58 * lgbm_prediction
```

The optimal blend weight (42/58) was found by grid search on the validation set.

---

## Project Structure

```
mov_rec_project/
├── data/
│   ├── train.csv           # 10M ratings — userId, movieId, rating, timestamp
│   ├── test.csv            # 5M user-movie pairs to predict
│   ├── movies.csv          # Movie titles and genres
│   ├── imdb_data.csv       # Cast, director, runtime, budget, keywords
│   ├── tags.csv            # User-generated tags
│   ├── genome_scores.csv   # 1128 tag relevance scores per movie (435MB)
│   ├── genome_tags.csv     # Tag descriptions
│   ├── links.csv           # MovieLens to IMDB/TMDB ID mapping
│   └── sample_submission.csv
├── notebook.ipynb          # Full pipeline — EDA, models, submissions
├── README.md
└── submissions/
    ├── submission_baseline.csv
    ├── submission_svd.csv
    ├── submission_lgbm.csv
    └── submission_ensemble.csv
```

---

## Setup

**Requirements**
- Python 3.10 or 3.11 (note: scikit-surprise is not compatible with Python 3.14)
- 16GB RAM recommended for genome_scores processing

**Install dependencies**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install pandas numpy lightgbm scikit-learn matplotlib seaborn scipy ipykernel
```

**Download data**
```bash
pip install kaggle
kaggle competitions download -c movie-recommendation-hackathon-2026
tar -xf movie-recommendation-hackathon-2026.zip -C data
```

**Run**

Open `notebook.ipynb` in VSCode, select the venv kernel, and run cells sequentially.

---

## Key Learnings

**The sparsity problem is real.** 99.88% of the user-movie matrix is empty. The median user has rated only 28 movies out of 48,000 available. Cold-start users and obscure movies are where the biggest prediction errors live.

**Regularisation matters more than it looks.** A movie rated once shouldn't be treated the same as a movie rated 30,000 times. Shrinking uncertain estimates toward the global mean with a regularisation parameter consistently improved validation RMSE.

**Training loss is not validation loss.** LightGBM's training l2 kept dropping past 700 trees but validation RMSE stopped improving — a textbook case of overfitting. Always validate on held-out data.

**Feature importance guides iteration.** Genre flags looked useful in theory but showed near-zero importance in practice. The feature importance plot directly told us to replace them with genome scores — that's data-driven feature engineering.

**Ensembling is almost always worth it.** Two models making different kinds of errors, blended at the right ratio, outperformed either model alone by 0.014 RMSE. That's a significant gain for zero additional training.

---

## Competition

Built for the [Movie Recommendation Hackathon 2026](https://www.kaggle.com/competitions/movie-recommendation-hackathon-2026) on Kaggle.

Evaluation metric: Root Mean Square Error (RMSE) on 5 million user-movie predictions.

---

## Dataset

MovieLens dataset maintained by the GroupLens research group, University of Minnesota.
Additional movie metadata scraped from IMDB.
License: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)