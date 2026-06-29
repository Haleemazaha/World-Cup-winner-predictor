# World-Cup-winner-predictor

# 🏆 World Cup 2026 Predictor

A machine learning project that predicts outcomes for the 2026 FIFA World Cup knockout stage, using historical match data, Elo ratings, and a Random Forest classifier — visualized through an interactive bracket website.

**[Live Demo](#)** *(add your deployed link here once live)*

![Status](https://img.shields.io/badge/status-complete-brightgreen) ![Python](https://img.shields.io/badge/python-3.9-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange)

---

## What this does

This project trains a model on **30,000+ international football matches (1990–2026)** to predict the outcome of any matchup between two national teams, then runs **10,000 Monte Carlo simulations** of the real 2026 World Cup Round of 32 bracket to estimate each team's probability of winning the tournament.

The result is an interactive website showing:
- A full bracket from Round of 32 to Final, with win probability for every matchup
- A "Title Odds" leaderboard ranking all 32 teams by chance of winning it all
- A head-to-head predictor — pick any two of the 32 teams and see the model's predicted win probability

## How it works

### 1. Data
Sourced from a public Kaggle dataset of international football results (1872–present). Filtered to:
- Matches from **1990 onwards** (modern football era)
- Only **FIFA-recognized national teams** (anchored against real World Cup/qualifier participants, to exclude regional/non-FIFA entities like CONIFA teams)

### 2. Feature Engineering
Three core features computed from scratch:
- **Elo rating** — a dynamically updated team strength score (starts at 1500, adjusts after every match based on opponent strength and goal difference), with a home-advantage adjustment that's disabled for neutral-venue matches
- **Recent form** — average points earned over each team's last 5 matches
- **Head-to-head win rate** — historical win percentage between the specific pair of teams

### 3. Model
A **Random Forest classifier** (scikit-learn) predicts match outcome (Home Win / Draw / Away Win) from the three features above. Chosen over a simpler Logistic Regression baseline specifically because it produces more realistic, less overconfident probability estimates and meaningfully predicts draws instead of defaulting to the majority class.

Trained on data through 2022, tested on 2022–2026 matches — a genuine out-of-sample evaluation rather than a random shuffle, since match data is time-ordered (training on future matches to predict the past would be data leakage).

### 4. Tournament Simulation
The real, confirmed 2026 Round of 32 bracket is simulated **10,000 times** using weighted-random outcomes drawn from the model's predicted probabilities (not just "always pick the favorite," since upsets are a real part of football). Knockout draws are redistributed proportionally between the two teams, since draws aren't a valid final result in single-elimination play.

### 5. Website
A static HTML/CSS/JS site reads two pre-computed JSON files (no backend required) to render the bracket and power the team picker.

## Project structure

```
world-cup-predictor/
├── results.csv                      # Raw Kaggle dataset
├── clean_data.py                    # Step 1: cleaning & filtering
├── matches_clean.csv                # Cleaned output
├── elo_ratings.py                   # Step 2: Elo rating engine
├── matches_with_elo.csv             # Matches + pre-match Elo
├── add_features.py                  # Step 3: form + head-to-head features
├── matches_features.csv             # Full feature set
├── train_model.py                   # Logistic Regression baseline
├── train_model_rf.py                # Random Forest comparison
├── round_of_32_bracket.csv          # Real, confirmed 2026 bracket
├── monte_carlo.py                   # 10,000-simulation tournament engine
├── generate_all_matchups.py         # Exact probabilities for all 496 team pairs
├── predictions.json                 # Monte Carlo results (for website)
├── matchup_probabilities.json       # All pairwise probabilities (for website)
└── index.html                       # The website itself
```

## Running it yourself

```bash
# 1. Set up environment
python3 -m venv venv
source venv/bin/activate
pip install pandas scikit-learn matplotlib

# 2. Run the pipeline in order
python3 clean_data.py
python3 elo_ratings.py
python3 add_features.py
python3 monte_carlo.py
python3 generate_all_matchups.py

# 3. Serve the website locally
python3 -m http.server 8000
# then open http://localhost:8000
```

## Key results

| Model | Test Accuracy (2022–2026) | Predicts Draws? |
|---|---|---|
| Logistic Regression | 59.8% | No (always picks Win/Loss) |
| **Random Forest (used)** | 57.6% | Yes — meaningfully |

Lower raw accuracy was an intentional tradeoff: a model that never predicts draws at all is less useful for realistic tournament simulation than one with balanced, if slightly lower, overall accuracy.

**Most important feature by far**: Elo rating difference (64% of feature importance), consistent with footballing intuition — long-term team strength dominates over short-term form or head-to-head quirks.

## Limitations & honest caveats

- Trained only on match results — no player-level data (injuries, suspensions, squad changes) is factored in
- A single simulation run is inherently random; probabilities are only meaningful in aggregate across thousands of runs
- Elo and form are backward-looking; they can't anticipate a team peaking or declining suddenly
- The model has no access to current tournament momentum/morale beyond what's captured in recent results

## Tech stack

Python · pandas · scikit-learn · HTML/CSS/JavaScript (no frontend framework — vanilla JS)

---

*Built as a learning project to understand the full ML pipeline: data cleaning, feature engineering, model comparison, and deployment — using a real, live sporting event as the test case.*
