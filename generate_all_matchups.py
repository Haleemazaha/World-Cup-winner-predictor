"""
Generates exact win probabilities for:
1. The 16 real Round of 32 matchups (for the bracket display)
2. ALL 496 possible pairs among the 32 teams (for the team-picker tool)
"""
import pandas as pd
import numpy as np
import json
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from elo_ratings import compute_elo_ratings
from collections import defaultdict, deque

np.random.seed(42)

# ============================================================
# Setup: Elo, form, h2h, trained model (same as before)
# ============================================================
df = pd.read_csv('matches_clean.csv')
df['date'] = pd.to_datetime(df['date'])
current_elo, _ = compute_elo_ratings(df)

df_sorted = df.sort_values('date').reset_index(drop=True)
recent_results = defaultdict(lambda: deque(maxlen=5))
h2h_record = defaultdict(lambda: [0, 0, 0])

for _, row in df_sorted.iterrows():
    home, away = row['home_team'], row['away_team']
    home_score, away_score = row['home_score'], row['away_score']
    if home_score > away_score:
        recent_results[home].append(3); recent_results[away].append(0)
        h2h_record[(home, away)][0] += 1; h2h_record[(away, home)][2] += 1
    elif home_score < away_score:
        recent_results[home].append(0); recent_results[away].append(3)
        h2h_record[(home, away)][2] += 1; h2h_record[(away, home)][0] += 1
    else:
        recent_results[home].append(1); recent_results[away].append(1)
        h2h_record[(home, away)][1] += 1; h2h_record[(away, home)][1] += 1

current_form = {team: (sum(d)/len(d) if d else 1.0) for team, d in recent_results.items()}

def get_h2h_winrate(team_a, team_b):
    wins, draws, losses = h2h_record[(team_a, team_b)]
    total = wins + draws + losses
    return wins / total if total > 0 else 0.5

features_df = pd.read_csv('matches_features.csv')
features_df['result'] = features_df.apply(
    lambda r: 'H' if r['home_score'] > r['away_score'] else ('A' if r['home_score'] < r['away_score'] else 'D'),
    axis=1
)
features_df['elo_diff'] = features_df['home_elo_before'] - features_df['away_elo_before']
features_df['form_diff'] = features_df['home_form'] - features_df['away_form']

feature_cols = ['elo_diff', 'form_diff', 'h2h_home_winrate']
X = features_df[feature_cols]
y = features_df['result']

model = RandomForestClassifier(
    n_estimators=200, max_depth=6, min_samples_leaf=20,
    class_weight='balanced', random_state=42
)
model.fit(X, y)

def get_match_probability(team_a, team_b):
    """Direct model prediction for team_a vs team_b (knockout style, no draws)"""
    elo_diff = current_elo[team_a] - current_elo[team_b]
    form_diff = current_form.get(team_a, 1.0) - current_form.get(team_b, 1.0)
    h2h = get_h2h_winrate(team_a, team_b)
    X_match = pd.DataFrame([[elo_diff, form_diff, h2h]], columns=feature_cols)
    probs = model.predict_proba(X_match)[0]
    class_order = list(model.classes_)
    p_away = probs[class_order.index('A')]
    p_draw = probs[class_order.index('D')]
    p_home = probs[class_order.index('H')]
    p_a = p_home + p_draw * (p_home / (p_home + p_away))
    p_b = p_away + p_draw * (p_away / (p_home + p_away))
    return round(p_a * 100, 1), round(p_b * 100, 1)

# ============================================================
# 1. Exact probabilities for the 16 real Round of 32 matchups
# ============================================================
bracket = pd.read_csv('round_of_32_bracket.csv')
all_32_teams = sorted(set(bracket['home_team']) | set(bracket['away_team']))

bracket_with_probs = []
for _, m in bracket.iterrows():
    p_a, p_b = get_match_probability(m['home_team'], m['away_team'])
    bracket_with_probs.append({
        'home': m['home_team'], 'away': m['away_team'],
        'home_pct': p_a, 'away_pct': p_b
    })

print("ROUND OF 32 EXACT PROBABILITIES:")
for m in bracket_with_probs:
    print(f"  {m['home']:25s} {m['home_pct']:5.1f}%  vs  {m['away_pct']:5.1f}%  {m['away']}")

# ============================================================
# 2. ALL possible pairs among the 32 teams (for team picker)
# ============================================================
all_pairs = {}
for team_a, team_b in combinations(all_32_teams, 2):
    p_a, p_b = get_match_probability(team_a, team_b)
    # Store with a consistent key so the website can look up either order
    all_pairs[f"{team_a}|{team_b}"] = {'team_a': team_a, 'team_b': team_b, 'pct_a': p_a, 'pct_b': p_b}

print(f"\nGenerated {len(all_pairs)} total pairings (32 choose 2 = {len(all_32_teams)*(len(all_32_teams)-1)//2})")

# Save everything
output = {
    'teams': all_32_teams,
    'bracket_matchups': bracket_with_probs,
    'all_pairs': all_pairs
}
with open('matchup_probabilities.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Saved matchup_probabilities.json")
