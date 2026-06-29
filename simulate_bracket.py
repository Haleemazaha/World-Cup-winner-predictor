"""
Step 5: Simulate the Real World Cup Bracket (Random Forest version)
=====================================================================
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random

random.seed(42)
np.random.seed(42)

from elo_ratings import compute_elo_ratings

df = pd.read_csv('matches_clean.csv')
df['date'] = pd.to_datetime(df['date'])
current_elo, _ = compute_elo_ratings(df)

from collections import defaultdict, deque

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
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
model.fit(X, y)

def predict_match(team_a, team_b):
    elo_diff = current_elo[team_a] - current_elo[team_b]
    form_diff = current_form.get(team_a, 1.0) - current_form.get(team_b, 1.0)
    h2h = get_h2h_winrate(team_a, team_b)

    X_match = pd.DataFrame([[elo_diff, form_diff, h2h]], columns=feature_cols)
    probs = model.predict_proba(X_match)[0]
    class_order = list(model.classes_)

    p_away = probs[class_order.index('A')]
    p_draw = probs[class_order.index('D')]
    p_home = probs[class_order.index('H')]

    p_a_wins = p_home + p_draw * (p_home / (p_home + p_away))
    p_b_wins = p_away + p_draw * (p_away / (p_home + p_away))

    return p_a_wins, p_b_wins

def simulate_match(team_a, team_b):
    p_a, p_b = predict_match(team_a, team_b)
    winner = np.random.choice([team_a, team_b], p=[p_a, p_b])
    return winner, p_a, p_b

bracket = pd.read_csv('round_of_32_bracket.csv')

print("="*70)
print("ROUND OF 32 PREDICTIONS")
print("="*70)

round_of_16_teams = []
for _, match in bracket.iterrows():
    team_a, team_b = match['home_team'], match['away_team']
    winner, p_a, p_b = simulate_match(team_a, team_b)
    round_of_16_teams.append(winner)
    favorite = team_a if p_a > p_b else team_b
    upset = "  <-- UPSET" if winner != favorite else ""
    print(f"{team_a:25s} ({p_a:.0%}) vs {team_b:25s} ({p_b:.0%})  ->  Winner: {winner}{upset}")

print(f"\nRound of 16 teams: {round_of_16_teams}")

def play_round(teams, round_name):
    print(f"\n{'='*70}")
    print(f"{round_name}")
    print(f"{'='*70}")
    next_round = []
    for i in range(0, len(teams), 2):
        team_a, team_b = teams[i], teams[i+1]
        winner, p_a, p_b = simulate_match(team_a, team_b)
        next_round.append(winner)
        print(f"{team_a:25s} ({p_a:.0%}) vs {team_b:25s} ({p_b:.0%})  ->  Winner: {winner}")
    return next_round

quarterfinal_teams = play_round(round_of_16_teams, "ROUND OF 16")
semifinal_teams = play_round(quarterfinal_teams, "QUARTERFINALS")
final_teams = play_round(semifinal_teams, "SEMIFINALS")
champion = play_round(final_teams, "FINAL")

print(f"\n{'='*70}")
print(f"🏆 PREDICTED 2026 WORLD CUP CHAMPION: {champion[0]}")
print(f"{'='*70}")