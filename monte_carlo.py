"""
Step 6: Monte Carlo Simulation
================================
Runs the full bracket thousands of times to get real win probabilities
for every team at every stage, instead of one random outcome.
"""
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from elo_ratings import compute_elo_ratings
from collections import defaultdict, deque

np.random.seed(42)
N_SIMULATIONS = 10000

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

# ============================================================
# Pre-compute win probabilities for matchups (cached for speed)
# ============================================================
bracket = pd.read_csv('round_of_32_bracket.csv')
all_32_teams = list(set(bracket['home_team']) | set(bracket['away_team']))

prob_cache = {}
def get_cached_prob(team_a, team_b):
    key = (team_a, team_b)
    if key not in prob_cache:
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
        prob_cache[key] = (p_a, p_b)
    return prob_cache[key]

def simulate_match(team_a, team_b):
    p_a, p_b = get_cached_prob(team_a, team_b)
    return np.random.choice([team_a, team_b], p=[p_a, p_b])

# ============================================================
# Run the Monte Carlo simulation
# ============================================================
initial_matchups = list(zip(bracket['home_team'], bracket['away_team']))

reached_r16 = defaultdict(int)
reached_qf = defaultdict(int)
reached_sf = defaultdict(int)
reached_final = defaultdict(int)
won_title = defaultdict(int)

for sim in range(N_SIMULATIONS):
    r16 = []
    for team_a, team_b in initial_matchups:
        winner = simulate_match(team_a, team_b)
        r16.append(winner)
        reached_r16[winner] += 1

    qf = []
    for i in range(0, len(r16), 2):
        winner = simulate_match(r16[i], r16[i+1])
        qf.append(winner)
        reached_qf[winner] += 1

    sf = []
    for i in range(0, len(qf), 2):
        winner = simulate_match(qf[i], qf[i+1])
        sf.append(winner)
        reached_sf[winner] += 1

    final = []
    for i in range(0, len(sf), 2):
        winner = simulate_match(sf[i], sf[i+1])
        final.append(winner)
        reached_final[winner] += 1

    champion = simulate_match(final[0], final[1])
    won_title[champion] += 1

# ============================================================
# Build results summary + save as JSON for the website
# ============================================================
results = []
for team in all_32_teams:
    results.append({
        'team': team,
        'reach_r16_pct': round(reached_r16[team] / N_SIMULATIONS * 100, 1),
        'reach_qf_pct': round(reached_qf[team] / N_SIMULATIONS * 100, 1),
        'reach_sf_pct': round(reached_sf[team] / N_SIMULATIONS * 100, 1),
        'reach_final_pct': round(reached_final[team] / N_SIMULATIONS * 100, 1),
        'win_title_pct': round(won_title[team] / N_SIMULATIONS * 100, 1),
    })

results.sort(key=lambda x: -x['win_title_pct'])

print(f"Ran {N_SIMULATIONS} simulations\n")
print(f"{'Team':25s} {'Reach R16':>10s} {'Reach QF':>10s} {'Reach SF':>10s} {'Reach Final':>12s} {'Win Title':>10s}")
for r in results:
    print(f"{r['team']:25s} {r['reach_r16_pct']:>9.1f}% {r['reach_qf_pct']:>9.1f}% {r['reach_sf_pct']:>9.1f}% {r['reach_final_pct']:>11.1f}% {r['win_title_pct']:>9.1f}%")

output = {
    'n_simulations': N_SIMULATIONS,
    'bracket': [{'home': h, 'away': a} for h, a in initial_matchups],
    'results': results
}
with open('predictions.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nSaved predictions.json for the website")