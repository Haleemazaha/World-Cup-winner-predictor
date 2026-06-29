"""
Feature Engineering: Recent Form + Head-to-Head
=================================================
Adds two more predictive features on top of Elo ratings:
1. Recent form: points earned in each team's last 5 matches (before this one)
2. Head-to-head: this team's historical win rate vs this specific opponent
"""
import pandas as pd
from collections import defaultdict, deque

def add_form_and_h2h(df):
    df = df.sort_values('date').reset_index(drop=True)

    # Track last 5 results (as points: win=3, draw=1, loss=0) per team
    recent_results = defaultdict(lambda: deque(maxlen=5))

    # Track head-to-head record: h2h[(teamA, teamB)] = [wins, draws, losses] for teamA vs teamB
    h2h_record = defaultdict(lambda: [0, 0, 0])

    home_form_list = []
    away_form_list = []
    h2h_home_winrate_list = []

    for _, row in df.iterrows():
        home, away = row['home_team'], row['away_team']

        # --- FORM: average points from last up-to-5 matches (BEFORE this one) ---
        home_form = sum(recent_results[home]) / len(recent_results[home]) if recent_results[home] else 1.0
        away_form = sum(recent_results[away]) / len(recent_results[away]) if recent_results[away] else 1.0
        home_form_list.append(home_form)
        away_form_list.append(away_form)

        # --- HEAD TO HEAD: home team's historical win rate vs this specific away team ---
        wins, draws, losses = h2h_record[(home, away)]
        total_h2h = wins + draws + losses
        h2h_winrate = wins / total_h2h if total_h2h > 0 else 0.5
        h2h_home_winrate_list.append(h2h_winrate)

        # --- NOW update trackers using this match's actual result ---
        home_score, away_score = row['home_score'], row['away_score']
        if home_score > away_score:
            recent_results[home].append(3)
            recent_results[away].append(0)
            h2h_record[(home, away)][0] += 1
            h2h_record[(away, home)][2] += 1
        elif home_score < away_score:
            recent_results[home].append(0)
            recent_results[away].append(3)
            h2h_record[(home, away)][2] += 1
            h2h_record[(away, home)][0] += 1
        else:
            recent_results[home].append(1)
            recent_results[away].append(1)
            h2h_record[(home, away)][1] += 1
            h2h_record[(away, home)][1] += 1

    df['home_form'] = home_form_list
    df['away_form'] = away_form_list
    df['h2h_home_winrate'] = h2h_home_winrate_list

    return df


if __name__ == "__main__":
    df = pd.read_csv('matches_with_elo.csv')
    df['date'] = pd.to_datetime(df['date'])

    df_featured = add_form_and_h2h(df)

    print("Sample of new features (last 5 rows):")
    print(df_featured[['date','home_team','away_team','home_elo_before','away_elo_before',
                        'home_form','away_form','h2h_home_winrate']].tail(5).to_string(index=False))

    df_featured.to_csv('matches_features.csv', index=False)
    print(f"\nSaved {len(df_featured)} rows with full features to matches_features.csv")