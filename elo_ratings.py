import pandas as pd

# ============================================================
# ELO RATING SYSTEM
# Computes a "strength score" for every team based on match history.
# Everyone starts at 1500. Winners gain points, losers lose points.
# Beating a stronger team gains you more than beating a weak one.
# ============================================================

def expected_score(rating_a, rating_b):
    """Probability that team A beats team B, based on rating difference."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def goal_diff_multiplier(goal_diff):
    """Bigger wins (more goals) cause slightly bigger rating swings."""
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return (11 + goal_diff) / 8

def compute_elo_ratings(df, k=20, base_rating=1500):
    ratings = {}
    history = []

    df_sorted = df.sort_values('date').reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        home, away = row['home_team'], row['away_team']
        home_score, away_score = row['home_score'], row['away_score']

        if home not in ratings:
            ratings[home] = base_rating
        if away not in ratings:
            ratings[away] = base_rating

        home_elo_before = ratings[home]
        away_elo_before = ratings[away]

        # Home advantage bonus, skipped for neutral-venue matches
        home_advantage = 0 if row['neutral'] else 100
        exp_home = expected_score(home_elo_before + home_advantage, away_elo_before)
        exp_away = 1 - exp_home

        if home_score > away_score:
            actual_home, actual_away = 1, 0
        elif home_score < away_score:
            actual_home, actual_away = 0, 1
        else:
            actual_home, actual_away = 0.5, 0.5

        goal_diff = abs(home_score - away_score)
        mult = goal_diff_multiplier(goal_diff)

        ratings[home] = home_elo_before + k * mult * (actual_home - exp_home)
        ratings[away] = away_elo_before + k * mult * (actual_away - exp_away)

        # Save PRE-match ratings (avoids leaking future info into training)
        history.append({
            'date': row['date'],
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'tournament': row['tournament'],
            'neutral': row['neutral'],
            'home_elo_before': home_elo_before,
            'away_elo_before': away_elo_before,
        })

    return ratings, pd.DataFrame(history)


if __name__ == "__main__":
    df = pd.read_csv('matches_clean.csv')
    df['date'] = pd.to_datetime(df['date'])

    final_ratings, match_history = compute_elo_ratings(df)

    ranked = sorted(final_ratings.items(), key=lambda x: -x[1])
    print("TOP 20 TEAMS BY ELO RATING (current):")
    for i, (team, rating) in enumerate(ranked[:20], 1):
        print(f"{i:2d}. {team:25s} {rating:.0f}")

    match_history.to_csv('matches_with_elo.csv', index=False)
    print(f"\nSaved {len(match_history)} matches with pre-match Elo ratings to matches_with_elo.csv")