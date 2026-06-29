import pandas as pd

# 1. Load raw data
df = pd.read_csv('results.csv')

# 2. Fix types
df['date'] = pd.to_datetime(df['date'])
df = df[df['home_score'].notnull()].copy()
df['home_score'] = df['home_score'].astype(int)
df['away_score'] = df['away_score'].astype(int)

# 3. Anchor to real FIFA teams (using World Cup history as the reference)
wc_related = df[df['tournament'].isin(['FIFA World Cup', 'FIFA World Cup qualification'])]
fifa_teams = set(wc_related['home_team']) | set(wc_related['away_team'])

# 4. Filter: recent era (1990+) + real FIFA teams only
df_modern = df[df['date'] >= '1990-01-01'].copy()
mask = df_modern['home_team'].isin(fifa_teams) & df_modern['away_team'].isin(fifa_teams)
df_clean = df_modern[mask].sort_values('date').reset_index(drop=True)

# 5. Save the cleaned result
df_clean.to_csv('matches_clean.csv', index=False)

print(f"Done! Cleaned {len(df_clean)} matches saved to matches_clean.csv")