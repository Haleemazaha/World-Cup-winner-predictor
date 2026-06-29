"""
Step 4b: Random Forest Model (handles draws + non-linear patterns better)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('matches_features.csv')
df['date'] = pd.to_datetime(df['date'])

def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'

df['result'] = df.apply(get_result, axis=1)
df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
df['form_diff'] = df['home_form'] - df['away_form']

feature_cols = ['elo_diff', 'form_diff', 'h2h_home_winrate']
X = df[feature_cols]
y = df['result']

train_mask = df['date'] < '2022-01-01'
test_mask = df['date'] >= '2022-01-01'
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

# Random Forest does NOT need feature scaling (tree-based models are scale-invariant)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=20,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"RANDOM FOREST TEST ACCURACY: {accuracy:.1%}")
print("\nDetailed performance:")
print(classification_report(y_test, y_pred, target_names=['Away Win','Draw','Home Win'], zero_division=0))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature importance:")
for feat, imp in zip(feature_cols, model.feature_importances_):
    print(f"  {feat:20s}: {imp:.3f}")