"""
Step 4: Train a Match Outcome Prediction Model
================================================
Predicts: Home Win / Draw / Away Win
Using: Elo ratings, recent form, head-to-head history
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load engineered features
df = pd.read_csv('matches_features.csv')
df['date'] = pd.to_datetime(df['date'])

# 2. Create the TARGET variable (what we're predicting)
def get_result(row):
    if row['home_score'] > row['away_score']:
        return 'H'  # Home win
    elif row['home_score'] < row['away_score']:
        return 'A'  # Away win
    else:
        return 'D'  # Draw

df['result'] = df.apply(get_result, axis=1)

# 3. Build model INPUT features (the numbers the model actually learns from)
df['elo_diff'] = df['home_elo_before'] - df['away_elo_before']
df['form_diff'] = df['home_form'] - df['away_form']

feature_cols = ['elo_diff', 'form_diff', 'h2h_home_winrate']
X = df[feature_cols]
y = df['result']

# 4. TIME-BASED split (critical: never shuffle randomly with time-series data!)
train_mask = df['date'] < '2022-01-01'
test_mask = df['date'] >= '2022-01-01'

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Training on {len(X_train)} matches (before 2022)")
print(f"Testing on {len(X_test)} matches (2022 onwards)")

# 5. Scale features (Logistic Regression performs better when features are on similar scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"TEST ACCURACY: {accuracy:.1%}")
print(f"{'='*50}")

print("\nDetailed performance per class:")
print(classification_report(y_test, y_pred, target_names=['Away Win','Draw','Home Win'], zero_division=0))

print("\nConfusion Matrix (rows=actual, columns=predicted):")
print("Order: Away Win, Draw, Home Win")
print(confusion_matrix(y_test, y_pred))

# 8. What did the model learn? (feature importance via coefficients)
print("\nModel coefficients (which features matter, and how):")
for class_label, coefs in zip(model.classes_, model.coef_):
    print(f"\nPredicting '{class_label}':")
    for feat, coef in zip(feature_cols, coefs):
        print(f"  {feat:20s}: {coef:+.3f}")

# Compare to baseline: what if we just always guessed "Home Win"?
baseline_acc = (y_test == 'H').mean()
print(f"\nBaseline (always guess Home Win): {baseline_acc:.1%}")
print(f"Our model: {accuracy:.1%}")