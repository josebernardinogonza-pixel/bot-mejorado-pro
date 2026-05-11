import os
import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import joblib

# Environment Variables expected
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Util: Load or initialize model
def load_or_init_model(model_name, model_type='classifier'):
    path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        if model_type == 'classifier':
            return XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
        else:
            return XGBRegressor(n_jobs=1)

# 1. Data Extraction Pipeline
def fetch_football_data():
    """Fetches fixtures and stats from API Football."""
    headers = {'x-apisports-key': API_FOOTBALL_KEY}
    # Example endpoint for upcoming fixtures (Can be adjusted)
    url_fixtures = "https://v3.football.api-sports.io/fixtures?status=NS&season=2023"
    try:
        resp = requests.get(url_fixtures, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        fixtures = data.get('response', [])
        return fixtures
    except Exception as e:
        print(f"Error fetching football data: {e}")
        return []

def fetch_odds_data(fixtures):
    """Fetches odds from Odds API and matches with fixtures."""
    headers = {'Authorization': f'Bearer {ODDS_API_KEY}'}
    odds_data = []
    try:
        for fixture in fixtures:
            # Only take matches with bookmakers info, or fetch on-demand if API supports
            fixture_id = fixture['fixture']['id']
            url_odds = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?regions=eu&markets=h2h,totals,btbs&fixture_id={fixture_id}"
            resp = requests.get(url_odds, headers=headers, timeout=8)
            if resp.status_code == 200:
                odds_json = resp.json()
                if isinstance(odds_json, list) and len(odds_json) > 0:
                    odds_data.append(odds_json[0])
    except Exception as e:
        print(f"Error fetching odds data: {e}")
    return odds_data

# 2. Feature Engineering
def engineer_features(fixtures, odds):
    """Create features for models using xG dynamics, momentum, Poisson expected goals."""
    # Simplified placeholder for demonstration purposes
    records = []
    for f, o in zip(fixtures, odds):
        fixture_info = f['fixture']
        teams = f['teams']
        goals = f['goals']
        home_team = teams['home']['name']
        away_team = teams['away']['name']
        # Example features
        xG_home = np.random.uniform(0.5, 2.5)
        xG_away = np.random.uniform(0.5, 2.5)
        momentum_home = np.random.uniform(-1, 1)
        momentum_away = np.random.uniform(-1, 1)
        poisson_lambda_home = xG_home
        poisson_lambda_away = xG_away
        btbs = int(goals['home'] > 0 and goals['away'] > 0)
        total_goals = goals['home'] + goals['away']

        rec = {
            "home_team": home_team,
            "away_team": away_team,
            "xG_home": xG_home,
            "xG_away": xG_away,
            "momentum_home": momentum_home,
            "momentum_away": momentum_away,
            "poisson_home": poisson_lambda_home,
            "poisson_away": poisson_lambda_away,
            "btbs": btbs,
            "total_goals": total_goals,
            "target_btts": btbs,
            "target_over25": int(total_goals > 2.5),
            "target_goal_diff": goals['home'] - goals['away']
        }
        records.append(rec)
    return pd.DataFrame(records)

# 3. Model Training, saving, and loading models
def train_and_persist_models(df):
    X = df[["xG_home", "xG_away", "momentum_home", "momentum_away", "poisson_home", "poisson_away"]]
    
    # BTTS model (binary)
    y_btts = df["target_btts"]
    model_btts = load_or_init_model("btts_model", 'classifier')
    X_train, X_val, y_train, y_val = train_test_split(X, y_btts, test_size=0.2, random_state=42)
    model_btts.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model_btts, os.path.join(MODEL_DIR, "btts_model.pkl"))

    # Over 2.5 Goals model (binary)
    y_over25 = df["target_over25"]
    model_xg = load_or_init_model("xg_model", 'classifier')
    model_xg.fit(X_train, y_over25, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model_xg, os.path.join(MODEL_DIR, "xg_model.pkl"))

    # Goal difference regression model
    y_goal_diff = df["target_goal_diff"]
    model_goals = load_or_init_model("goals_model", 'regressor')
    model_goals.fit(X_train, y_goal_diff)
    joblib.dump(model_goals, os.path.join(MODEL_DIR, "goals_model.pkl"))

    print("Models trained and saved.")

# 4. Inference and Value Calculation
def predict_and_evaluate(fixtures):
    df = engineer_features(fixtures, fixtures)  # Using fixtures twice for demonstration
    X = df[["xG_home", "xG_away", "momentum_home", "momentum_away", "poisson_home", "poisson_away"]]

    # Load models
    model_btts = joblib.load(os.path.join(MODEL_DIR, "btts_model.pkl"))
    model_xg = joblib.load(os.path.join(MODEL_DIR, "xg_model.pkl"))
    model_goals = joblib.load(os.path.join(MODEL_DIR, "goals_model.pkl"))

    preds_btts = model_btts.predict_proba(X)[:, 1]
    preds_xg = model_xg.predict_proba(X)[:, 1]
    preds_goals = model_goals.predict(X)

    # Assume we get odds directly from external source or static example here
    odds_btts = 1.9
    odds_xg = 1.85
    odds_goals_over = 1.95

    def kelly_criterion(prob, odds, bankroll=1.0):
        ev = (prob * (odds - 1)) - (1 - prob)
        if ev <= 0:
            return 0
        k = ev / (odds - 1)
        return max(min(k, 1), 0)

    results = []
    for idx in range(len(df)):
        result = {
            "match": f"{df.iloc[idx].home_team} vs {df.iloc[idx].away_team}",
            "prob_btts": preds_btts[idx],
            "stake_btts": kelly_criterion(preds_btts[idx], odds_btts),
            "prob_xg_over25": preds_xg[idx],
            "stake_xg_over25": kelly_criterion(preds_xg[idx], odds_xg),
            "predicted_goal_diff": preds_goals[idx],
            "odds_btts": odds_btts,
            "odds_xg": odds_xg,
            "odds_goals_over": odds_goals_over
        }
        results.append(result)
    return results

def main():
    fixtures = fetch_football_data()
    if not fixtures:
        print("No fixtures data available. Exiting.")
        return
    odds = fetch_odds_data(fixtures)
    if not odds:
        print("No odds data retrieved. Continuing with available fixtures only.")

    df_features = engineer_features(fixtures, odds if odds else fixtures)
    train_and_persist_models(df_features)

    results = predict_and_evaluate(fixtures)
    for res in results:
        print(res)

if __name__ == "__main__":
    main()
