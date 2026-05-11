import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_models():
    features_path = os.path.join(PROCESSED_DATA_DIR, "features.parquet")
    df = pd.read_parquet(features_path)

    X = df[["xG_home", "xG_away", "momentum_home", "momentum_away", "poisson_home", "poisson_away"]]

    # BTTS classifier
    y_btts = df["target_btts"]
    X_train, X_val, y_train, y_val = train_test_split(X, y_btts, test_size=0.2, random_state=42)
    model_btts = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    model_btts.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    joblib.dump(model_btts, os.path.join(MODEL_DIR, "btts_model.pkl"))

    # Over 2.5 goals classifier
    y_over25 = df["target_over25"]
    model_xg = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    model_xg.fit(X_train, y_over25, eval_set=[(X_val, y_over25.loc[y_val.index])], verbose=False)
    joblib.dump(model_xg, os.path.join(MODEL_DIR, "xg_model.pkl"))

    # Goal difference regressor
    y_diff = df["goal_diff"]
    model_goals = XGBRegressor(n_jobs=1)
    model_goals.fit(X_train, y_diff)
    joblib.dump(model_goals, os.path.join(MODEL_DIR, "goals_model.pkl"))

    print("Modelos entrenados y guardados.")

if __name__=="__main__":
    train_models()
