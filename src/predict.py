import os
import joblib
import pandas as pd
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "models")

def kelly_criterion(prob, odds):
    ev = (prob * (odds - 1)) - (1 - prob)
    if ev <= 0:
        return 0.0
    k = ev / (odds - 1)
    return max(min(k, 1), 0)

def infer_and_value(fixtures_df, odds_df):
    # Cargar modelos guardados
    model_btts = joblib.load(os.path.join(MODEL_DIR, "btts_model.pkl"))
    model_xg = joblib.load(os.path.join(MODEL_DIR, "xg_model.pkl"))
    model_goals = joblib.load(os.path.join(MODEL_DIR, "goals_model.pkl"))

    # Generar features reales de fixtures_df con ingeniería realista (simplificado acá)
    # -> Suponer que fixtures_df ya contiene las variables necesarias
    
    X = fixtures_df[["xG_home", "xG_away", "momentum_home", "momentum_away", "poisson_home", "poisson_away"]]

    preds_btts = model_btts.predict_proba(X)[:, 1]
    preds_xg = model_xg.predict_proba(X)[:, 1]
    preds_goals = model_goals.predict(X)

    results = []
    for i, row in fixtures_df.iterrows():
        # Unir odds reales para cada evento
        # Aquí se asume que odds_df ya está sincronizado, ajustar según estructura real
        # Ejemplo:
        odds_btts = float(odds_df.loc[i, 'odds_btts']) if 'odds_btts' in odds_df.columns else 1.9
        odds_xg = float(odds_df.loc[i, 'odds_xg']) if 'odds_xg' in odds_df.columns else 1.85

        stake_btts = kelly_criterion(preds_btts[i], odds_btts)
        stake_xg = kelly_criterion(preds_xg[i], odds_xg)

        results.append({
            "match": f"{row['home_team']} vs {row['away_team']}",
            "prob_btts": preds_btts[i],
            "stake_btts": stake_btts,
            "prob_xg_over25": preds_xg[i],
            "stake_xg_over25": stake_xg,
            "predicted_goal_diff": preds_goals[i],
            "odds_btts": odds_btts,
            "odds_xg": odds_xg
        })
    return results

if __name__ == "__main__":
    print("Para ejecutar predicciones, use integración con extracción e ingeniería.")
