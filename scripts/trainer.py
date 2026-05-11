import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import dump, load
import os
import logging
from scipy.stats import poisson
from scripts.data_fetcher import DataFetcher

MODEL_PATH = "models/model_v2.pkl"
PREDICTIONS_LOG = "predictions_log.csv"

class Trainer:

    def __init__(self):
        self.fetcher = DataFetcher()
        self.model = None

    def load_predictions_log(self):
        if os.path.exists(PREDICTIONS_LOG):
            return pd.read_csv(PREDICTIONS_LOG)
        else:
            logging.warning(f"{PREDICTIONS_LOG} not found, starting fresh.")
            return pd.DataFrame(columns=['match_id', 'predicted_home_goals', 'predicted_away_goals', 'actual_home_goals', 'actual_away_goals', 'timestamp'])

    def fetch_actual_results(self, match_ids):
        data = self.fetcher.fetch_historical_results()
        if data is None:
            return None

        records = []
        for match in data:
            fixture_id = match['fixture']['id']
            if fixture_id in match_ids:
                home_goals = match['goals']['home']
                away_goals = match['goals']['away']
                records.append({'match_id': fixture_id, 'actual_home_goals': home_goals, 'actual_away_goals': away_goals})
        return pd.DataFrame(records)

    def calculate_performance(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        return mse

    def poisson_prob(self, lamb, k):
        return poisson.pmf(k, lamb)

    def run_training_pipeline(self):
        log_df = self.load_predictions_log()

        if log_df.empty:
            logging.error("No prediction logs found, cannot train.")
            return

        match_ids = log_df['match_id'].unique()
        actuals_df = self.fetch_actual_results(match_ids)
        if actuals_df is None or actuals_df.empty:
            logging.error("No actual results fetched, aborting training.")
            return

        # Merge actual results into prediction log
        merged = pd.merge(log_df, actuals_df, on='match_id', how='inner')

        # Calculate error metrics and prepare features for retraining
        merged = merged.dropna(subset=['actual_home_goals', 'actual_away_goals'])
        merged['home_goal_error'] = merged['actual_home_goals'] - merged['predicted_home_goals']
        merged['away_goal_error'] = merged['actual_away_goals'] - merged['predicted_away_goals']

        # Features could involve original predictions and errors plus added football metrics (placeholders)
        X = merged[['predicted_home_goals', 'predicted_away_goals']].copy()
        y = (merged['actual_home_goals'] + merged['actual_away_goals']) / 2

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        mse = self.calculate_performance(y, self.model.predict(X))
        logging.info(f"Retrained model MSE: {mse}")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        dump(self.model, MODEL_PATH)
        logging.info(f"Model saved to {MODEL_PATH}")
