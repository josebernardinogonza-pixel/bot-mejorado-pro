import pandas as pd
import numpy as np
from joblib import load
import os
import logging
from scipy.stats import poisson
from scripts.data_fetcher import DataFetcher

MODEL_PATH = "models/model_v2.pkl"
PREDICTIONS_LOG = "predictions_log.csv"

class Predictor:

    def __init__(self):
        self.fetcher = DataFetcher()
        self.model = None
        if os.path.exists(MODEL_PATH):
            self.model = load(MODEL_PATH)
        else:
            logging.error(f"Model file {MODEL_PATH} not found.")

    def poisson_prob(self, lamb, k):
        return poisson.pmf(k, lamb)

    def calculate_expected_goals(self, odds_home, odds_draw, odds_away):
        # Using odds to estimate lambda - very simplified approach for demo purposes
        lambda_home = 1 / odds_home
        lambda_away = 1 / odds_away
        return lambda_home, lambda_away

    def generate_value_bets(self):
        odds_data = self.fetcher.fetch_odds()
        if odds_data is None:
            logging.error("No odds data fetched, cannot generate predictions.")
            return

        results = []
        for event in odds_data:
            try:
                match_id = event.get('id', None)
                teams = (event['home_team'], event['away_team'])
                markets = event['bookmakers'][0]['markets'] if event['bookmakers'] else []
                h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
                if not h2h_market:
                    continue

                prices = h2h_market['outcomes']
                odds_home = next((p['price'] for p in prices if p['name'] == event['home_team']), None)
                odds_away = next((p['price'] for p in prices if p['name'] == event['away_team']), None)
                odds_draw = next((p['price'] for p in prices if p['name'] == 'Draw'), None)
                if None in (odds_home, odds_away, odds_draw):
                    continue

                # Calculate expected goals via Poisson model with odds
                lambda_home, lambda_away = self.calculate_expected_goals(odds_home, odds_draw, odds_away)

                # Prepare features for prediction
                X_pred = [[lambda_home, lambda_away]]

                predicted_value = 0
                if self.model:
                    predicted_value = self.model.predict(X_pred)[0]

                # Decision logic for "value bets" (simplified example)
                value_threshold = 0.5  # Placeholder threshold for value bet
                if predicted_value > value_threshold:
                    results.append({
                        'match_id': match_id,
                        'home_team': teams[0],
                        'away_team': teams[1],
                        'predicted_value': predicted_value,
                        'lambda_home': lambda_home,
                        'lambda_away': lambda_away,
                        'odds_home': odds_home,
                        'odds_draw': odds_draw,
                        'odds_away': odds_away,
                    })

            except Exception as e:
                logging.error(f"Error processing event {event.get('id')}: {e}")

        if results:
            df = pd.DataFrame(results)
            # Append to predictions_log (with timestamp)
            import datetime
            df['timestamp'] = datetime.datetime.now().isoformat()
            if os.path.exists(PREDICTIONS_LOG):
                df_old = pd.read_csv(PREDICTIONS_LOG)
                df = pd.concat([df_old, df], ignore_index=True)
            df.to_csv(PREDICTIONS_LOG, index=False)
            logging.info(f"Saved {len(results)} value bets to {PREDICTIONS_LOG}")
        else:
            logging.info("No value bets generated.")
