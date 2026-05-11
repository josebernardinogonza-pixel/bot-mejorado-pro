import os
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    THE_ODDS_API_KEY = os.getenv('THE_ODDS_API_KEY')
    APIOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')

    def __init__(self):
        self.headers_odds = {'x-apisports-key': self.THE_ODDS_API_KEY}
        self.headers_football = {'x-apisports-key': self.API_FOOTBALL_KEY}

    def fetch_odds(self):
        = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
        params = {"apiKey": self.THE_ODDS_API_KEY, "regions": "eu", "markets": "h2h,spreads,totals"}
        try:
            response = requests.get(url, headers=self.headers_odds, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if not data:
                raise ValueError("No data from THE_ODDS_API")
            return data
        except Exception as e:
            logging.error(f"Error fetching odds: {e}")
            return None

    def fetch_historical_results(self, season='2023'):
        url = f"https://v3.football.api-sports.io/fixtures?seasonseason}&league=39"  # example EPL league 39
        try:
            response = requests.get(url, headers=self.headers_football, timeout=15)
            response.raise_for_status()
            data = response.json()
            if 'response' not in data or not data['response']:
                raise ValueError("No historical results found")
            return data['response']
        except Exception as e:
            logging.error(f"Error fetching historical results: {e}")
            return None

    def fetch_live_results(self):
        url = "https://v3.football.api-sports.io/fixtures?live=all"
        try:
            response = requests.get(url, headers=self.headers_football, timeout=15)
            response.raise_for_status()
            data = response.json()
            if 'response' not in data or not data['response']:
                raise ValueError("No live results found")
            return data['response']
        except Exception as e:
            logging.error(f"Error fetching live results: {e}")
            return None
