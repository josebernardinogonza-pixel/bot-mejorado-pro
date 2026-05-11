import os
import requests
import pandas as pd

API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY')
ODDS_API_KEY = os.getenv('ODDS_API_KEY')

def fetch_fixtures():
    url = "https://v3.football.api-sports.io/fixtures?league=39&season=2023"
    headers = {'x-apisports-key': API_FOOTBALL_KEY}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json()['response']

def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds?apiKey={ODDS_API_KEY}&regions=uk&markets=h2h"
    res = requests.get(url)
    res.raise_for_status()
    return res.json()

def save_fixtures_parquet(fixtures):
    os.makedirs('data/raw', exist_ok=True)
    df = pd.json_normalize(fixtures)
    df.to_parquet('data/raw/fixtures_39_2023.parquet')
    print("Fixtures guardados en data/raw/fixtures_39_2023.parquet")
