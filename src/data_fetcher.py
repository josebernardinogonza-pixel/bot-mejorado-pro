import os
import requests
import time
import pandas as pd

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

HEADERS_FOOTBALL = {'x-apisports-key': API_FOOTBALL_KEY}
HEADERS_ODDS = {'Authorization': f'Bearer {ODDS_API_KEY}'}

def fetch_fixtures(season=2023, league_id=39):
    """Extrae fixtures y resultados históricos reales (ligas, temporadas especificas)"""
    url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season={season}"
    try:
        resp = requests.get(url, headers=HEADERS_FOOTBALL, timeout=10)
        resp.raise_for_status()
        data = resp.json()['response']
        df = pd.json_normalize(data)
        df.to_parquet(os.path.join(RAW_DATA_DIR, f"fixtures_{league_id}_{season}.parquet"), index=False)
        return df
    except Exception as e:
        print(f"Error fetching fixtures: {e}")
        return pd.DataFrame()

def fetch_odds(sport_key='soccer_epl', regions='eu', markets='h2h,totals,btbs', odds_date=None):
    """Extrae odds reales diarias y los guarda para entrenamiento/inferencia"""
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds/"
    params = {
        "regions": regions,
        "markets": markets
    }
    if odds_date:
        params["date"] = odds_date
    try:
        resp = requests.get(url, headers=HEADERS_ODDS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        df = pd.json_normalize(data)
        filename = f"odds_{sport_key}_{time.strftime('%Y%m%d')}.parquet"
        df.to_parquet(os.path.join(RAW_DATA_DIR, filename), index=False)
        return df
    except Exception as e:
        print(f"Error fetching odds: {e}")
        return pd.DataFrame()

def fetch_stats(team_id, season=2023):
    """Ejemplo: Obtener estadísticas históricas detalladas por equipo"""
    url = f"https://v3.football.api-sports.io/teams/statistics?season={season}&team={team_id}"
    try:
        resp = requests.get(url, headers=HEADERS_FOOTBALL, timeout=10)
        resp.raise_for_status()
        return resp.json()  # Se estructura después para features
    except Exception as e:
        print(f"Error fetching stats for team {team_id}: {e}")
        return {}
