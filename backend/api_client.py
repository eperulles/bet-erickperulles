"""
Unified API client for BetIQ.
- Historical data: football-data.co.uk with caching + retries
- Upcoming fixtures + odds: The Odds API with caching
- Match scores: The Odds API scores endpoint
"""
import os
import json
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime

import db

# ─── Config ──────────────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

FD_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

LEAGUE_TO_SPORT = {
    "premier_league": "soccer_epl",
    "la_liga":        "soccer_spain_la_liga",
    "serie_a":        "soccer_italy_serie_a",
    "bundesliga":     "soccer_germany_bundesliga",
    "ligue_1":        "soccer_france_ligue_one",
}
SPORT_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_SPORT.items()}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/csv,text/plain,*/*",
}


# ─── Historical Data (football-data.co.uk) ──────────────────────────────────

def download_csv_cached(season: str, code: str, max_age_days: int = 7) -> pd.DataFrame | None:
    """
    Download CSV with local file caching and retry logic.
    Only re-downloads if cache is older than max_age_days.
    """
    cache_file = CACHE_DIR / f"{code}_{season}.csv"

    # Use cache if fresh
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < max_age_days:
            try:
                df = pd.read_csv(cache_file, on_bad_lines="skip")
                if len(df) > 0:
                    print(f"  [CACHE] Cache: {cache_file.name} → {len(df)} partidos")
                    return df
            except Exception:
                pass

    # Download with retries
    url = FD_BASE_URL.format(season=season, code=code)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=20)
            if resp.status_code == 200:
                for enc in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        text = resp.content.decode(enc)
                        df = pd.read_csv(StringIO(text), on_bad_lines="skip")
                        if len(df) > 0:
                            cache_file.write_text(text, encoding="utf-8")
                            print(f"  [OK] {url} → {len(df)} partidos")
                            return df
                    except Exception:
                        continue
                print(f"  [WARN]  No se pudo parsear: {url}")
                return None
            elif resp.status_code == 404:
                print(f"  [WARN]  No encontrado: {url}")
                return None
            else:
                print(f"  [WARN]  HTTP {resp.status_code}: {url}")
        except requests.exceptions.Timeout:
            print(f"  [WAIT] Timeout (intento {attempt+1}/3): {url}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [ERR] Error (intento {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    # All retries failed — use stale cache if available
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, on_bad_lines="skip")
            if len(df) > 0:
                print(f"  [CACHE] Cache antiguo: {cache_file.name} → {len(df)} partidos")
                return df
        except Exception:
            pass
    return None


# ─── The Odds API — Upcoming Odds ───────────────────────────────────────────

def fetch_upcoming_odds(sport_key: str, use_cache: bool = True) -> list:
    """Fetch upcoming fixtures with odds. Uses cache if < 6 hours old."""
    if use_cache:
        cached = db.get_cached_odds(sport_key, max_age_hours=6)
        if cached:
            print(f"  [CACHE] Odds cache: {sport_key} → {len(cached)} eventos")
            return [json.loads(c["odds_json"]) for c in cached]

    if not ODDS_API_KEY:
        print("  [WARN]  ODDS_API_KEY no configurada")
        return []

    url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu,uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            events = resp.json()
            remaining = resp.headers.get("x-requests-remaining", "?")
            print(f"  [OK] Odds API: {sport_key} → {len(events)} eventos (quota: {remaining})")
            db.save_odds_cache(events, sport_key)
            return events
        elif resp.status_code == 401:
            print("  [ERR] Odds API: key inválida")
        elif resp.status_code == 429:
            print("  [ERR] Odds API: quota excedida — usando cache")
            cached = db.get_cached_odds(sport_key, max_age_hours=168)  # 7 days
            if cached:
                return [json.loads(c["odds_json"]) for c in cached]
        else:
            print(f"  [WARN]  Odds API HTTP {resp.status_code}")
        return []
    except Exception as e:
        print(f"  [ERR] Error Odds API: {e}")
        return []


# ─── The Odds API — Scores ──────────────────────────────────────────────────

def fetch_scores(sport_key: str, days_from: int = 3) -> list:
    """Fetch recent completed match scores."""
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores"
    params = {"apiKey": ODDS_API_KEY, "daysFrom": days_from, "dateFormat": "iso"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            events = resp.json()
            completed = [e for e in events if e.get("completed")]
            print(f"  [OK] Scores: {sport_key} → {len(completed)} completados")
            return completed
        return []
    except Exception as e:
        print(f"  [ERR] Error scores: {e}")
        return []


# ─── Aggregate Helpers ───────────────────────────────────────────────────────

def get_all_upcoming_fixtures() -> dict:
    """Get upcoming fixtures across all leagues. Returns {league_id: [events]}."""
    fixtures = {}
    for league_id, sport_key in LEAGUE_TO_SPORT.items():
        events = fetch_upcoming_odds(sport_key)
        if events:
            fixtures[league_id] = events
    return fixtures


def get_recent_results() -> dict:
    """Get recent match results across all leagues. Returns {league_id: [events]}."""
    results = {}
    for league_id, sport_key in LEAGUE_TO_SPORT.items():
        scores = fetch_scores(sport_key)
        if scores:
            results[league_id] = scores
    return results
