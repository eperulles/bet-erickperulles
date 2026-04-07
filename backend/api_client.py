"""
Unified API client for BetIQ.
- Historical data: football-data.co.uk (top 5 EU) + API-Football (all leagues)
- Upcoming fixtures + odds: The Odds API
- Match scores: The Odds API scores endpoint + API-Football
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

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

FD_BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

# ─── League Mappings ─────────────────────────────────────────────────────────
# The Odds API sport keys
LEAGUE_TO_SPORT = {
    "premier_league":   "soccer_epl",
    "la_liga":          "soccer_spain_la_liga",
    "serie_a":          "soccer_italy_serie_a",
    "bundesliga":       "soccer_germany_bundesliga",
    "ligue_1":          "soccer_france_ligue_one",
    "eredivisie":       "soccer_netherlands_eredivisie",
    "liga_portugal":    "soccer_portugal_primeira_liga",
    "liga_mx":          "soccer_mexico_ligamx",
    "mls":              "soccer_usa_mls",
    "serie_a_brazil":   "soccer_brazil_campeonato",
    "primera_argentina":"soccer_argentina_primera_division",
    "super_lig":        "soccer_turkey_super_league",
    "champions_league": "soccer_uefa_champs_league",
    "europa_league":    "soccer_uefa_europa_league",
}
SPORT_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_SPORT.items()}

# API-Football league IDs
LEAGUE_TO_APIFB = {
    "premier_league":    39,
    "la_liga":           140,
    "serie_a":           135,
    "bundesliga":        78,
    "ligue_1":           61,
    "eredivisie":        88,
    "liga_portugal":     94,
    "liga_mx":           262,
    "mls":               253,
    "serie_a_brazil":    71,
    "primera_argentina": 128,
    "super_lig":         203,
    "champions_league":  2,
    "europa_league":     3,
}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/csv,text/plain,*/*",
}


# ─── Historical Data (football-data.co.uk) ──────────────────────────────────

def download_csv_cached(season: str, code: str, max_age_days: int = 7) -> pd.DataFrame | None:
    """Download CSV with local file caching and retry logic."""
    cache_file = CACHE_DIR / f"{code}_{season}.csv"

    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < max_age_days:
            try:
                df = pd.read_csv(cache_file, on_bad_lines="skip")
                if len(df) > 0:
                    print(f"  [CACHE] Cache: {cache_file.name} -> {len(df)} partidos")
                    return df
            except Exception:
                pass

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
                            print(f"  [OK] {url} -> {len(df)} partidos")
                            return df
                    except Exception:
                        continue
                print(f"  [WARN] No se pudo parsear: {url}")
                return None
            elif resp.status_code == 404:
                print(f"  [WARN] No encontrado: {url}")
                return None
            else:
                print(f"  [WARN] HTTP {resp.status_code}: {url}")
        except requests.exceptions.Timeout:
            print(f"  [WAIT] Timeout (intento {attempt+1}/3): {url}")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [ERR] Error (intento {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, on_bad_lines="skip")
            if len(df) > 0:
                print(f"  [CACHE] Cache antiguo: {cache_file.name} -> {len(df)} partidos")
                return df
        except Exception:
            pass
    return None


# ─── API-Football — Historical Fixtures ─────────────────────────────────────

def _apifb_headers():
    return {"x-apisports-key": API_FOOTBALL_KEY}


def fetch_apifb_fixtures(league_id: str, season: int, max_age_days: int = 7) -> pd.DataFrame | None:
    """
    Fetch completed fixtures from API-Football for a given league/season.
    Returns DataFrame with: date, home_team, away_team, home_goals, away_goals
    """
    apifb_id = LEAGUE_TO_APIFB.get(league_id)
    if not apifb_id or not API_FOOTBALL_KEY:
        return None

    cache_file = CACHE_DIR / f"apifb_{league_id}_{season}.json"

    # Use cache if fresh
    if cache_file.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age_days < max_age_days:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data:
                    df = _apifb_to_dataframe(data)
                    if df is not None and not df.empty:
                        print(f"  [CACHE] API-Football cache: {league_id} {season} -> {len(df)} partidos")
                        return df
            except Exception:
                pass

    # Fetch from API
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": apifb_id, "season": season, "status": "FT-AET-PEN"}
    try:
        resp = requests.get(url, headers=_apifb_headers(), params=params, timeout=20)
        if resp.status_code == 200:
            body = resp.json()
            remaining = body.get("errors", {})
            if remaining:
                print(f"  [WARN] API-Football errors: {remaining}")
                return None
            fixtures = body.get("response", [])
            if fixtures:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(fixtures, f, ensure_ascii=False)
                df = _apifb_to_dataframe(fixtures)
                if df is not None and not df.empty:
                    print(f"  [OK] API-Football: {league_id} {season} -> {len(df)} partidos")
                    return df
            else:
                print(f"  [WARN] API-Football: sin fixtures para {league_id} {season}")
        else:
            print(f"  [WARN] API-Football HTTP {resp.status_code}")
        return None
    except Exception as e:
        print(f"  [ERR] API-Football error: {e}")
        # Fallback to stale cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return _apifb_to_dataframe(data)
            except Exception:
                pass
        return None


def _apifb_to_dataframe(fixtures: list) -> pd.DataFrame | None:
    """Convert API-Football fixtures response to standard DataFrame."""
    rows = []
    for fix in fixtures:
        goals = fix.get("goals", {})
        teams = fix.get("teams", {})
        fixture_info = fix.get("fixture", {})
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            continue
        rows.append({
            "date": fixture_info.get("date", "")[:10],
            "home_team": teams.get("home", {}).get("name", ""),
            "away_team": teams.get("away", {}).get("name", ""),
            "home_goals": int(hg),
            "away_goals": int(ag),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def fetch_apifb_upcoming(league_id: str, next_n: int = 15) -> list:
    """Fetch next N upcoming fixtures from API-Football."""
    apifb_id = LEAGUE_TO_APIFB.get(league_id)
    if not apifb_id or not API_FOOTBALL_KEY:
        return []

    cache_file = CACHE_DIR / f"apifb_upcoming_{league_id}.json"

    # Use cache if < 6 hours old
    if cache_file.exists():
        age_hours = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() / 3600
        if age_hours < 6:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": apifb_id, "next": next_n}
    try:
        resp = requests.get(url, headers=_apifb_headers(), params=params, timeout=15)
        if resp.status_code == 200:
            body = resp.json()
            fixtures = body.get("response", [])
            if fixtures:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(fixtures, f, ensure_ascii=False)
                print(f"  [OK] API-Football upcoming: {league_id} -> {len(fixtures)} fixtures")
            return fixtures
        return []
    except Exception as e:
        print(f"  [ERR] API-Football upcoming error: {e}")
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []


def fetch_apifb_results(league_id: str, last_n: int = 15) -> list:
    """Fetch last N completed fixtures from API-Football for result tracking."""
    apifb_id = LEAGUE_TO_APIFB.get(league_id)
    if not apifb_id or not API_FOOTBALL_KEY:
        return []
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": apifb_id, "last": last_n}
    try:
        resp = requests.get(url, headers=_apifb_headers(), params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json().get("response", [])
        return []
    except Exception as e:
        print(f"  [ERR] API-Football results error: {e}")
        return []


# ─── The Odds API — Upcoming Odds ───────────────────────────────────────────

def fetch_upcoming_odds(sport_key: str, use_cache: bool = True) -> list:
    """Fetch upcoming fixtures with odds. Uses cache if < 6 hours old."""
    if use_cache:
        cached = db.get_cached_odds(sport_key, max_age_hours=6)
        if cached:
            print(f"  [CACHE] Odds cache: {sport_key} -> {len(cached)} eventos")
            return [json.loads(c["odds_json"]) for c in cached]

    if not ODDS_API_KEY:
        print("  [WARN] ODDS_API_KEY no configurada")
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
            print(f"  [OK] Odds API: {sport_key} -> {len(events)} eventos (quota: {remaining})")
            db.save_odds_cache(events, sport_key)
            return events
        elif resp.status_code == 401:
            print("  [ERR] Odds API: key invalida")
        elif resp.status_code == 429:
            print("  [ERR] Odds API: quota excedida -- usando cache")
            cached = db.get_cached_odds(sport_key, max_age_hours=168)
            if cached:
                return [json.loads(c["odds_json"]) for c in cached]
        else:
            print(f"  [WARN] Odds API HTTP {resp.status_code}")
        return []
    except Exception as e:
        print(f"  [ERR] Error Odds API: {e}")
        return []


# ─── The Odds API — Scores ──────────────────────────────────────────────────

def fetch_scores(sport_key: str, days_from: int = 3) -> list:
    """Fetch recent completed match scores from The Odds API."""
    if not ODDS_API_KEY:
        return []
    url = f"{ODDS_API_BASE}/sports/{sport_key}/scores"
    params = {"apiKey": ODDS_API_KEY, "daysFrom": days_from, "dateFormat": "iso"}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            events = resp.json()
            completed = [e for e in events if e.get("completed")]
            print(f"  [OK] Scores: {sport_key} -> {len(completed)} completados")
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


def get_recent_results_apifb() -> dict:
    """Get recent results via API-Football (uses fewer Odds API credits)."""
    results = {}
    for league_id in LEAGUE_TO_APIFB:
        fixtures = fetch_apifb_results(league_id, last_n=10)
        if fixtures:
            results[league_id] = fixtures
    return results
