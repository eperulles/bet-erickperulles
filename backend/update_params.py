"""
Historical Data Pipeline -- BetIQ
=================================
Downloads real match data from:
  - football-data.co.uk (top 5 EU leagues) — free CSV, no key required
  - API-Football (all other leagues) — free, 100 req/day

Applies exponential time-weighting and fits Dixon-Coles parameters
via Maximum Likelihood Estimation using scipy.optimize.

Output: backend/team_params.json  (loaded automatically by main.py)
"""

import json
import math
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from api_client import download_csv_cached, fetch_apifb_fixtures

warnings.filterwarnings("ignore")

# ─── League config ────────────────────────────────────────────────────────────
# source: "fd" = football-data.co.uk, "apifb" = API-Football
LEAGUES_CONFIG = {
    # Top 5 EU — use football-data.co.uk (no API key, generous)
    "premier_league": {
        "name": "Premier League", "country": "England",
        "home_adv_init": 0.25, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["E0"],
        "seasons": ["2425", "2324", "2223"],
    },
    "la_liga": {
        "name": "La Liga", "country": "Spain",
        "home_adv_init": 0.22, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["SP1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "serie_a": {
        "name": "Serie A", "country": "Italy",
        "home_adv_init": 0.20, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["I1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "bundesliga": {
        "name": "Bundesliga", "country": "Germany",
        "home_adv_init": 0.28, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["D1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "ligue_1": {
        "name": "Ligue 1", "country": "France",
        "home_adv_init": 0.23, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["F1"],
        "seasons": ["2425", "2324", "2223"],
    },
    # Extended EU — football-data.co.uk also covers these
    "eredivisie": {
        "name": "Eredivisie", "country": "Netherlands",
        "home_adv_init": 0.24, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["N1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "liga_portugal": {
        "name": "Liga Portugal", "country": "Portugal",
        "home_adv_init": 0.26, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["P1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "super_lig": {
        "name": "Super Lig", "country": "Turkey",
        "home_adv_init": 0.30, "market_margin": 0.05,
        "source": "fd", "fd_keys": ["T1"],
        "seasons": ["2425", "2324", "2223"],
    },
    # Americas — use API-Football (no CSV available)
    "liga_mx": {
        "name": "Liga MX", "country": "Mexico",
        "home_adv_init": 0.35, "market_margin": 0.05,
        "source": "apifb",
        "apifb_seasons": [2025, 2024, 2023],
    },
    "mls": {
        "name": "MLS", "country": "USA",
        "home_adv_init": 0.28, "market_margin": 0.05,
        "source": "apifb",
        "apifb_seasons": [2025, 2024, 2023],
    },
    "serie_a_brazil": {
        "name": "Serie A", "country": "Brazil",
        "home_adv_init": 0.35, "market_margin": 0.05,
        "source": "apifb",
        "apifb_seasons": [2025, 2024, 2023],
    },
    "primera_argentina": {
        "name": "Primera Division", "country": "Argentina",
        "home_adv_init": 0.38, "market_margin": 0.05,
        "source": "apifb",
        "apifb_seasons": [2025, 2024, 2023],
    },
}

DECAY_THETA = 0.0019  # ln(2)/365
OUTPUT_PATH = Path(__file__).parent / "team_params.json"


# ─── Parse football-data.co.uk CSVs ─────────────────────────────────────────

def parse_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize columns from football-data.co.uk CSV."""
    col_map = {
        "Date": "date",
        "HomeTeam": "home_team", "Home": "home_team",
        "AwayTeam": "away_team", "Away": "away_team",
        "FTHG": "home_goals", "HG": "home_goals",
        "FTAG": "away_goals", "AG": "away_goals",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    df = df[required].dropna()
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna()
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)

    for fmt in ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"]:
        try:
            df["date"] = pd.to_datetime(df["date"], format=fmt)
            break
        except Exception:
            continue
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def compute_weights(dates: pd.Series, theta: float = DECAY_THETA) -> np.ndarray:
    today = pd.Timestamp.now()
    days_ago = (today - dates).dt.days.clip(lower=0).values
    return np.exp(-theta * days_ago)


# ─── Dixon-Coles MLE ────────────────────────────────────────────────────────

def dc_tau(x, y, lam_h, lam_a, rho):
    if x == 0 and y == 0: return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0: return 1 + lam_a * rho
    if x == 0 and y == 1: return 1 + lam_h * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def neg_log_likelihood(params, df, teams, weights):
    n = len(teams)
    attack  = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ll = 0.0
    for i, row in enumerate(df.itertuples()):
        ht, at = row.home_team, row.away_team
        if ht not in attack or at not in attack:
            continue
        lam_h = math.exp(attack[ht] + defense[at] + home_adv)
        lam_a = math.exp(attack[at] + defense[ht])
        hg, ag = row.home_goals, row.away_goals
        tau = dc_tau(hg, ag, lam_h, lam_a, rho)
        if tau <= 0:
            continue
        ll += weights[i] * (
            math.log(tau)
            + hg * math.log(lam_h) - lam_h - sum(math.log(k) for k in range(1, hg+1))
            + ag * math.log(lam_a) - lam_a - sum(math.log(k) for k in range(1, ag+1))
        )
    return -ll


def fit_dixon_coles(df, home_adv_init=0.25):
    recent_matches = df.tail(200)
    active_teams = set(recent_matches["home_team"].unique()) | set(recent_matches["away_team"].unique())
    df = df[df["home_team"].isin(active_teams) & df["away_team"].isin(active_teams)].copy()

    teams = sorted(list(active_teams))
    n = len(teams)
    if n < 4:
        return None

    weights = compute_weights(df["date"])

    rng = np.random.default_rng(42)
    x0 = np.concatenate([
        rng.uniform(0, 0.1, n),
        rng.uniform(-0.1, 0, n),
        [home_adv_init],
        [-0.1],
    ])

    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]

    print(f"  [FIT] Optimizando Dixon-Coles para {n} equipos, {len(df)} partidos...")
    result = minimize(
        neg_log_likelihood, x0, args=(df, teams, weights),
        method="L-BFGS-B", constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success and result.fun > 1e8:
        print(f"  [WARN] Optimizacion no convergio: {result.message}")

    params = result.x
    attack  = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))

    return {
        "teams": {t: {"attack": round(float(attack[t]), 4),
                       "defense": round(float(defense[t]), 4)} for t in teams},
        "home_adv": round(float(params[2*n]), 4),
        "rho": round(float(params[2*n + 1]), 4),
    }


# ─── Data Fetchers ──────────────────────────────────────────────────────────

def _fetch_fd_data(cfg):
    """Fetch data from football-data.co.uk."""
    all_dfs = []
    for season in cfg["seasons"]:
        for code in cfg["fd_keys"]:
            df_raw = download_csv_cached(season, code)
            if df_raw is not None:
                df_parsed = parse_matches(df_raw)
                if not df_parsed.empty:
                    all_dfs.append(df_parsed)
    return all_dfs


def _fetch_apifb_data(league_id, cfg):
    """Fetch data from API-Football."""
    all_dfs = []
    for season in cfg.get("apifb_seasons", []):
        df = fetch_apifb_fixtures(league_id, season)
        if df is not None and not df.empty:
            all_dfs.append(df)
    return all_dfs


# ─── Main ────────────────────────────────────────────────────────────────────

def update_all():
    print("=" * 60)
    print("  BetIQ -- Calibracion de parametros Dixon-Coles")
    print("  Fuentes: football-data.co.uk + API-Football")
    print("=" * 60)

    output = {}

    for league_id, cfg in LEAGUES_CONFIG.items():
        print(f"\n{'--'*30}")
        print(f"[DATA] {cfg['name']} ({cfg['country']}) [source: {cfg['source']}]")

        if cfg["source"] == "fd":
            all_dfs = _fetch_fd_data(cfg)
        elif cfg["source"] == "apifb":
            all_dfs = _fetch_apifb_data(league_id, cfg)
        else:
            all_dfs = []

        if not all_dfs:
            print(f"  [ERR] Sin datos para {cfg['name']} -- usando parametros por defecto")
            continue

        df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        ).sort_values("date")

        print(f"  [DATE] {df['date'].min().date()} -> {df['date'].max().date()} | {len(df)} partidos")

        fitted = fit_dixon_coles(df, cfg["home_adv_init"])
        if fitted is None:
            print(f"  [ERR] No se pudo ajustar el modelo")
            continue

        output[league_id] = {
            "name": cfg["name"],
            "country": cfg["country"],
            "home_adv": fitted["home_adv"],
            "rho": fitted["rho"],
            "market_margin": cfg["market_margin"],
            "teams": fitted["teams"],
            "updated_at": datetime.now().isoformat(),
            "num_matches": len(df),
        }
        print(f"  [OK] home_adv={fitted['home_adv']:.3f}  rho={fitted['rho']:.3f}")
        top = sorted(fitted["teams"].items(), key=lambda x: x[1]["attack"], reverse=True)[:3]
        for t, p in top:
            print(f"      {t}: ataque={p['attack']:.3f}, defensa={p['defense']:.3f}")

    if output:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"[OK] Parametros guardados en: {OUTPUT_PATH}")
        print(f"   Ligas calibradas: {list(output.keys())}")
        print(f"{'='*60}")
    else:
        print("\n[ERR] No se genero ningun archivo de parametros.")


if __name__ == "__main__":
    update_all()
