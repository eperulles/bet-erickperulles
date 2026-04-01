"""
Historical Data Pipeline — BetIQ
=================================
Downloads real match data from football-data.co.uk,
applies exponential time-weighting and fits Dixon-Coles parameters
via Maximum Likelihood Estimation using scipy.optimize.

Run this script to calibrate team parameters from real data:
    python update_params.py

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

from api_client import download_csv_cached

warnings.filterwarnings("ignore")

# ─── League config ────────────────────────────────────────────────────────────
# football-data.co.uk season format: "2425" = 2024/25, "2324" = 2023/24
LEAGUES_CONFIG = {
    "premier_league": {
        "name": "Premier League",
        "country": "England",
        "home_adv_init": 0.25,
        "market_margin": 0.05,
        "fd_keys": ["E0"],          # football-data.co.uk league codes
        "seasons": ["2425", "2324", "2223"],
    },
    "la_liga": {
        "name": "La Liga",
        "country": "Spain",
        "home_adv_init": 0.22,
        "market_margin": 0.05,
        "fd_keys": ["SP1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "serie_a": {
        "name": "Serie A",
        "country": "Italy",
        "home_adv_init": 0.20,
        "market_margin": 0.05,
        "fd_keys": ["I1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "home_adv_init": 0.28,
        "market_margin": 0.05,
        "fd_keys": ["D1"],
        "seasons": ["2425", "2324", "2223"],
    },
    "ligue_1": {
        "name": "Ligue 1",
        "country": "France",
        "home_adv_init": 0.23,
        "market_margin": 0.05,
        "fd_keys": ["F1"],
        "seasons": ["2425", "2324", "2223"],
    },
}

# Exponential decay: matches 365 days old get ~50% weight
DECAY_THETA = 0.0019  # ln(2)/365

OUTPUT_PATH = Path(__file__).parent / "team_params.json"


def parse_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns from football-data.co.uk CSV.
    Returns DataFrame with: date, home_team, away_team, home_goals, away_goals
    """
    col_map = {
        # Date columns
        "Date": "date",
        # Team columns
        "HomeTeam": "home_team", "Home": "home_team",
        "AwayTeam": "away_team", "Away": "away_team",
        # Goals
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

    # Parse date
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
    """Exponential time decay: w = exp(-theta * days_ago)"""
    today = pd.Timestamp.now()
    days_ago = (today - dates).dt.days.clip(lower=0).values
    return np.exp(-theta * days_ago)


# ─── Dixon-Coles MLE ──────────────────────────────────────────────────────────

def dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles correction factor for low scores."""
    if x == 0 and y == 0:
        return 1 - lam_h * lam_a * rho
    if x == 1 and y == 0:
        return 1 + lam_a * rho
    if x == 0 and y == 1:
        return 1 + lam_h * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def neg_log_likelihood(params: np.ndarray, df: pd.DataFrame, teams: list, weights: np.ndarray) -> float:
    """
    Negative weighted log-likelihood for Dixon-Coles model.
    params layout: [attack_0..n, defense_0..n, home_adv, rho]
    """
    n = len(teams)
    attack  = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))
    home_adv = params[2*n]
    rho      = params[2*n + 1]

    ll = 0.0
    for i, row in enumerate(df.itertuples()):
        ht = row.home_team
        at = row.away_team
        if ht not in attack or at not in attack:
            continue
        lam_h = math.exp(attack[ht] + defense[at] + home_adv)
        lam_a = math.exp(attack[at] + defense[ht])
        hg = row.home_goals
        ag = row.away_goals
        tau = dc_tau(hg, ag, lam_h, lam_a, rho)
        if tau <= 0:
            continue
        ll += weights[i] * (
            math.log(tau)
            + hg * math.log(lam_h) - lam_h - sum(math.log(k) for k in range(1, hg+1))
            + ag * math.log(lam_a) - lam_a - sum(math.log(k) for k in range(1, ag+1))
        )
    return -ll


def fit_dixon_coles(df: pd.DataFrame, home_adv_init: float = 0.25) -> dict | None:
    """
    Fit Dixon-Coles parameters via MLE.
    Only keeps teams that appear in the current season (2425 matches are at the end of the sorted df).
    Returns dict: {team: {attack, defense}, home_adv, rho}
    """
    # Find teams active in the last 150 matches (proxy for current season)
    recent_matches = df.tail(200)
    active_teams = set(recent_matches["home_team"].unique()) | set(recent_matches["away_team"].unique())
    
    # Filter dataframe to only include matches where BOTH teams are currently active
    df = df[df["home_team"].isin(active_teams) & df["away_team"].isin(active_teams)].copy()
    
    teams = sorted(list(active_teams))
    n = len(teams)
    if n < 4:
        return None

    weights = compute_weights(df["date"])

    # Initial params: small random values
    rng = np.random.default_rng(42)
    x0 = np.concatenate([
        rng.uniform(0, 0.1, n),   # attack
        rng.uniform(-0.1, 0, n),  # defense
        [home_adv_init],           # home advantage
        [-0.1],                    # rho
    ])

    # Constraint: sum of attacks = 0 (model identifiability)
    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]

    print(f"  [FIT] Optimizando Dixon-Coles para {n} equipos, {len(df)} partidos...")
    result = minimize(
        neg_log_likelihood,
        x0,
        args=(df, teams, weights),
        method="L-BFGS-B",
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    if not result.success and result.fun > 1e8:
        print(f"  [WARN]  Optimización no convergió completamente: {result.message}")

    params = result.x
    attack  = dict(zip(teams, params[:n]))
    defense = dict(zip(teams, params[n:2*n]))
    home_adv = float(params[2*n])
    rho      = float(params[2*n + 1])

    return {
        "teams": {t: {"attack": round(float(attack[t]), 4), "defense": round(float(defense[t]), 4)} for t in teams},
        "home_adv": round(home_adv, 4),
        "rho": round(rho, 4),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def update_all():
    print("=" * 60)
    print("  BetIQ — Calibración de parámetros Dixon-Coles")
    print("  Fuente: football-data.co.uk")
    print("=" * 60)

    output = {}

    for league_id, cfg in LEAGUES_CONFIG.items():
        print(f"\n{'─'*60}")
        print(f"[DATA] {cfg['name']} ({cfg['country']})")

        all_dfs = []
        for season in cfg["seasons"]:
            for code in cfg["fd_keys"]:
                df_raw = download_csv_cached(season, code)
                if df_raw is not None:
                    df_parsed = parse_matches(df_raw)
                    if not df_parsed.empty:
                        all_dfs.append(df_parsed)

        if not all_dfs:
            print(f"  [ERR] Sin datos para {cfg['name']} — usando parámetros por defecto")
            continue

        df = pd.concat(all_dfs, ignore_index=True).drop_duplicates(
            subset=["date", "home_team", "away_team"]
        ).sort_values("date")

        print(f"  [DATE] {df['date'].min().date()} → {df['date'].max().date()} | {len(df)} partidos totales")

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
        print(f"  [TOP] Top 3 equipos por ataque:")
        top = sorted(fitted["teams"].items(), key=lambda x: x[1]["attack"], reverse=True)[:3]
        for t, p in top:
            print(f"      {t}: ataque={p['attack']:.3f}, defensa={p['defense']:.3f}")

    if output:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n{'='*60}")
        print(f"[OK] Parámetros guardados en: {OUTPUT_PATH}")
        print(f"   Ligas calibradas: {list(output.keys())}")
        print(f"{'='*60}")
    else:
        print("\n[ERR] No se generó ningún archivo de parámetros.")


if __name__ == "__main__":
    update_all()
