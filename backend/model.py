"""
Dixon-Coles model for football match probability prediction.
All computations use NumPy for speed.
"""

import numpy as np
from math import exp, factorial
from typing import Dict, Tuple


def poisson_pmf(lam: float, k: int) -> float:
    """Poisson probability mass function."""
    return exp(-lam) * (lam ** k) / factorial(k)


def dc_correction(i: int, j: int, lam_h: float, lam_a: float, rho: float = -0.13) -> float:
    """
    Dixon-Coles low-score correction factor.
    Adjusts joint probability for scores 0-0, 1-0, 0-1, 1-1.
    rho = -0.13 is the standard empirically estimated value.
    """
    if i == 0 and j == 0:
        return 1 - lam_h * lam_a * rho
    elif i == 1 and j == 0:
        return 1 + lam_a * rho
    elif i == 0 and j == 1:
        return 1 + lam_h * rho
    elif i == 1 and j == 1:
        return 1 - rho
    else:
        return 1.0


def compute_lambdas(
    home_attack: float, home_defense: float,
    away_attack: float, away_defense: float,
    home_adv: float
) -> Tuple[float, float]:
    """
    Compute expected goals for each team.
    lambda_home = exp(attack_h + defense_a + home_adv)
    lambda_away = exp(attack_a + defense_h)
    """
    lam_home = exp(home_attack + away_defense + home_adv)
    lam_away = exp(away_attack + home_defense)
    return lam_home, lam_away


def build_score_matrix(lam_home: float, lam_away: float, max_goals: int = 7) -> np.ndarray:
    """
    Build joint probability matrix P(X=i, Y=j) with Dixon-Coles correction.
    Returns matrix of shape (max_goals+1, max_goals+1).
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_home = poisson_pmf(lam_home, i)
            p_away = poisson_pmf(lam_away, j)
            dc = dc_correction(i, j, lam_home, lam_away)
            matrix[i][j] = p_home * p_away * dc
    # Normalize to ensure sum = 1
    matrix = matrix / matrix.sum()
    return matrix


def compute_1x2(matrix: np.ndarray) -> Dict[str, float]:
    """Compute 1X2 probabilities from score matrix."""
    home_win = float(np.sum(np.tril(matrix, k=-1)))  # i > j → rows below diagonal
    draw = float(np.sum(np.diag(matrix)))
    away_win = float(np.sum(np.triu(matrix, k=1)))   # j > i → rows above diagonal
    total = home_win + draw + away_win
    return {
        "home": round(home_win / total, 4),
        "draw": round(draw / total, 4),
        "away": round(away_win / total, 4),
    }


def compute_total_goals_dist(matrix: np.ndarray) -> Dict[int, float]:
    """
    Compute P(Total Goals = k) for k = 0, 1, 2, ..., 2*max_goals
    by summing anti-diagonals of the score matrix.
    """
    n = matrix.shape[0]
    max_total = (n - 1) * 2
    dist = {}
    for total in range(max_total + 1):
        prob = 0.0
        for i in range(n):
            j = total - i
            if 0 <= j < n:
                prob += matrix[i][j]
        dist[total] = prob
    return dist


def compute_ou_markets(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute Over/Under probabilities for 0.5, 1.5, 2.5, 3.5 (full time).
    """
    dist = compute_total_goals_dist(matrix)
    markets = {}
    for threshold in [0.5, 1.5, 2.5, 3.5]:
        k = int(threshold - 0.5)  # 0, 1, 2, 3
        under = sum(dist.get(i, 0) for i in range(k + 1))
        over = 1.0 - under
        key = f"ou_{int(threshold*10)}"  # "ou_05", "ou_15", etc.
        markets[key] = {
            "label": f"O/U {threshold}",
            "over": round(over, 4),
            "under": round(under, 4),
        }
    return markets


def compute_ht_ou_markets(lam_home: float, lam_away: float) -> Dict[str, Dict[str, float]]:
    """
    Estimate half-time Over/Under using lambda/2 approximation.
    Covers 0.5 and 1.5 for HT market.
    """
    lam_h_ht = lam_home / 2.0
    lam_a_ht = lam_away / 2.0
    matrix_ht = build_score_matrix(lam_h_ht, lam_a_ht, max_goals=4)
    dist_ht = compute_total_goals_dist(matrix_ht)
    markets = {}
    for threshold in [0.5, 1.5]:
        k = int(threshold - 0.5)
        under = sum(dist_ht.get(i, 0) for i in range(k + 1))
        over = 1.0 - under
        key = f"ht_ou_{int(threshold*10)}"
        markets[key] = {
            "label": f"HT O/U {threshold}",
            "over": round(over, 4),
            "under": round(under, 4),
        }
    return markets


def remove_vig(raw_probs: Dict[str, float]) -> Dict[str, float]:
    """Remove bookmaker overround (vig) from implied probabilities."""
    total = sum(raw_probs.values())
    return {k: v / total for k, v in raw_probs.items()}


def implied_probs_from_odds(home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
    """Convert decimal odds to vig-free implied probabilities."""
    raw = {
        "home": 1.0 / home_odds,
        "draw": 1.0 / draw_odds,
        "away": 1.0 / away_odds,
    }
    return remove_vig(raw)


def compute_edge(model_prob: float, market_prob: float) -> float:
    """Edge = model probability - market implied probability."""
    return round(model_prob - market_prob, 4)


def get_value_bets(
    model_probs: Dict,
    market_implied: Dict,
    ou_markets: Dict,
    ht_markets: Dict,
    market_ou: Dict = None,
    min_edge: float = 0.01
) -> list:
    """
    Compare model probabilities vs market for all markets.
    Returns list of value bets with edge > min_edge.
    """
    value_bets = []

    # 1X2
    for outcome in ["home", "draw", "away"]:
        mp = model_probs.get(outcome, 0)
        mkt = market_implied.get(outcome, 0)
        edge = compute_edge(mp, mkt)
        if edge >= min_edge:
            label_map = {"home": "Local gana", "draw": "Empate", "away": "Visitante gana"}
            implied_odds = round(1 / mkt, 2) if mkt > 0 else 0
            value_bets.append({
                "market": "1X2",
                "selection": label_map[outcome],
                "model_prob": round(mp * 100, 1),
                "market_prob": round(mkt * 100, 1),
                "edge": round(edge * 100, 2),
                "implied_odds": implied_odds,
            })

    # Over/Under FT
    for key, data in ou_markets.items():
        threshold = data["label"]
        for side in ["over", "under"]:
            mp = data[side]
            if market_ou and key in market_ou:
                mkt = market_ou[key].get(side)
                if mkt:
                    edge = compute_edge(mp, mkt)
                    if edge >= min_edge:
                        value_bets.append({
                            "market": f"{threshold}",
                            "selection": f"{'Más de' if side == 'over' else 'Menos de'} {threshold.split()[-1]}",
                            "model_prob": round(mp * 100, 1),
                            "market_prob": round(mkt * 100, 1),
                            "edge": round(edge * 100, 2),
                            "implied_odds": round(1 / mkt, 2) if mkt > 0 else 0,
                        })

    # HT markets
    for key, data in ht_markets.items():
        threshold = data["label"]
        for side in ["over", "under"]:
            mp = data[side]
            mkt = None
            if market_ou and key in market_ou:
                mkt = market_ou[key].get(side)
            if mkt:
                edge = compute_edge(mp, mkt)
                if edge >= min_edge:
                    value_bets.append({
                        "market": f"{threshold}",
                        "selection": f"{'Más de' if side == 'over' else 'Menos de'} {threshold.split()[-1]}",
                        "model_prob": round(mp * 100, 1),
                        "market_prob": round(mkt * 100, 1),
                        "edge": round(edge * 100, 2),
                        "implied_odds": round(1 / mkt, 2) if mkt > 0 else 0,
                    })

    value_bets.sort(key=lambda x: x["edge"], reverse=True)
    return value_bets


def predict_match(
    home_attack: float, home_defense: float,
    away_attack: float, away_defense: float,
    home_adv: float,
    home_odds: float = None, draw_odds: float = None, away_odds: float = None,
) -> Dict:
    """
    Full match prediction pipeline.
    Returns all probabilities, markets, heatmap data, and value bets.
    """
    lam_home, lam_away = compute_lambdas(home_attack, home_defense, away_attack, away_defense, home_adv)
    matrix = build_score_matrix(lam_home, lam_away)
    probs_1x2 = compute_1x2(matrix)
    ou_markets = compute_ou_markets(matrix)
    ht_markets = compute_ht_ou_markets(lam_home, lam_away)

    # Market implied probs
    if home_odds and draw_odds and away_odds:
        market_implied = implied_probs_from_odds(home_odds, draw_odds, away_odds)
    else:
        # Simulate typical market odds (slight favorite home)
        market_implied = remove_vig({
            "home": 1.0 / max((1 / probs_1x2["home"]) * 1.055, 1.01),
            "draw": 1.0 / max((1 / probs_1x2["draw"]) * 1.055, 1.01),
            "away": 1.0 / max((1 / probs_1x2["away"]) * 1.055, 1.01),
        })

    # Heatmap: P(home_goals, away_goals) as percentage
    max_display = 6
    heatmap = []
    for i in range(max_display + 1):
        row = []
        for j in range(max_display + 1):
            row.append(round(float(matrix[i][j]) * 100, 2))
        heatmap.append(row)

    value_bets = get_value_bets(probs_1x2, market_implied, ou_markets, ht_markets)

    return {
        "lambda_home": round(lam_home, 3),
        "lambda_away": round(lam_away, 3),
        "probs_1x2": probs_1x2,
        "market_implied": {k: round(v, 4) for k, v in market_implied.items()},
        "ou_markets": ou_markets,
        "ht_markets": ht_markets,
        "heatmap": heatmap,
        "value_bets": value_bets,
    }
