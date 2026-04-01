"""
Prediction tracking and evaluation for BetIQ.
Compares predictions vs actual results, computes accuracy metrics.
"""
import json
import numpy as np
import db
import api_client


def update_results_from_api():
    """Fetch recent results from The Odds API and save to database."""
    all_results = api_client.get_recent_results()
    saved = 0
    for league_id, events in all_results.items():
        for event in events:
            if not event.get("completed"):
                continue
            scores = event.get("scores")
            if not scores or len(scores) < 2:
                continue
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            home_score = away_score = None
            for s in scores:
                if s.get("name") == home_team:
                    home_score = int(s.get("score", 0))
                elif s.get("name") == away_team:
                    away_score = int(s.get("score", 0))
            if home_score is not None and away_score is not None:
                date_str = event.get("commence_time", "")[:10]
                db.save_result(league_id, home_team, away_team,
                               date_str, home_score, away_score)
                saved += 1
    return saved


def compute_metrics(league_id=None):
    """Compute prediction accuracy metrics."""
    data = db.get_predictions_with_results(league_id)
    if not data:
        return {"total": 0, "accuracy_1x2": 0, "brier_score": 0,
                "roi": 0, "total_staked": 0, "total_returns": 0}

    correct = 0
    brier_sum = 0.0
    total_staked = 0
    total_returns = 0.0

    for row in data:
        actual = row["result"]
        ph = row["prob_home"] or 0
        pd_ = row["prob_draw"] or 0
        pa = row["prob_away"] or 0

        # 1X2 accuracy
        predicted = max([("H", ph), ("D", pd_), ("A", pa)], key=lambda x: x[1])[0]
        if predicted == actual:
            correct += 1

        # Brier score
        actual_vec = [int(actual == "H"), int(actual == "D"), int(actual == "A")]
        pred_vec = [ph, pd_, pa]
        brier_sum += sum((p - a) ** 2 for p, a in zip(pred_vec, actual_vec))

        # ROI on value bets
        vb = json.loads(row["value_bets_json"]) if row["value_bets_json"] else []
        for bet in vb:
            total_staked += 1
            odds = bet.get("implied_odds", 0)
            sel = bet.get("selection", "")
            won = False
            tg = row["total_goals"]
            if "Local" in sel and actual == "H":
                won = True
            elif "Empate" in sel and actual == "D":
                won = True
            elif "Visitante" in sel and actual == "A":
                won = True
            elif "Más de 0.5" in sel and tg > 0:
                won = True
            elif "Menos de 0.5" in sel and tg == 0:
                won = True
            elif "Más de 1.5" in sel and tg > 1:
                won = True
            elif "Menos de 1.5" in sel and tg <= 1:
                won = True
            elif "Más de 2.5" in sel and tg > 2:
                won = True
            elif "Menos de 2.5" in sel and tg <= 2:
                won = True
            elif "Más de 3.5" in sel and tg > 3:
                won = True
            elif "Menos de 3.5" in sel and tg <= 3:
                won = True
            if won:
                total_returns += odds

    n = len(data)
    return {
        "total": n,
        "accuracy_1x2": round(correct / n * 100, 1) if n else 0,
        "brier_score": round(brier_sum / n, 4) if n else 0,
        "roi": round((total_returns - total_staked) / total_staked * 100, 1) if total_staked > 0 else 0,
        "total_staked": total_staked,
        "total_returns": round(total_returns, 2),
    }


def get_calibration_data(league_id=None, bins=10):
    """Predicted probability vs actual frequency for calibration plot."""
    data = db.get_predictions_with_results(league_id)
    if not data:
        return {"bins": [], "predicted": [], "actual": [], "counts": []}

    pairs = []
    for row in data:
        for key, res_code in [("prob_home", "H"), ("prob_draw", "D"), ("prob_away", "A")]:
            p = row[key] or 0
            a = 1 if row["result"] == res_code else 0
            pairs.append((p, a))

    bin_edges = np.linspace(0, 1, bins + 1)
    bin_predicted, bin_actual, bin_counts = [], [], []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(p, a) for p, a in pairs if lo <= p < hi]
        if in_bin:
            bin_predicted.append(round(np.mean([p for p, _ in in_bin]), 3))
            bin_actual.append(round(np.mean([a for _, a in in_bin]), 3))
            bin_counts.append(len(in_bin))
        else:
            bin_predicted.append(round((lo + hi) / 2, 3))
            bin_actual.append(0)
            bin_counts.append(0)

    return {
        "bins": [f"{int(bin_edges[i]*100)}-{int(bin_edges[i+1]*100)}%" for i in range(bins)],
        "predicted": bin_predicted,
        "actual": bin_actual,
        "counts": bin_counts,
    }


def get_recent_predictions_with_results(limit=50):
    """Recent predictions with their actual results for display."""
    return db.get_predictions_with_results(limit=limit)


def get_accuracy_by_league():
    """Accuracy breakdown by league."""
    conn = db.get_conn()
    try:
        leagues = conn.execute("SELECT DISTINCT league_id FROM predictions").fetchall()
        return {row["league_id"]: compute_metrics(row["league_id"]) for row in leagues}
    finally:
        conn.close()
