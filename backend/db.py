"""
SQLite database for BetIQ — prediction tracking, results, odds caching.
"""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "betiq.db"


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            league_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            match_date TEXT,
            prob_home REAL,
            prob_draw REAL,
            prob_away REAL,
            lambda_home REAL,
            lambda_away REAL,
            prob_over_25 REAL,
            prob_under_25 REAL,
            odds_home REAL,
            odds_draw REAL,
            odds_away REAL,
            edge_best REAL,
            value_bets_json TEXT,
            ml_adjusted INTEGER DEFAULT 0,
            ml_prob_home REAL,
            ml_prob_draw REAL,
            ml_prob_away REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id TEXT UNIQUE,
            league_id TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            match_date TEXT,
            home_goals INTEGER,
            away_goals INTEGER,
            result TEXT,
            total_goals INTEGER,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS odds_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            sport_key TEXT,
            home_team TEXT,
            away_team TEXT,
            commence_time TEXT,
            odds_json TEXT,
            fetched_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_pred_league ON predictions(league_id);
        CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(match_date);
        CREATE INDEX IF NOT EXISTS idx_res_match ON results(match_id);
    """)
    conn.commit()
    conn.close()


def _match_id(date_str: str, home: str, away: str) -> str:
    d = date_str[:10] if date_str else "unknown"
    return f"{d}_{home}_{away}".replace(" ", "_").lower()


def save_prediction(league_id, home_team, away_team, match_date,
                    prediction, odds=None, ml_probs=None):
    mid = _match_id(match_date, home_team, away_team)
    conn = get_conn()
    try:
        p = prediction.get("probs_1x2", {})
        ou = prediction.get("ou_markets", {}).get("ou_25", {})
        vb = prediction.get("value_bets", [])
        best_edge = max((b["edge"] for b in vb), default=0)
        conn.execute("""
            INSERT OR REPLACE INTO predictions
            (match_id, league_id, home_team, away_team, match_date,
             prob_home, prob_draw, prob_away, lambda_home, lambda_away,
             prob_over_25, prob_under_25, odds_home, odds_draw, odds_away,
             edge_best, value_bets_json, ml_adjusted,
             ml_prob_home, ml_prob_draw, ml_prob_away)
            VALUES (?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?,?,?, ?,?,?)
        """, (
            mid, league_id, home_team, away_team, match_date,
            p.get("home"), p.get("draw"), p.get("away"),
            prediction.get("lambda_home"), prediction.get("lambda_away"),
            ou.get("over"), ou.get("under"),
            odds.get("home") if odds else None,
            odds.get("draw") if odds else None,
            odds.get("away") if odds else None,
            best_edge, json.dumps(vb),
            1 if ml_probs else 0,
            ml_probs.get("home") if ml_probs else None,
            ml_probs.get("draw") if ml_probs else None,
            ml_probs.get("away") if ml_probs else None,
        ))
        conn.commit()
        return mid
    finally:
        conn.close()


def save_result(league_id, home_team, away_team, match_date, home_goals, away_goals):
    mid = _match_id(match_date, home_team, away_team)
    if home_goals > away_goals:
        result = "H"
    elif home_goals == away_goals:
        result = "D"
    else:
        result = "A"
    conn = get_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO results
            (match_id, league_id, home_team, away_team, match_date,
             home_goals, away_goals, result, total_goals)
            VALUES (?,?,?,?,?, ?,?,?,?)
        """, (mid, league_id, home_team, away_team, match_date,
              home_goals, away_goals, result, home_goals + away_goals))
        conn.commit()
        return mid
    finally:
        conn.close()


def get_predictions_with_results(league_id=None, limit=500):
    conn = get_conn()
    try:
        q = """SELECT p.*, r.home_goals, r.away_goals, r.result, r.total_goals
               FROM predictions p
               INNER JOIN results r ON p.match_id = r.match_id"""
        params = []
        if league_id:
            q += " WHERE p.league_id = ?"
            params.append(league_id)
        q += " ORDER BY p.match_date DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in conn.execute(q, params).fetchall()]
    finally:
        conn.close()


def get_pending_predictions(limit=100):
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT p.* FROM predictions p
            LEFT JOIN results r ON p.match_id = r.match_id
            WHERE r.id IS NULL
            ORDER BY p.match_date DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_cached_odds(sport_key, max_age_hours=6):
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT * FROM odds_cache
            WHERE sport_key = ?
            AND fetched_at > datetime('now', ?)
        """, (sport_key, f"-{max_age_hours} hours")).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def save_odds_cache(events, sport_key):
    conn = get_conn()
    try:
        for ev in events:
            conn.execute("""
                INSERT OR REPLACE INTO odds_cache
                (event_id, sport_key, home_team, away_team, commence_time, odds_json)
                VALUES (?,?,?,?,?,?)
            """, (ev.get("id", ""), sport_key,
                  ev.get("home_team", ""), ev.get("away_team", ""),
                  ev.get("commence_time", ""), json.dumps(ev)))
        conn.commit()
    finally:
        conn.close()


def get_prediction_count():
    conn = get_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        matched = conn.execute("""
            SELECT COUNT(*) FROM predictions p
            INNER JOIN results r ON p.match_id = r.match_id
        """).fetchone()[0]
        return {"total_predictions": total, "matched_with_results": matched,
                "pending": total - matched}
    finally:
        conn.close()
