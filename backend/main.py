"""
FastAPI backend for BetIQ Dashboard v2.
- Real upcoming fixtures from The Odds API (no more invented matches)
- Dixon-Coles predictions with optional ML adjustment
- Prediction tracking and analysis endpoints
- Automated recalibration via scheduler
"""

import os
import json
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from data import LEAGUES as LEAGUES_HARDCODED
from model import predict_match
import db
import tracker
import ml_model
import api_client

# ─── Load parameters ──────────────────────────────────────────────────────────
PARAMS_FILE = Path(__file__).parent / "team_params.json"

def load_league_data() -> dict:
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, encoding="utf-8") as f:
            calibrated = json.load(f)
        merged = {}
        # Merge hardcoded + calibrated (calibrated wins)
        for league_id, hc in LEAGUES_HARDCODED.items():
            if league_id in calibrated:
                cal = calibrated[league_id]
                merged[league_id] = {
                    **hc,
                    "home_adv": cal["home_adv"],
                    "teams": cal["teams"],
                    "_source": "calibrado",
                    "_updated_at": cal.get("updated_at", ""),
                    "_num_matches": cal.get("num_matches", 0),
                }
            else:
                merged[league_id] = {**hc, "_source": "hardcoded"}
        # Add any league in calibrated but not in hardcoded (API-Football discovered)
        for league_id, cal in calibrated.items():
            if league_id not in merged:
                merged[league_id] = {
                    "name": cal["name"],
                    "country": cal["country"],
                    "home_adv": cal["home_adv"],
                    "market_margin": cal.get("market_margin", 0.05),
                    "avg_goals": 2.5,
                    "teams": cal["teams"],
                    "_source": "calibrado",
                    "_updated_at": cal.get("updated_at", ""),
                    "_num_matches": cal.get("num_matches", 0),
                }
        return merged
    return {k: {**v, "_source": "hardcoded"} for k, v in LEAGUES_HARDCODED.items()}


LEAGUES = load_league_data()

# ─── Scheduler ─────────────────────────────────────────────────────────────────
_scheduler = None

def _start_scheduler():
    global _scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        _scheduler = BackgroundScheduler()
        _scheduler.add_job(tracker.update_results_from_api,
                           'interval', hours=6, id='fetch_results')
        _scheduler.add_job(_run_recalibration,
                           'cron', day_of_week='sun', hour=3, id='recalibrate')
        _scheduler.add_job(ml_model.train_model,
                           'cron', day_of_week='mon', hour=4, id='train_ml')
        _scheduler.start()
        print("[OK] Scheduler iniciado (resultados c/6h, recalibración dom 3am, ML lun 4am)")
    except ImportError:
        print("[WARN]  APScheduler no instalado — ejecuta: pip install apscheduler")
    except Exception as e:
        print(f"[WARN]  Error scheduler: {e}")

def _stop_scheduler():
    if _scheduler:
        _scheduler.shutdown(wait=False)

def _run_recalibration():
    global LEAGUES
    import update_params
    update_params.update_all()
    LEAGUES = load_league_data()
    print("[OK] Parámetros recalibrados")


# ─── App lifecycle ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    print("[OK] Base de datos inicializada")
    _start_scheduler()
    yield
    _stop_scheduler()

app = FastAPI(title="BetIQ API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_team_params(league_id: str, team_name: str):
    teams = LEAGUES[league_id]["teams"]
    if team_name in teams:
        return teams[team_name]
    for name, params in teams.items():
        if team_name.lower() in name.lower() or name.lower() in team_name.lower():
            return params
    return None


def parse_best_odds(event: dict) -> Optional[dict]:
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None
    preferred = ["pinnacle", "bet365", "williamhill"]
    bm = None
    for pref in preferred:
        bm = next((b for b in bookmakers if pref in b["key"]), None)
        if bm:
            break
    if not bm:
        bm = bookmakers[0]
    markets = bm.get("markets", [])
    h2h = next((m for m in markets if m["key"] == "h2h"), None)
    if not h2h:
        return None
    outcomes = {o["name"]: o["price"] for o in h2h["outcomes"]}
    return {
        "home": outcomes.get(event["home_team"]),
        "draw": outcomes.get("Draw"),
        "away": outcomes.get(event["away_team"]),
    }


# ─── Routes: Core ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "version": "2.0.0"}


@app.get("/leagues")
def get_leagues():
    return [{"id": k, "name": v["name"], "country": v["country"]}
            for k, v in LEAGUES.items()]


@app.get("/teams/{league_id}")
def get_teams(league_id: str):
    if league_id not in LEAGUES:
        raise HTTPException(404, "League not found")
    return {"league": LEAGUES[league_id]["name"],
            "teams": sorted(LEAGUES[league_id]["teams"].keys())}


class PredictRequest(BaseModel):
    league_id: str
    home_team: str
    away_team: str
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None


@app.post("/predict")
def predict(req: PredictRequest):
    if req.league_id not in LEAGUES:
        raise HTTPException(404, "League not found")
    league = LEAGUES[req.league_id]
    hp = get_team_params(req.league_id, req.home_team)
    ap = get_team_params(req.league_id, req.away_team)
    if not hp:
        raise HTTPException(404, f"Team '{req.home_team}' not found")
    if not ap:
        raise HTTPException(404, f"Team '{req.away_team}' not found")

    result = predict_match(
        home_attack=hp["attack"], home_defense=hp["defense"],
        away_attack=ap["attack"], away_defense=ap["defense"],
        home_adv=league["home_adv"],
        home_odds=req.home_odds, draw_odds=req.draw_odds, away_odds=req.away_odds,
    )

    # ML adjustment
    odds_dict = None
    if req.home_odds and req.draw_odds and req.away_odds:
        odds_dict = {"home": req.home_odds, "draw": req.draw_odds, "away": req.away_odds}
    ml_probs = ml_model.adjust_probabilities(result, odds_dict)
    if ml_probs:
        result["ml_adjusted"] = ml_probs

    result["home_team"] = req.home_team
    result["away_team"] = req.away_team
    result["league"] = league["name"]
    return result


# ─── Routes: Value Picks (REAL fixtures only) ─────────────────────────────────

@app.get("/value-picks")
def value_picks(min_edge: float = Query(default=2.0, ge=0.0, le=20.0)):
    """
    Fetch REAL upcoming fixtures from The Odds API.
    No more invented matches — only real scheduled games.
    Tracks each prediction automatically in the database.
    """
    all_picks = []

    for league_id, sport_key in api_client.LEAGUE_TO_SPORT.items():
        if league_id not in LEAGUES:
            continue
        league = LEAGUES[league_id]
        events = api_client.fetch_upcoming_odds(sport_key)

        for event in events:
            home_name = event.get("home_team", "")
            away_name = event.get("away_team", "")
            commence = event.get("commence_time", "")

            hp = get_team_params(league_id, home_name)
            ap = get_team_params(league_id, away_name)
            if not hp or not ap:
                continue

            odds_data = parse_best_odds(event)
            h_odds = odds_data["home"] if odds_data else None
            d_odds = odds_data["draw"] if odds_data else None
            a_odds = odds_data["away"] if odds_data else None

            result = predict_match(
                home_attack=hp["attack"], home_defense=hp["defense"],
                away_attack=ap["attack"], away_defense=ap["defense"],
                home_adv=league["home_adv"],
                home_odds=h_odds, draw_odds=d_odds, away_odds=a_odds,
            )

            # ML adjustment
            odds_dict = {"home": h_odds, "draw": d_odds, "away": a_odds} if h_odds else None
            ml_probs = ml_model.adjust_probabilities(result, odds_dict)

            # Track prediction in DB
            date_str = commence[:10] if commence else ""
            try:
                db.save_prediction(league_id, home_name, away_name,
                                   date_str, result, odds_dict, ml_probs)
            except Exception:
                pass  # Don't fail the request if tracking fails

            for bet in result["value_bets"]:
                if bet["edge"] >= min_edge:
                    pick = {
                        "league": league["name"],
                        "league_id": league_id,
                        "home_team": home_name,
                        "away_team": away_name,
                        "date": date_str,
                        "time": commence[11:16] if len(commence) > 10 else "",
                        **bet,
                    }
                    if ml_probs:
                        pick["ml_adjusted"] = True
                    all_picks.append(pick)

    all_picks.sort(key=lambda x: x["edge"], reverse=True)

    return {
        "source": "The Odds API (tiempo real)" if api_client.ODDS_API_KEY else "Sin API key",
        "picks_count": len(all_picks),
        "min_edge_pct": min_edge,
        "picks": all_picks,
        "message": "" if all_picks else "No hay partidos próximos o la API key no está configurada.",
    }


# ─── Routes: Upcoming Fixtures ────────────────────────────────────────────────

@app.get("/fixtures/{league_id}")
def get_fixtures(league_id: str):
    """Get real upcoming fixtures for a specific league."""
    if league_id not in LEAGUES:
        raise HTTPException(404, "League not found")
    sport_key = api_client.LEAGUE_TO_SPORT.get(league_id)
    if not sport_key:
        raise HTTPException(404, "Sport key not found for league")
    events = api_client.fetch_upcoming_odds(sport_key)
    fixtures = []
    for ev in events:
        odds_data = parse_best_odds(ev)
        fixtures.append({
            "home_team": ev.get("home_team", ""),
            "away_team": ev.get("away_team", ""),
            "commence_time": ev.get("commence_time", ""),
            "odds": odds_data,
        })
    return {
        "league": LEAGUES[league_id]["name"],
        "fixtures_count": len(fixtures),
        "fixtures": fixtures,
    }


# ─── Routes: Analysis ─────────────────────────────────────────────────────────

@app.get("/analysis/metrics")
def analysis_metrics(league_id: str = None):
    return tracker.compute_metrics(league_id)


@app.get("/analysis/calibration")
def analysis_calibration(league_id: str = None):
    return tracker.get_calibration_data(league_id)


@app.get("/analysis/predictions")
def analysis_predictions(limit: int = Query(default=50, ge=1, le=500)):
    return tracker.get_recent_predictions_with_results(limit)


@app.get("/analysis/by-league")
def analysis_by_league():
    return tracker.get_accuracy_by_league()


@app.post("/results/update")
def update_results():
    saved = tracker.update_results_from_api()
    return {"message": f"{saved} resultados guardados", "saved": saved}


# ─── Routes: ML ────────────────────────────────────────────────────────────────

@app.get("/ml/status")
def ml_status():
    return ml_model.get_model_status()


@app.post("/ml/train")
def ml_train():
    return ml_model.train_model()


# ─── Routes: System ───────────────────────────────────────────────────────────

@app.post("/recalibrate")
def recalibrate():
    try:
        _run_recalibration()
        return {"message": "Parámetros recalibrados exitosamente"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/system/status")
def system_status():
    counts = db.get_prediction_count()
    ml_info = ml_model.get_model_status()
    sched_running = _scheduler is not None and _scheduler.running if _scheduler else False
    return {
        "api_key_configured": bool(api_client.ODDS_API_KEY),
        "scheduler_running": sched_running,
        "predictions": counts,
        "ml_model": ml_info,
        "leagues": list(LEAGUES.keys()),
        "params_source": {k: v.get("_source", "unknown") for k, v in LEAGUES.items()},
    }


# ─── Serve Frontend (production) ──────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if FRONTEND_DIR.exists():
    @app.get("/app")
    def serve_dashboard():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/app/results")
    def serve_results():
        return FileResponse(FRONTEND_DIR / "results.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
