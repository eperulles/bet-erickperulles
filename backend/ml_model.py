"""
ML meta-model for BetIQ.
Learns from historical predictions vs results to adjust Dixon-Coles probabilities.
Uses gradient boosting with calibration.
"""
import pickle
import numpy as np
from pathlib import Path
import db

MODEL_PATH = Path(__file__).parent / "ml_meta_model.pkl"
MIN_SAMPLES = 30


def _prepare_features(data):
    """Build feature matrix and target from predictions with results."""
    X, y = [], []
    for row in data:
        X.append([
            row["prob_home"] or 0,
            row["prob_draw"] or 0,
            row["prob_away"] or 0,
            row["lambda_home"] or 0,
            row["lambda_away"] or 0,
            row["prob_over_25"] or 0,
            row["odds_home"] or 0,
            row["odds_draw"] or 0,
            row["odds_away"] or 0,
            row["edge_best"] or 0,
        ])
        y.append({"H": 0, "D": 1, "A": 2}.get(row["result"], 0))
    return np.array(X), np.array(y)


def train_model():
    """Train the ML meta-model on historical predictions."""
    data = db.get_predictions_with_results()
    if len(data) < MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "samples": len(data),
            "min_required": MIN_SAMPLES,
            "message": f"Necesitas al menos {MIN_SAMPLES} predicciones con resultado. Tienes {len(data)}.",
        }

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return {"status": "error", "message": "scikit-learn no instalado. pip install scikit-learn"}

    X, y = _prepare_features(data)

    # Check we have all 3 classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 3:
        return {
            "status": "insufficient_variety",
            "samples": len(data),
            "message": "Necesitas resultados de los 3 tipos (H/D/A) para entrenar.",
        }

    base = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )

    cv_folds = min(5, len(data) // 10)
    if cv_folds < 2:
        cv_folds = 2

    scores = cross_val_score(base, X, y, cv=cv_folds, scoring="accuracy")

    # Train final model with calibration
    model = CalibratedClassifierCV(base, cv=cv_folds, method="isotonic")
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return {
        "status": "trained",
        "samples": len(data),
        "cv_accuracy": round(np.mean(scores) * 100, 1),
        "cv_std": round(np.std(scores) * 100, 1),
        "message": f"Modelo entrenado con {len(data)} muestras. Accuracy CV: {np.mean(scores)*100:.1f}%",
    }


def load_model():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


def adjust_probabilities(prediction, odds=None):
    """
    Adjust Dixon-Coles probabilities using the trained ML model.
    Returns adjusted {home, draw, away} or None if model not available.
    """
    model = load_model()
    if model is None:
        return None

    probs = prediction.get("probs_1x2", {})
    ou = prediction.get("ou_markets", {}).get("ou_25", {})

    features = np.array([[
        probs.get("home", 0),
        probs.get("draw", 0),
        probs.get("away", 0),
        prediction.get("lambda_home", 0),
        prediction.get("lambda_away", 0),
        ou.get("over", 0),
        odds.get("home", 0) if odds else 0,
        odds.get("draw", 0) if odds else 0,
        odds.get("away", 0) if odds else 0,
        0,  # edge_best placeholder
    ]])

    try:
        adjusted = model.predict_proba(features)[0]
        # Classes are [0=H, 1=D, 2=A]
        return {
            "home": round(float(adjusted[0]), 4),
            "draw": round(float(adjusted[1]), 4),
            "away": round(float(adjusted[2]), 4),
        }
    except Exception:
        return None


def get_model_status():
    """Get ML model status information."""
    counts = db.get_prediction_count()
    model = load_model()
    return {
        "model_available": model is not None,
        "predictions_total": counts["total_predictions"],
        "predictions_matched": counts["matched_with_results"],
        "min_samples_required": MIN_SAMPLES,
        "ready_to_train": counts["matched_with_results"] >= MIN_SAMPLES,
    }
