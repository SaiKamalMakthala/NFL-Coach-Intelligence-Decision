import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from src.utils.db import read_table, write_table
from src.utils.log import log


def pick_existing(df, candidates):
    """Return list of candidates that exist in df.columns."""
    return [c for c in candidates if c in df.columns]


def main():
    log("Day 20: Explainability via permutation importance (robust)")

    df = read_table("fourth_down_predictions")
    log(f"Columns available: {df.columns.tolist()}")

    # Candidate feature columns (try many names used across your scripts)
    candidate_features = [
        "ydstogo", "yards_to_go", "to_go",
        "yardline_100", "yardline", "yardline100",
        "clock_pressure", "clock_pressure_norm",
        "pressure_score",
        "score_diff_abs", "score_differential",
        "decision_risk_score",
        "field_position_score"
    ]

    FEATURES = pick_existing(df, candidate_features)

    # If ydstogo/yardline aren't in predictions, we can still explain using any existing features.
    if len(FEATURES) == 0:
        raise ValueError(
            "No usable feature columns found in fourth_down_predictions. "
            "Fix: include ydstogo/yardline_100 (and other features) when saving Day 16 predictions."
        )

    # Target columns expected from Day 16 output
    if "actual_go" not in df.columns:
        raise ValueError("Missing 'actual_go' in fourth_down_predictions")
    y = df["actual_go"].astype(int)

    X = df[FEATURES].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # Train/test split (for a stable explainability run)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Evaluate baseline AUC
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    log(f"AUC (retrained for explainability): {auc:.4f}")

    # Permutation importance
    imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=42, scoring="roc_auc"
    )

    out = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std
    }).sort_values("importance_mean", ascending=False)

    write_table(out, "fourth_down_feature_importance")
    log("Saved → fourth_down_feature_importance ")
    print(out)


if __name__ == "__main__":
    main()
