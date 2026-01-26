import sqlite3
import pandas as pd
from datetime import datetime
from src.config import DB_PATH


def check_required_columns(df, required_cols, table_name):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return [{
            "table": table_name,
            "check": "required_columns",
            "status": "FAIL",
            "details": f"Missing columns: {missing}"
        }]
    return [{
        "table": table_name,
        "check": "required_columns",
        "status": "PASS",
        "details": "All required columns present"
    }]


def check_no_duplicate_keys(df, keys, table_name):
    dup_count = df.duplicated(subset=keys).sum()
    status = "PASS" if dup_count == 0 else "FAIL"
    return [{
        "table": table_name,
        "check": "duplicate_keys",
        "status": status,
        "details": f"Duplicate rows by keys {keys}: {dup_count}"
    }]


def check_range(df, col, min_val, max_val, table_name):
    if col not in df.columns:
        return [{
            "table": table_name,
            "check": f"range_{col}",
            "status": "SKIP",
            "details": f"Column {col} not present"
        }]

    bad = df[(df[col] < min_val) | (df[col] > max_val)].shape[0]
    status = "PASS" if bad == 0 else "FAIL"
    return [{
        "table": table_name,
        "check": f"range_{col}",
        "status": status,
        "details": f"Out-of-range count for {col}: {bad} (expected {min_val}..{max_val})"
    }]


def check_non_negative(df, col, table_name):
    if col not in df.columns:
        return [{
            "table": table_name,
            "check": f"non_negative_{col}",
            "status": "SKIP",
            "details": f"Column {col} not present"
        }]

    bad = df[df[col] < 0].shape[0]
    status = "PASS" if bad == 0 else "FAIL"
    return [{
        "table": table_name,
        "check": f"non_negative_{col}",
        "status": status,
        "details": f"Negative count for {col}: {bad}"
    }]


def check_null_rate(df, col, threshold, table_name):
    if col not in df.columns:
        return [{
            "table": table_name,
            "check": f"null_rate_{col}",
            "status": "SKIP",
            "details": f"Column {col} not present"
        }]

    null_rate = df[col].isna().mean()
    status = "PASS" if null_rate <= threshold else "FAIL"
    return [{
        "table": table_name,
        "check": f"null_rate_{col}",
        "status": status,
        "details": f"Null rate for {col}: {null_rate:.4f} (threshold {threshold})"
    }]


def check_pass_run_rates(df, table_name):
    # pass_rate and run_rate should be between 0 and 1
    results = []
    results += check_range(df, "pass_rate", 0, 1, table_name)
    results += check_range(df, "run_rate", 0, 1, table_name)

    if "pass_rate" in df.columns and "run_rate" in df.columns:
        df["pass_run_sum"] = df["pass_rate"] + df["run_rate"]
        bad = df[(df["pass_run_sum"] < 0) | (df["pass_run_sum"] > 1.001)].shape[0]
        status = "PASS" if bad == 0 else "FAIL"
        results.append({
            "table": table_name,
            "check": "pass_run_sum",
            "status": status,
            "details": f"Rows where pass_rate+run_rate > 1: {bad}"
        })
    else:
        results.append({
            "table": table_name,
            "check": "pass_run_sum",
            "status": "SKIP",
            "details": "pass_rate or run_rate missing"
        })

    return results


def run_checks():
    conn = sqlite3.connect(DB_PATH)

    # Load tables
    feature = pd.read_sql("SELECT * FROM feature_table_4", conn)
    tendencies = pd.read_sql("SELECT * FROM playcalling_tendencies", conn)

    conn.close()

    report = []
    now = datetime.utcnow().isoformat()

    # ---------------------------
    # Checks: feature_table_4
    # ---------------------------
    feature_required = [
        "game_id", "play_id", "posteam", "defteam",
        "down", "ydstogo", "yardline_100",
        "epa", "success"
    ]

    report += check_required_columns(feature, feature_required, "feature_table_4")
    report += check_no_duplicate_keys(feature, ["game_id", "play_id"], "feature_table_4")
    report += check_range(feature, "down", 1, 4, "feature_table_4")
    report += check_non_negative(feature, "ydstogo", "feature_table_4")
    report += check_range(feature, "yardline_100", 0, 100, "feature_table_4")
    report += check_range(feature, "success", 0, 1, "feature_table_4")
    report += check_null_rate(feature, "epa", 0.05, "feature_table_4")

    if "pressure_score" in feature.columns:
        report += check_range(feature, "pressure_score", 0, 1, "feature_table_4")

    # ---------------------------
    # Checks: playcalling_tendencies
    # ---------------------------
    tend_required = [
        "posteam", "down", "distance_bucket", "field_zone", "pressure_bucket",
        "plays", "success_rate", "avg_epa", "pass_rate", "run_rate"
    ]

    report += check_required_columns(tendencies, tend_required, "playcalling_tendencies")
    report += check_range(tendencies, "down", 1, 4, "playcalling_tendencies")
    report += check_range(tendencies, "success_rate", 0, 1, "playcalling_tendencies")
    report += check_pass_run_rates(tendencies, "playcalling_tendencies")

    # Add timestamp
    for r in report:
        r["run_timestamp_utc"] = now

    return pd.DataFrame(report)


def main():
    print("Starting Day 13: Sanity Checks")

    report_df = run_checks()
    print(report_df)

    # Save report to SQLite for tracking
    conn = sqlite3.connect(DB_PATH)
    report_df.to_sql("data_quality_report", conn, if_exists="replace", index=False)
    conn.close()

    # Print quick failure summary
    fails = report_df[report_df["status"] == "FAIL"]
    if len(fails) > 0:
        print("/nSanity Checks FAILED:")
        print(fails[["table", "check", "details"]])
    else:
        print("\n✅ All sanity checks PASSED")

    print("data_quality_report saved to SQLite")


if __name__ == "__main__":
    main()
