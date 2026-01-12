#!/usr/bin/env python3
"""
Utility script for batch interpretation workflow.
Safe wrapper for CSV operations - designed for scoped permissions.

Usage:
    py -3.12 batch_utils.py find-uninterpreted --sort-by rank_control --limit 6
    py -3.12 batch_utils.py update-csv --feature 17240 --interpretation "..."
    py -3.12 batch_utils.py check-existing --features 17240,3080,1177
    py -3.12 batch_utils.py backup-csv
    py -3.12 batch_utils.py ensure-output-dir --feature 17240
"""

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

# Paths relative to script location
SCRIPT_DIR = Path(__file__).parent
FEATURE_DATA_DIR = SCRIPT_DIR / "feature data"
CSV_PATH = FEATURE_DATA_DIR / "Feature_output.csv"
OUTPUT_DIR = SCRIPT_DIR / "output" / "interpretations"


def find_uninterpreted(sort_by: str, limit: int) -> None:
    """Find features with empty interpretations, sorted by specified column."""
    if sort_by not in ("rank_control", "rank_nocontrol"):
        print(f"ERROR: sort_by must be 'rank_control' or 'rank_nocontrol', got '{sort_by}'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Empty interpretation means empty string or whitespace only
            interpretation = row.get("interpretation", "").strip()
            if not interpretation:
                try:
                    rows.append({
                        "feature_index": int(row["feature_index"]),
                        "rank_control": int(row["rank_control"]),
                        "rank_nocontrol": int(row["rank_nocontrol"])
                    })
                except (ValueError, KeyError) as e:
                    # Skip malformed rows
                    continue

    # Sort by specified column ascending
    rows.sort(key=lambda x: x[sort_by])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "total_uninterpreted": len(rows),
        "selected_count": len(selected),
        "sort_by": sort_by,
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def check_existing(features: list[int]) -> None:
    """Check which features have existing output files and pre-computed data."""
    results = {
        "has_output": [],      # Has results.json in output/interpretations/
        "missing_output": [],  # No results.json
        "has_precomputed": [], # Has feature_N.json in feature data/
        "missing_precomputed": []  # No pre-computed data
    }

    for feat_id in features:
        # Check for output (no underscore in folder name)
        output_path = OUTPUT_DIR / f"feature{feat_id}" / "results.json"
        if output_path.exists():
            results["has_output"].append(feat_id)
        else:
            results["missing_output"].append(feat_id)

        # Check for pre-computed data (underscore in filename)
        precomputed_path = FEATURE_DATA_DIR / f"feature_{feat_id}.json"
        if precomputed_path.exists():
            results["has_precomputed"].append(feat_id)
        else:
            results["missing_precomputed"].append(feat_id)

    print(json.dumps(results, indent=2))


def update_csv(feature: int, interpretation: str) -> None:
    """Update the interpretation column for a specific feature."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    updated = False

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row["feature_index"]) == feature:
                row["interpretation"] = interpretation
                updated = True
            rows.append(row)

    if not updated:
        print(f"ERROR: Feature {feature} not found in CSV", file=sys.stderr)
        sys.exit(1)

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"status": "success", "feature": feature, "updated": True}))


def backup_csv() -> None:
    """Create a backup of the CSV file."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    backup_path = CSV_PATH.with_suffix(".csv.backup")
    shutil.copy2(CSV_PATH, backup_path)

    print(json.dumps({
        "status": "success",
        "source": str(CSV_PATH),
        "backup": str(backup_path)
    }))


def ensure_output_dir(feature: int) -> None:
    """Create output directory for a feature interpretation."""
    output_path = OUTPUT_DIR / f"feature{feature}"
    output_path.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "status": "success",
        "feature": feature,
        "path": str(output_path)
    }))


def extract_interpretation(feature: int) -> None:
    """Extract interpretation from results.json for a feature."""
    results_path = OUTPUT_DIR / f"feature{feature}" / "results.json"

    if not results_path.exists():
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"results.json not found at {results_path}"
        }))
        sys.exit(1)

    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different result formats
        # Format 1: top-level fields (label, verdict, executive_summary)
        # Format 2: nested in final_interpretation (label, short_description) + challenge_phase.verdict
        if "final_interpretation" in data:
            fi = data["final_interpretation"]
            label = fi.get("label", "Unknown")
            summary = fi.get("short_description", fi.get("detailed_description", "")[:200])
            verdict = data.get("challenge_phase", {}).get("verdict", "UNKNOWN")
        else:
            label = data.get("label", "Unknown")
            verdict = data.get("verdict", "UNKNOWN")
            summary = data.get("executive_summary", "")

        # Format: "Label [VERDICT]: Summary"
        interpretation = f"{label} [{verdict}]: {summary}"

        # Truncate to 250 chars if needed
        if len(interpretation) > 250:
            interpretation = interpretation[:247] + "..."

        print(json.dumps({
            "status": "success",
            "feature": feature,
            "interpretation": interpretation,
            "label": label,
            "verdict": verdict,
            "truncated": len(f"{label} [{verdict}]: {summary}") > 250
        }))
    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Malformed JSON: {e}"
        }))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Batch interpretation utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # find-uninterpreted
    find_parser = subparsers.add_parser("find-uninterpreted",
        help="Find features with empty interpretations")
    find_parser.add_argument("--sort-by", required=True,
        choices=["rank_control", "rank_nocontrol"],
        help="Column to sort by")
    find_parser.add_argument("--limit", type=int, required=True,
        help="Maximum number of features to return")

    # check-existing
    check_parser = subparsers.add_parser("check-existing",
        help="Check which features have existing outputs")
    check_parser.add_argument("--features", required=True,
        help="Comma-separated list of feature IDs")

    # update-csv
    update_parser = subparsers.add_parser("update-csv",
        help="Update interpretation for a feature")
    update_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    update_parser.add_argument("--interpretation", required=True,
        help="Interpretation text to set")

    # backup-csv
    subparsers.add_parser("backup-csv", help="Create CSV backup")

    # ensure-output-dir
    ensure_parser = subparsers.add_parser("ensure-output-dir",
        help="Create output directory for a feature")
    ensure_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to create directory for")

    # extract-interpretation
    extract_parser = subparsers.add_parser("extract-interpretation",
        help="Extract interpretation from results.json")
    extract_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to extract")

    args = parser.parse_args()

    if args.command == "find-uninterpreted":
        find_uninterpreted(args.sort_by, args.limit)
    elif args.command == "check-existing":
        features = [int(x.strip()) for x in args.features.split(",")]
        check_existing(features)
    elif args.command == "update-csv":
        update_csv(args.feature, args.interpretation)
    elif args.command == "backup-csv":
        backup_csv()
    elif args.command == "ensure-output-dir":
        ensure_output_dir(args.feature)
    elif args.command == "extract-interpretation":
        extract_interpretation(args.feature)


if __name__ == "__main__":
    main()
