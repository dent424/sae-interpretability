#!/usr/bin/env python3
"""
Utility script for batch interpretation workflow.
Safe wrapper for CSV operations - designed for scoped permissions.

Usage:
    py -3.12 batch_utils.py find-uninterpreted --sort-by rank_control --limit 6
    py -3.12 batch_utils.py find-unverified --sort-by rank_nocontrol --limit 10
    py -3.12 batch_utils.py find-auditable --limit 10
    py -3.12 batch_utils.py find-fixable --limit 10
    py -3.12 batch_utils.py update-csv --feature 17240 --interpretation "..."
    py -3.12 batch_utils.py update-verify --feature 17240 --status PASS
    py -3.12 batch_utils.py update-audit --feature 17240 --status COMPLETE
    py -3.12 batch_utils.py force-update-audit --feature 17240 --status COMPLETE
    py -3.12 batch_utils.py update-fix-status --feature 17240 --status schema_fix
    py -3.12 batch_utils.py check-existing --features 17240,3080,1177
    py -3.12 batch_utils.py backup-csv
    py -3.12 batch_utils.py ensure-output-dir --feature 17240
    py -3.12 batch_utils.py timestamp
    py -3.12 batch_utils.py extract-interpretation --feature 17240
"""

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime, timezone
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


def find_unverified(sort_by: str, limit: int) -> None:
    """Find features with interpretations but no verification status.

    Finds features where:
    - interpretation column is NOT empty (has been interpreted)
    - verify_status column IS empty (not yet verified)
    - results.json exists in output/interpretations/feature<ID>/
    """
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
            # Has interpretation but no verification
            interpretation = row.get("interpretation", "").strip()
            verify_status = row.get("verify_status", "").strip()

            if interpretation and not verify_status:
                try:
                    feature_id = int(row["feature_index"])
                    # Check if results.json exists
                    results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                    if results_path.exists():
                        rows.append({
                            "feature_index": feature_id,
                            "rank_control": int(row["rank_control"]),
                            "rank_nocontrol": int(row["rank_nocontrol"]),
                            "interpretation": interpretation[:50] + "..." if len(interpretation) > 50 else interpretation
                        })
                except (ValueError, KeyError):
                    continue

    # Sort by specified column ascending
    rows.sort(key=lambda x: x[sort_by])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "total_unverified": len(rows),
        "selected_count": len(selected),
        "sort_by": sort_by,
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def find_auditable(sort_by: str, limit: int) -> None:
    """Find features with interpretations that haven't been audited yet.

    Finds features where:
    - interpretation column is NOT empty (has been interpreted)
    - audit_status column IS empty (not yet audited)
    - results.json exists in output/interpretations/feature<ID>/
    """
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
            # Has interpretation but no audit yet
            interpretation = row.get("interpretation", "").strip()
            audit_status = row.get("audit_status", "").strip()

            # Include if: has interpretation AND audit_status is empty
            if interpretation and not audit_status:
                try:
                    feature_id = int(row["feature_index"])
                    # Check if results.json exists
                    results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                    if results_path.exists():
                        rows.append({
                            "feature_index": feature_id,
                            "rank_control": int(row["rank_control"]),
                            "rank_nocontrol": int(row["rank_nocontrol"]),
                            "interpretation": interpretation[:50] + "..." if len(interpretation) > 50 else interpretation,
                            "current_audit_status": audit_status if audit_status else "(empty)"
                        })
                except (ValueError, KeyError):
                    continue

    # Sort by specified column ascending
    rows.sort(key=lambda x: x[sort_by])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "total_auditable": len(rows),
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
        fieldnames = list(reader.fieldnames)
        # Add interpretation column if missing
        if "interpretation" not in fieldnames:
            fieldnames.append("interpretation")
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


def update_verify(feature: int, status: str, reason: str = "") -> None:
    """Update verification status for a specific feature.

    Adds verify_status, verify_date, and verify_reason columns if they don't exist.
    Only fills empty cells - never overwrites existing values.
    """
    if status not in ("PASS", "WARN", "FAIL"):
        print(f"ERROR: status must be PASS, WARN, or FAIL, got '{status}'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    found = False
    already_set = False

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add new columns if they don't exist
        columns_added = []
        if "verify_status" not in fieldnames:
            fieldnames.append("verify_status")
            columns_added.append("verify_status")
        if "verify_date" not in fieldnames:
            fieldnames.append("verify_date")
            columns_added.append("verify_date")
        if "verify_reason" not in fieldnames:
            fieldnames.append("verify_reason")
            columns_added.append("verify_reason")

        for row in reader:
            # Ensure new columns exist in row
            if "verify_status" not in row:
                row["verify_status"] = ""
            if "verify_date" not in row:
                row["verify_date"] = ""
            if "verify_reason" not in row:
                row["verify_reason"] = ""

            if int(row["feature_index"]) == feature:
                found = True
                # Only update if cells are empty
                if row["verify_status"].strip():
                    already_set = True
                else:
                    row["verify_status"] = status
                    row["verify_date"] = datetime.now(timezone.utc).isoformat()
                    row["verify_reason"] = reason if status in ("WARN", "FAIL") else ""
            rows.append(row)

    if not found:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Feature {feature} not found in CSV"
        }))
        sys.exit(1)

    if already_set:
        print(json.dumps({
            "status": "skipped",
            "feature": feature,
            "reason": "verify_status already has a value - not overwriting"
        }))
        return

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "feature": feature,
        "verify_status": status,
        "verify_reason": reason if status in ("WARN", "FAIL") else "",
        "columns_added": columns_added if columns_added else None
    }))


def update_audit(feature: int, status: str) -> None:
    """Update audit status for a specific feature.

    Adds audit_status and audit_date columns if they don't exist.
    Only fills empty cells - never overwrites existing values.

    Valid status prefixes: COMPLETE, SCHEMA, INCOMPLETE, PROCESS_INCOMPLETE, FIXED
    Status can include details after colon (e.g., "SCHEMA:label,description")
    """
    # Validate status prefix
    valid_prefixes = ("COMPLETE", "SCHEMA", "INCOMPLETE", "PROCESS_INCOMPLETE", "FIXED")
    prefix = status.split(":")[0]
    if prefix not in valid_prefixes:
        print(f"ERROR: status must start with {valid_prefixes}, got '{status}'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    found = False
    already_set = False

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add new columns if they don't exist
        columns_added = []
        if "audit_status" not in fieldnames:
            fieldnames.append("audit_status")
            columns_added.append("audit_status")
        if "audit_date" not in fieldnames:
            fieldnames.append("audit_date")
            columns_added.append("audit_date")

        for row in reader:
            # Ensure new columns exist in row
            if "audit_status" not in row:
                row["audit_status"] = ""
            if "audit_date" not in row:
                row["audit_date"] = ""

            if int(row["feature_index"]) == feature:
                found = True
                # Only update if cells are empty
                if row["audit_status"].strip():
                    already_set = True
                else:
                    row["audit_status"] = status
                    row["audit_date"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

    if not found:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Feature {feature} not found in CSV"
        }))
        sys.exit(1)

    if already_set:
        print(json.dumps({
            "status": "skipped",
            "feature": feature,
            "reason": "audit_status already has a value - not overwriting"
        }))
        return

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "feature": feature,
        "audit_status": status,
        "columns_added": columns_added if columns_added else None
    }))


def force_update_audit(feature: int, status: str) -> None:
    """Force update audit status for a specific feature, overwriting existing value.

    Unlike update_audit, this WILL overwrite existing values.
    Used after fixing a feature to re-audit and update status.

    Valid status prefixes: COMPLETE, SCHEMA, INCOMPLETE, PROCESS_INCOMPLETE
    Status can include details after colon (e.g., "SCHEMA:label,description")
    """
    # Validate status prefix
    valid_prefixes = ("COMPLETE", "SCHEMA", "INCOMPLETE", "PROCESS_INCOMPLETE")
    prefix = status.split(":")[0]
    if prefix not in valid_prefixes:
        print(f"ERROR: status must start with {valid_prefixes}, got '{status}'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    found = False
    old_status = ""

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add new columns if they don't exist
        columns_added = []
        if "audit_status" not in fieldnames:
            fieldnames.append("audit_status")
            columns_added.append("audit_status")
        if "audit_date" not in fieldnames:
            fieldnames.append("audit_date")
            columns_added.append("audit_date")

        for row in reader:
            # Ensure new columns exist in row
            if "audit_status" not in row:
                row["audit_status"] = ""
            if "audit_date" not in row:
                row["audit_date"] = ""

            if int(row["feature_index"]) == feature:
                found = True
                old_status = row["audit_status"]
                # Force overwrite
                row["audit_status"] = status
                row["audit_date"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

    if not found:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Feature {feature} not found in CSV"
        }))
        sys.exit(1)

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "feature": feature,
        "audit_status": status,
        "old_status": old_status if old_status else None,
        "columns_added": columns_added if columns_added else None
    }))


def update_fix_status(feature: int, status: str) -> None:
    """Update fix status for a specific feature.

    Adds fix_status and fix_date columns if they don't exist.
    Always overwrites existing values (fixes can be re-run).

    Valid status values: schema_fix, synthesis_fix, schema+synthesis, unsalvageable
    """
    valid_statuses = ("schema_fix", "synthesis_fix", "schema+synthesis", "unsalvageable")
    if status not in valid_statuses:
        print(f"ERROR: status must be one of {valid_statuses}, got '{status}'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    found = False
    old_status = ""

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add new columns if they don't exist
        columns_added = []
        if "fix_status" not in fieldnames:
            fieldnames.append("fix_status")
            columns_added.append("fix_status")
        if "fix_date" not in fieldnames:
            fieldnames.append("fix_date")
            columns_added.append("fix_date")

        for row in reader:
            # Ensure new columns exist in row
            if "fix_status" not in row:
                row["fix_status"] = ""
            if "fix_date" not in row:
                row["fix_date"] = ""

            if int(row["feature_index"]) == feature:
                found = True
                old_status = row["fix_status"]
                row["fix_status"] = status
                row["fix_date"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)

    if not found:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Feature {feature} not found in CSV"
        }))
        sys.exit(1)

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "feature": feature,
        "fix_status": status,
        "old_status": old_status if old_status else None,
        "columns_added": columns_added if columns_added else None
    }))


def find_fixable(limit: int) -> None:
    """Find features with INCOMPLETE or SCHEMA audit status that may be fixable.

    Finds features where:
    - audit_status starts with INCOMPLETE or SCHEMA
    - results.json exists in output/interpretations/feature<ID>/
    - fix_status is empty or not 'unsalvageable'
    """
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audit_status = row.get("audit_status", "").strip()
            fix_status = row.get("fix_status", "").strip()

            # Check if audit status indicates fixable issues
            is_fixable_status = (
                audit_status.startswith("INCOMPLETE") or
                audit_status.startswith("SCHEMA")
            )

            # Skip if already marked as unsalvageable
            if fix_status == "unsalvageable":
                continue

            if is_fixable_status:
                try:
                    feature_id = int(row["feature_index"])
                    # Check if results.json exists
                    results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                    if results_path.exists():
                        rows.append({
                            "feature_index": feature_id,
                            "rank_control": int(row.get("rank_control", 0)),
                            "rank_nocontrol": int(row.get("rank_nocontrol", 0)),
                            "audit_status": audit_status,
                            "fix_status": fix_status if fix_status else "(empty)"
                        })
                except (ValueError, KeyError):
                    continue

    # Sort by rank_nocontrol ascending (prioritize higher-ranked features)
    rows.sort(key=lambda x: x["rank_nocontrol"])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "total_fixable": len(rows),
        "selected_count": len(selected),
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


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


def timestamp() -> None:
    """Output current UTC timestamp in ISO 8601 format.

    Platform-agnostic timestamp for audit trails.
    Output is just the timestamp string, no JSON wrapper.
    """
    print(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


def batch_summary(features: list[int]) -> None:
    """Generate and save a summary of a batch verification run.

    Reads verification status from CSV and verification.md files,
    outputs summary JSON, and appends to batch_history.csv.
    """
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read verification status from CSV
    csv_statuses = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feat_id = int(row["feature_index"])
            if feat_id in features:
                csv_statuses[feat_id] = {
                    "verify_status": row.get("verify_status", ""),
                    "verify_date": row.get("verify_date", ""),
                    "interpretation": row.get("interpretation", "")[:50]
                }

    # Read detailed status from verification.md files
    results = []
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for feat_id in features:
        verification_path = OUTPUT_DIR / f"feature{feat_id}" / "verification.md"

        # Default values
        logic = "N/A"
        causal = "N/A"
        repro = "N/A"
        overall = csv_statuses.get(feat_id, {}).get("verify_status", "N/A")
        reason = ""

        # Try to parse verification.md for detailed status
        if verification_path.exists():
            try:
                content = verification_path.read_text(encoding="utf-8")
                lines = content.split("\n")

                # Parse summary table and collect reasons
                in_flags = False
                flags_text = []

                for i, line in enumerate(lines):
                    line_stripped = line.strip()

                    # Parse status table
                    if line_stripped.startswith("| Logic"):
                        parts = [p.strip() for p in line_stripped.split("|")]
                        if len(parts) >= 3:
                            logic = parts[2]
                    elif line_stripped.startswith("| Causal"):
                        parts = [p.strip() for p in line_stripped.split("|")]
                        if len(parts) >= 3:
                            causal = parts[2]
                    elif line_stripped.startswith("| Repro"):
                        parts = [p.strip() for p in line_stripped.split("|")]
                        if len(parts) >= 3:
                            repro = parts[2]
                    elif "**Overall:**" in line_stripped or "**Overall**" in line_stripped:
                        if "PASS" in line_stripped:
                            overall = "PASS"
                        elif "WARN" in line_stripped:
                            overall = "WARN"
                        elif "FAIL" in line_stripped:
                            overall = "FAIL"

                    # Collect flags/reasons for WARN/FAIL
                    if "**Flags:**" in line_stripped:
                        in_flags = True
                        continue
                    if in_flags:
                        if line_stripped.startswith("- WARN:") or line_stripped.startswith("- FAIL:"):
                            flags_text.append(line_stripped[2:])  # Remove "- "
                        elif line_stripped.startswith("**") or line_stripped.startswith("---"):
                            in_flags = False

                    # Also check for Assessment/Note sections that explain WARN
                    for label in ["**Assessment:**", "**Note:**"]:
                        if label in line_stripped:
                            # Content may be on same line or next line
                            after_label = line_stripped.split(label, 1)[-1].strip()
                            if after_label and not reason:
                                reason = after_label[:100]
                            elif not reason:
                                # Get next non-empty line
                                for j in range(i+1, min(i+3, len(lines))):
                                    next_line = lines[j].strip()
                                    if next_line and not next_line.startswith(("-", "**", "---", "|")):
                                        reason = next_line[:100]
                                        break
                            break

                # Build reason string from flags
                if flags_text:
                    reason = "; ".join(flags_text)[:150]

                # If still no reason but WARN, try to find it in specific sections
                if not reason and overall == "WARN":
                    # Check which check failed
                    warn_checks = []
                    if logic == "WARN":
                        warn_checks.append("Logic")
                    if causal == "WARN":
                        warn_checks.append("Causal")
                    if repro == "WARN":
                        warn_checks.append("Repro")
                    if warn_checks:
                        reason = f"{'/'.join(warn_checks)} check flagged"

            except Exception:
                pass

        # Count statuses
        if overall == "PASS":
            pass_count += 1
        elif overall == "WARN":
            warn_count += 1
        elif overall == "FAIL":
            fail_count += 1

        results.append({
            "feature": feat_id,
            "overall": overall,
            "logic": logic,
            "causal": causal,
            "repro": repro,
            "reason": reason if overall in ("WARN", "FAIL") else ""
        })

    # Create summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_features": len(features),
        "pass_count": pass_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "features": features,
        "results": results
    }

    # Update verify_reason in Feature_output.csv for WARN/FAIL features
    # Read current CSV
    rows = []
    fieldnames = None
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Ensure verify_reason column exists
        if "verify_reason" not in fieldnames:
            fieldnames.append("verify_reason")

        for row in reader:
            if "verify_reason" not in row:
                row["verify_reason"] = ""
            rows.append(row)

    # Build lookup of reasons by feature
    reason_lookup = {r["feature"]: r["reason"] for r in results if r["reason"]}

    # Update rows with reasons
    updated_count = 0
    for row in rows:
        feat_id = int(row["feature_index"])
        if feat_id in reason_lookup and not row.get("verify_reason", "").strip():
            row["verify_reason"] = reason_lookup[feat_id][:150]  # Truncate to 150 chars
            updated_count += 1

    # Write back
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary["reasons_updated"] = updated_count

    # Output JSON summary
    print(json.dumps(summary, indent=2))


def _extract_verdict(data: dict) -> str:
    """Extract verdict from results.json, checking all known paths.

    Returns normalized verdict string (CONFIRMED, REFINED, REJECTED, or UNKNOWN).
    """
    verdict = None

    # Path 1: challenge_phase.verdict (string) - most common (27 files)
    cp = data.get("challenge_phase", {})
    if isinstance(cp.get("verdict"), str):
        verdict = cp["verdict"]

    # Path 2: top-level verdict (string) - second most common (25 files)
    elif isinstance(data.get("verdict"), str):
        verdict = data["verdict"]

    # Path 3: challenge_phase.verdict.result (nested dict)
    elif isinstance(cp.get("verdict"), dict):
        v = cp["verdict"]
        if "result" in v:
            verdict = v["result"]
        elif "outcome" in v:
            verdict = v["outcome"]
        elif "final_label" in v:
            # Path 4: challenge_phase.verdict.final_label - use as fallback info
            pass

    # Path 5: challenge_phase.challenge_verdict.verdict
    elif isinstance(cp.get("challenge_verdict"), dict):
        cv = cp["challenge_verdict"]
        if "verdict" in cv:
            verdict = cv["verdict"]
        elif "decision" in cv:
            verdict = cv["decision"]
        elif "revision_needed" in cv:
            # revision_needed: true -> REFINED, false -> CONFIRMED
            verdict = "REFINED" if cv["revision_needed"] else "CONFIRMED"

    if verdict is None:
        return "UNKNOWN"

    # Normalize verdict strings
    verdict = verdict.upper().strip()
    if verdict == "REFINE":
        verdict = "REFINED"
    elif verdict == "SURVIVES_WITH_REFINEMENT":
        verdict = "REFINED"

    return verdict


def _extract_label(data: dict) -> str:
    """Extract label from results.json, checking all known paths.

    Returns the interpretation label or 'Unknown'.
    """
    # Path 1: top-level label (25 files)
    if "label" in data and data["label"]:
        return data["label"]

    # Path 2: final_interpretation.label (8 files)
    fi = data.get("final_interpretation", {})
    if isinstance(fi, dict) and "label" in fi and fi["label"]:
        return fi["label"]

    # Path 3: final_interpretation is a string itself (1 file)
    if isinstance(fi, str) and fi:
        return fi

    # Path 4: challenge_phase.verdict.final_label (1 file)
    cp = data.get("challenge_phase", {})
    v = cp.get("verdict", {})
    if isinstance(v, dict) and "final_label" in v and v["final_label"]:
        return v["final_label"]

    # Path 5: challenge_phase.revised_interpretation.label (1 file)
    ri = cp.get("revised_interpretation", {})
    if isinstance(ri, dict) and "label" in ri and ri["label"]:
        return ri["label"]

    return "Unknown"


def _extract_summary(data: dict) -> str:
    """Extract summary from results.json, checking all known paths.

    Returns a brief summary description.
    """
    # Path 1: final_interpretation fields
    fi = data.get("final_interpretation", {})
    if isinstance(fi, dict):
        if "short_description" in fi and fi["short_description"]:
            return fi["short_description"]
        if "detailed_description" in fi and fi["detailed_description"]:
            return fi["detailed_description"][:200]
        # Also check plain "description" in final_interpretation
        if "description" in fi and fi["description"]:
            return fi["description"][:200]

    # Path 2: final_summary
    if "final_summary" in data and data["final_summary"]:
        return data["final_summary"]

    # Path 3: executive_summary
    if "executive_summary" in data and data["executive_summary"]:
        return data["executive_summary"][:200]

    # Path 4: top-level description (if no other summary)
    if "description" in data and data["description"]:
        return data["description"][:200]

    return ""


def extract_interpretation(feature: int) -> None:
    """Extract interpretation from results.json for a feature.

    Uses normalizer functions to check all known JSON structure variants.
    """
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

        # Use normalizer functions to extract from any structure
        label = _extract_label(data)
        verdict = _extract_verdict(data)
        summary = _extract_summary(data)

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

    # find-unverified
    find_unverified_parser = subparsers.add_parser("find-unverified",
        help="Find features with interpretations but no verification")
    find_unverified_parser.add_argument("--sort-by", required=True,
        choices=["rank_control", "rank_nocontrol"],
        help="Column to sort by")
    find_unverified_parser.add_argument("--limit", type=int, required=True,
        help="Maximum number of features to return")

    # find-auditable
    find_auditable_parser = subparsers.add_parser("find-auditable",
        help="Find features with interpretations that need auditing")
    find_auditable_parser.add_argument("--sort-by", default="rank_nocontrol",
        choices=["rank_control", "rank_nocontrol"],
        help="Column to sort by (default: rank_nocontrol)")
    find_auditable_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")

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

    # update-verify
    verify_parser = subparsers.add_parser("update-verify",
        help="Update verification status for a feature")
    verify_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    verify_parser.add_argument("--status", required=True,
        choices=["PASS", "WARN", "FAIL"],
        help="Verification status")
    verify_parser.add_argument("--reason", default="",
        help="Reason for WARN/FAIL status (optional)")

    # update-audit
    audit_parser = subparsers.add_parser("update-audit",
        help="Update audit status for a feature")
    audit_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    audit_parser.add_argument("--status", required=True,
        help="Audit status (COMPLETE, SCHEMA:..., INCOMPLETE:..., PROCESS_INCOMPLETE:...)")

    # force-update-audit
    force_audit_parser = subparsers.add_parser("force-update-audit",
        help="Force update audit status (overwrites existing)")
    force_audit_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    force_audit_parser.add_argument("--status", required=True,
        help="Audit status (COMPLETE, SCHEMA:..., INCOMPLETE:..., PROCESS_INCOMPLETE:...)")

    # update-fix-status
    fix_status_parser = subparsers.add_parser("update-fix-status",
        help="Update fix status for a feature")
    fix_status_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    fix_status_parser.add_argument("--status", required=True,
        choices=["schema_fix", "synthesis_fix", "schema+synthesis", "unsalvageable"],
        help="Fix status")

    # find-fixable
    find_fixable_parser = subparsers.add_parser("find-fixable",
        help="Find features with INCOMPLETE/SCHEMA status that may be fixable")
    find_fixable_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")

    # backup-csv
    subparsers.add_parser("backup-csv", help="Create CSV backup")

    # timestamp
    subparsers.add_parser("timestamp", help="Output current UTC timestamp (ISO 8601)")

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

    # batch-summary
    summary_parser = subparsers.add_parser("batch-summary",
        help="Generate and save batch verification summary")
    summary_parser.add_argument("--features", required=True,
        help="Comma-separated list of feature IDs that were verified")

    args = parser.parse_args()

    if args.command == "find-uninterpreted":
        find_uninterpreted(args.sort_by, args.limit)
    elif args.command == "find-unverified":
        find_unverified(args.sort_by, args.limit)
    elif args.command == "find-auditable":
        find_auditable(args.sort_by, args.limit)
    elif args.command == "check-existing":
        features = [int(x.strip()) for x in args.features.split(",")]
        check_existing(features)
    elif args.command == "update-csv":
        update_csv(args.feature, args.interpretation)
    elif args.command == "update-verify":
        update_verify(args.feature, args.status, args.reason)
    elif args.command == "update-audit":
        update_audit(args.feature, args.status)
    elif args.command == "force-update-audit":
        force_update_audit(args.feature, args.status)
    elif args.command == "update-fix-status":
        update_fix_status(args.feature, args.status)
    elif args.command == "find-fixable":
        find_fixable(args.limit)
    elif args.command == "backup-csv":
        backup_csv()
    elif args.command == "timestamp":
        timestamp()
    elif args.command == "ensure-output-dir":
        ensure_output_dir(args.feature)
    elif args.command == "extract-interpretation":
        extract_interpretation(args.feature)
    elif args.command == "batch-summary":
        features = [int(x.strip()) for x in args.features.split(",")]
        batch_summary(features)


if __name__ == "__main__":
    main()
