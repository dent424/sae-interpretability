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
    py -3.12 batch_utils.py classify-feature --feature 17240
    py -3.12 batch_utils.py update-category --feature 17240 --category "Semantic > Temporal"
    py -3.12 batch_utils.py force-update-category --feature 17240 --category "Semantic > Temporal"
    py -3.12 batch_utils.py find-classifiable --limit 100
    py -3.12 batch_utils.py find-by-category --category "Lexical > Specific Tokens" --limit 50
    py -3.12 batch_utils.py category-summary
    py -3.12 batch_utils.py stats
    py -3.12 batch_utils.py extract-example-summary --feature 13235
    py -3.12 batch_utils.py find-features-needing-example --limit 20
    py -3.12 batch_utils.py batch-update-feature-groups --updates '[{"feature": 123, "example": "...", "executive_summary": "..."}]'
    py -3.12 batch_utils.py update-scores --updates '[{"feature": 123, "paralinguistic_score": 7, "general_interest_score": 5}]'
    py -3.12 batch_utils.py find-unscored --limit 10
    py -3.12 batch_utils.py get-unscored-batch --limit 5
    py -3.12 batch_utils.py score-stats
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
FEATURE_GROUPS_CSV = SCRIPT_DIR / "output" / "Feature_groups.csv"


# =============================================================================
# Category Classification Keywords
# Based on textual paralanguage research: "characters, formatting, and words
# that create meaning beyond semantic content"
# =============================================================================

PARALINGUISTIC_KEYWORDS = [
    # Punctuation and symbols
    "punctuation", "exclamation", "ellipsis", "dash", "hyphen", "colon pattern",
    "quotation", "quote marks", "parentheses", "brackets", "asterisk",
    # Formatting and structure
    "formatting", "whitespace", "paragraph", "line break", "newline", "blank line",
    "spacing", "layout", "typography", "structural", "double-newline",
    # Lists and markers
    "list", "bullet", "numbered", "label marker", "heading", "section marker",
    # Emphasis patterns
    "all caps", "repetition", "emphasis pattern", "capitalization pattern",
    # Discourse structure (non-semantic)
    "discourse marker followed by", "labeling pattern", "transition marker",
    "paragraph boundary", "sentence boundary", "line-start",
]

SEMANTIC_KEYWORDS = [
    # Meaning and evaluation
    "meaning", "evaluative", "sentiment", "opinion", "judgment", "preference",
    "evaluation", "comparison", "recommendation", "persuasion",
    # Named entities and domain
    "named entity", "business name", "restaurant name", "place name", "person name",
    "domain", "topic", "concept",
    # Discourse function (semantic)
    "narrative", "dialogue", "meta-commentary", "announcement", "epistemic",
    "certainty", "belief", "hedging", "intensifier", "emphasis meaning",
    # Relations
    "negation pattern", "coordination", "enumeration", "anticipation", "excitement",
]

LEXICAL_KEYWORDS = [
    # Word-level features
    "token", "word", "lexical", "lexeme", "vocabulary", "word family",
    # Parts of speech
    "verb", "noun", "adjective", "adverb", "pronoun", "determiner", "preposition",
    "copula", "auxiliary", "conjunction", "particle",
    # Morphology
    "morphology", "morpheme", "prefix", "suffix", "subword", "conjugation",
    "agreement", "tense", "singular", "plural", "possessive",
    # Tokenization
    "bpe", "tokenization", "token detector", "token pattern",
    # Syntax
    "syntax", "syntactic", "phrase", "clause", "complement", "infinitive",
    # Person
    "first-person", "second-person", "third-person",
    # N-grams and patterns
    "collocation", "bigram", "trigram", "compound",
    # Language-specific
    "spanish", "foreign", "contraction",
    # Temporal
    "temporal adverb", "time word",
]


def _classify_text(text: str) -> dict:
    """Classify text based on keyword matching.

    Returns dict with counts and confidence for each category.
    """
    if not text:
        return {"Paralinguistic": 0, "Semantic": 0, "Lexical": 0}

    text_lower = text.lower()

    counts = {
        "Paralinguistic": 0,
        "Semantic": 0,
        "Lexical": 0,
    }

    # Count keyword matches
    for kw in PARALINGUISTIC_KEYWORDS:
        if kw.lower() in text_lower:
            counts["Paralinguistic"] += 1

    for kw in SEMANTIC_KEYWORDS:
        if kw.lower() in text_lower:
            counts["Semantic"] += 1

    for kw in LEXICAL_KEYWORDS:
        if kw.lower() in text_lower:
            counts["Lexical"] += 1

    return counts


def classify_feature(feature: int) -> None:
    """Classify a feature based on description text analysis.

    Reads results.json, extracts description/label/category fields,
    counts keyword matches for each category, and determines classification.
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

        # Collect all text to analyze
        texts_to_analyze = []

        # Primary: description field (most detailed)
        if data.get("description"):
            texts_to_analyze.append(data["description"])

        # Secondary: label
        if data.get("label"):
            texts_to_analyze.append(data["label"])

        # Tertiary: category field (often has structural hints)
        if data.get("category"):
            texts_to_analyze.append(data["category"])

        # Also check executive_summary and linguistic_function
        if data.get("executive_summary"):
            texts_to_analyze.append(data["executive_summary"])
        if data.get("linguistic_function"):
            texts_to_analyze.append(data["linguistic_function"])

        # Combine all text
        combined_text = " ".join(texts_to_analyze)

        # Classify
        counts = _classify_text(combined_text)

        # Determine winner(s)
        max_count = max(counts.values())

        if max_count == 0:
            # No keywords matched - mark as Unknown
            category = "Unknown"
            confidence = 0.0
            matched_keywords = []
        else:
            # Find categories with max count
            winners = [cat for cat, count in counts.items() if count == max_count]

            if len(winners) == 1:
                category = winners[0]
                # Confidence based on how dominant the winner is
                total = sum(counts.values())
                confidence = counts[category] / total if total > 0 else 0.0
            else:
                # Multiple categories tied - list them
                category = ",".join(sorted(winners))
                confidence = 0.5  # Lower confidence for ties

            # Find which keywords matched (for debugging)
            matched_keywords = []
            text_lower = combined_text.lower()
            all_keywords = [
                (kw, "Paralinguistic") for kw in PARALINGUISTIC_KEYWORDS
            ] + [
                (kw, "Semantic") for kw in SEMANTIC_KEYWORDS
            ] + [
                (kw, "Lexical") for kw in LEXICAL_KEYWORDS
            ]
            for kw, cat in all_keywords:
                if kw.lower() in text_lower:
                    matched_keywords.append(f"{kw} ({cat})")

        print(json.dumps({
            "status": "success",
            "feature": feature,
            "category": category,
            "confidence": round(confidence, 2),
            "counts": counts,
            "matched_keywords": matched_keywords[:10],  # Limit to first 10
            "analyzed_text_length": len(combined_text)
        }, indent=2))

    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error",
            "feature": feature,
            "error": f"Malformed JSON: {e}"
        }))
        sys.exit(1)


def validate_category(category: str) -> bool:
    """Validate category string format.

    Accepts:
    - Main categories: Paralinguistic, Semantic, Lexical, Unknown
    - With subcategory: "Category > Subcategory" (e.g., "Semantic > Temporal")
    - Legacy combinations: "Lexical,Paralinguistic"

    Returns True if valid, False otherwise.
    """
    valid_main = ["Paralinguistic", "Semantic", "Lexical", "Unknown"]

    # Check for subcategory format "Category > Subcategory"
    if " > " in category:
        main_cat = category.split(" > ")[0].strip()
        return main_cat in valid_main

    # Check for legacy combination format "Cat1,Cat2"
    if "," in category:
        parts = [c.strip() for c in category.split(",")]
        return all(p in valid_main for p in parts)

    # Simple category
    return category in valid_main


def update_category(feature: int, category: str) -> None:
    """Update/add category column in CSV for a feature.

    Adds category column after interpretation if not present.
    Only fills empty cells - never overwrites existing values.

    Valid formats:
    - Simple: Paralinguistic, Semantic, Lexical, Unknown
    - With subcategory: "Category > Subcategory" (e.g., "Semantic > Temporal")
    - Legacy combinations: "Lexical,Paralinguistic"
    """
    if not validate_category(category):
        print(f"ERROR: invalid category format '{category}'", file=sys.stderr)
        print("Valid formats: 'Category', 'Category > Subcategory', or 'Cat1,Cat2'", file=sys.stderr)
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

        # Add category column after interpretation if it doesn't exist
        columns_added = []
        if "category" not in fieldnames:
            # Find interpretation column index
            if "interpretation" in fieldnames:
                idx = fieldnames.index("interpretation") + 1
                fieldnames.insert(idx, "category")
            else:
                fieldnames.append("category")
            columns_added.append("category")

        for row in reader:
            # Ensure category column exists in row
            if "category" not in row:
                row["category"] = ""

            if int(float(row["feature_index"])) == feature:
                found = True
                # Only update if cell is empty
                if row["category"].strip():
                    already_set = True
                else:
                    row["category"] = category
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
            "reason": "category already has a value - not overwriting"
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
        "category": category,
        "columns_added": columns_added if columns_added else None
    }))


def force_update_category(feature: int, category: str) -> None:
    """Force update category column in CSV for a feature, overwriting existing value.

    Unlike update_category, this WILL overwrite existing values.
    Used for reclassification when switching from keyword to LLM-based classification.

    Valid formats:
    - Simple: Paralinguistic, Semantic, Lexical, Unknown
    - With subcategory: "Category > Subcategory" (e.g., "Semantic > Temporal")
    """
    if not validate_category(category):
        print(f"ERROR: invalid category format '{category}'", file=sys.stderr)
        print("Valid formats: 'Category', 'Category > Subcategory', or 'Cat1,Cat2'", file=sys.stderr)
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Read all rows
    rows = []
    fieldnames = None
    found = False
    old_category = ""

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add category column after interpretation if it doesn't exist
        columns_added = []
        if "category" not in fieldnames:
            if "interpretation" in fieldnames:
                idx = fieldnames.index("interpretation") + 1
                fieldnames.insert(idx, "category")
            else:
                fieldnames.append("category")
            columns_added.append("category")

        for row in reader:
            # Ensure category column exists in row
            if "category" not in row:
                row["category"] = ""

            if int(float(row["feature_index"])) == feature:
                found = True
                old_category = row["category"]
                # Force overwrite
                row["category"] = category
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
        "category": category,
        "old_category": old_category if old_category else None,
        "columns_added": columns_added if columns_added else None
    }))


def find_classifiable(limit: int, include_classified: bool = False) -> None:
    """Find features with results.json that can be classified.

    Finds features where:
    - interpretation column is NOT empty (has been interpreted)
    - results.json exists in output/interpretations/feature<ID>/
    - If include_classified=False (default): category column IS empty
    - If include_classified=True: includes features with existing categories
    """
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interpretation = row.get("interpretation", "").strip()
            category = row.get("category", "").strip()

            # Must have interpretation
            if not interpretation:
                continue

            # Skip if already classified (unless include_classified=True)
            if category and not include_classified:
                continue

            try:
                feature_id = int(float(row["feature_index"]))
                # Check if results.json exists
                results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                if results_path.exists():
                    rows.append({
                        "feature_index": feature_id,
                        "rank_control": int(row.get("rank_control", 0)),
                        "rank_nocontrol": int(row.get("rank_nocontrol", 0)),
                        "interpretation": interpretation[:50] + "..." if len(interpretation) > 50 else interpretation
                    })
            except (ValueError, KeyError):
                continue

    # Sort by rank_nocontrol ascending
    rows.sort(key=lambda x: x["rank_nocontrol"])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "total_classifiable": len(rows),
        "selected_count": len(selected),
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def find_by_category(category: str, limit: int) -> None:
    """Find features with a specific category for reclassification.

    Args:
        category: Category string to match (exact or partial).
                  Examples: "Lexical > Specific Tokens", "Semantic", "Paralinguistic > Punctuation"
        limit: Maximum number of features to return.

    Finds features where:
    - category column matches the specified category (exact or starts with)
    - results.json exists in output/interpretations/feature<ID>/
    """
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_category = row.get("category", "").strip()

            # Match exact or prefix (e.g., "Lexical" matches "Lexical > Specific Tokens")
            if not row_category:
                continue

            # Exact match or the category starts with the search term
            matches = (
                row_category == category or
                row_category.startswith(category + " >") or
                row_category.startswith(category + ">")
            )

            if matches:
                try:
                    feature_id = int(float(row["feature_index"]))
                    # Check if results.json exists
                    results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                    if results_path.exists():
                        rows.append({
                            "feature_index": feature_id,
                            "rank_control": int(row.get("rank_control", 0)),
                            "rank_nocontrol": int(row.get("rank_nocontrol", 0)),
                            "category": row_category,
                            "interpretation": row.get("interpretation", "")[:50] + "..." if len(row.get("interpretation", "")) > 50 else row.get("interpretation", "")
                        })
                except (ValueError, KeyError):
                    continue

    # Sort by rank_nocontrol ascending
    rows.sort(key=lambda x: x["rank_nocontrol"])

    # Take top N
    selected = rows[:limit]

    # Output as JSON for easy parsing
    result = {
        "category_filter": category,
        "total_matching": len(rows),
        "selected_count": len(selected),
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def clear_all_categories() -> None:
    """Clear the category column for all features in the CSV."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    rows = []
    fieldnames = None
    cleared = 0

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            if row.get("category", "").strip():
                cleared += 1
                row["category"] = ""
            rows.append(row)

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"status": "success", "cleared": cleared, "total_rows": len(rows)}))


def category_summary() -> None:
    """Generate summary statistics of category classifications.

    Counts features in each category and calculates percentages.
    Supports subcategory format "Category > Subcategory".
    """
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    # Main category counts
    main_counts = {
        "Paralinguistic": 0,
        "Semantic": 0,
        "Lexical": 0,
        "Mixed": 0,
        "Unknown": 0,
        "Unclassified": 0,
    }

    # Subcategory breakdown: {"Paralinguistic": {"Punctuation": 5, "Formatting": 3}, ...}
    subcategory_counts = {
        "Paralinguistic": {},
        "Semantic": {},
        "Lexical": {},
    }

    total_interpreted = 0
    total_classified = 0
    mixed_details = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interpretation = row.get("interpretation", "").strip()
            category = row.get("category", "").strip()

            if interpretation:
                total_interpreted += 1

                if not category:
                    main_counts["Unclassified"] += 1
                elif category == "Unknown":
                    main_counts["Unknown"] += 1
                    total_classified += 1
                elif " > " in category:
                    # New subcategory format: "Category > Subcategory"
                    parts = category.split(" > ", 1)
                    main_cat = parts[0].strip()
                    sub_cat = parts[1].strip() if len(parts) > 1 else "Unspecified"

                    if main_cat in main_counts:
                        main_counts[main_cat] += 1
                        total_classified += 1

                        # Track subcategory
                        if main_cat in subcategory_counts:
                            subcategory_counts[main_cat][sub_cat] = \
                                subcategory_counts[main_cat].get(sub_cat, 0) + 1
                    else:
                        main_counts["Unknown"] += 1
                        total_classified += 1
                elif "," in category:
                    # Legacy mixed category format
                    main_counts["Mixed"] += 1
                    total_classified += 1
                    mixed_details.append({
                        "feature": int(float(row["feature_index"])),
                        "categories": category
                    })
                elif category in main_counts:
                    # Simple category without subcategory
                    main_counts[category] += 1
                    total_classified += 1
                else:
                    # Unknown category value
                    main_counts["Unknown"] += 1
                    total_classified += 1

    # Calculate percentages (excluding unclassified)
    percentages = {}
    if total_classified > 0:
        for cat, count in main_counts.items():
            if cat != "Unclassified":
                percentages[cat] = round(100.0 * count / total_classified, 1)

    # Build detailed breakdown with subcategories
    detailed_breakdown = {}
    for main_cat in ["Paralinguistic", "Semantic", "Lexical"]:
        subcats = subcategory_counts.get(main_cat, {})
        # Sort subcategories by count descending
        sorted_subcats = dict(sorted(subcats.items(), key=lambda x: -x[1]))
        detailed_breakdown[main_cat] = {
            "total": main_counts[main_cat],
            "subcategories": sorted_subcats if sorted_subcats else None
        }

    result = {
        "total_interpreted": total_interpreted,
        "total_classified": total_classified,
        "total_unclassified": main_counts["Unclassified"],
        "counts": main_counts,
        "percentages": percentages,
        "detailed_breakdown": detailed_breakdown,
        "mixed_features": mixed_details[:10] if mixed_details else None
    }
    print(json.dumps(result, indent=2))


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
                        "feature_index": int(float(row["feature_index"])),
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
                    feature_id = int(float(row["feature_index"]))
                    # Check if results.json exists
                    results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                    if results_path.exists():
                        rows.append({
                            "feature_index": feature_id,
                            "rank_control": int(float(row["rank_control"])),
                            "rank_nocontrol": int(float(row["rank_nocontrol"])),
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
                    feature_id = int(float(row["feature_index"]))
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
            if int(float(row["feature_index"])) == feature:
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

            if row["feature_index"].strip() and int(float(row["feature_index"])) == feature:
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

            if int(float(row["feature_index"])) == feature:
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

    Valid status prefixes: COMPLETE, SCHEMA, INCOMPLETE, PROCESS_INCOMPLETE, CLEAR
    Status can include details after colon (e.g., "SCHEMA:label,description")
    Use CLEAR to reset audit_status to empty (for re-auditing).
    """
    # Validate status prefix
    valid_prefixes = ("COMPLETE", "SCHEMA", "INCOMPLETE", "PROCESS_INCOMPLETE", "CLEAR")
    prefix = status.split(":")[0]
    if prefix not in valid_prefixes:
        print(f"ERROR: status must start with {valid_prefixes}, got '{status}'", file=sys.stderr)
        sys.exit(1)

    # CLEAR means reset to empty
    if status == "CLEAR":
        status = ""

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

            if int(float(row["feature_index"])) == feature:
                found = True
                old_status = row["audit_status"]
                # Force overwrite (empty status clears both fields)
                row["audit_status"] = status
                row["audit_date"] = "" if status == "" else datetime.now(timezone.utc).isoformat()
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

            if int(float(row["feature_index"])) == feature:
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
                    feature_id = int(float(row["feature_index"]))
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


def stats() -> None:
    """Generate summary statistics for the feature CSV."""
    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    total = 0
    with_interpretation = 0
    with_complete_audit = 0
    with_verification = 0
    categories = {}

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1

            # Count interpretations
            if row.get("interpretation", "").strip():
                with_interpretation += 1

            # Count COMPLETE audits
            if row.get("audit_status", "").strip() == "COMPLETE":
                with_complete_audit += 1

            # Count verifications (any status)
            if row.get("verify_status", "").strip():
                with_verification += 1

            # Count categories
            category = row.get("category", "").strip()
            if category:
                categories[category] = categories.get(category, 0) + 1

    result = {
        "total_features": total,
        "with_interpretation": with_interpretation,
        "with_complete_audit": with_complete_audit,
        "with_verification": with_verification,
        "categories": categories
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
            if not row.get("feature_index"):
                continue
            feat_id = int(float(row["feature_index"]))
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
        if not row.get("feature_index"):
            continue
        feat_id = int(float(row["feature_index"]))
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


def extract_example_summary(feature: int) -> None:
    """Extract example and executive_summary from results.json for a feature.

    Output JSON: {feature, example, executive_summary, status}
    Falls back to report.md if JSON parsing fails.
    """
    results_path = OUTPUT_DIR / f"feature{feature}" / "results.json"
    report_path = OUTPUT_DIR / f"feature{feature}" / "report.md"

    example = ""
    executive_summary = ""
    source = "none"

    # Try results.json first
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract executive_summary
            if data.get("executive_summary"):
                executive_summary = data["executive_summary"]

            # Extract example from top_activations[0]["context"]
            top_activations = data.get("top_activations", [])
            if top_activations and len(top_activations) > 0:
                first_activation = top_activations[0]
                if isinstance(first_activation, dict) and "context" in first_activation:
                    example = first_activation["context"]

            source = "results.json"
        except (json.JSONDecodeError, KeyError) as e:
            # Fall back to report.md
            pass

    # Fallback to report.md if we didn't get what we need
    if (not example or not executive_summary) and report_path.exists():
        try:
            content = report_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Look for Executive Summary section
            in_exec_summary = False
            exec_lines = []
            for line in lines:
                if "## Executive Summary" in line or "**Executive Summary**" in line:
                    in_exec_summary = True
                    continue
                if in_exec_summary:
                    if line.startswith("##") or line.startswith("**") and ":" in line:
                        break
                    if line.strip():
                        exec_lines.append(line.strip())

            if exec_lines and not executive_summary:
                executive_summary = " ".join(exec_lines)
                source = "report.md" if source == "none" else source + "+report.md"

            # Look for example in Top Activations section or Key Examples
            if not example:
                for i, line in enumerate(lines):
                    if "context" in line.lower() and "..." in line:
                        # Try to extract context snippet
                        if "**" in line:
                            # Format: ...context** token**...
                            example = line.strip()
                            source = "report.md" if source == "none" else source + "+report.md"
                            break

        except Exception:
            pass

    # Output result
    result = {
        "feature": feature,
        "example": example,
        "executive_summary": executive_summary,
        "source": source,
        "status": "success" if (example or executive_summary) else "no_data"
    }
    print(json.dumps(result, indent=2))


def find_features_needing_example(limit: int) -> None:
    """Find features with empty example column but existing results.json.

    CRITICAL: Only returns features where BOTH example AND executive_summary are empty.
    """
    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            example = (row.get("example") or "").strip()
            exec_summary = (row.get("executive_summary") or "").strip()

            # Only return features where BOTH are empty
            if example or exec_summary:
                continue

            try:
                feature_id = int(float(row["feature_index"]))
                # Check if results.json exists
                results_path = OUTPUT_DIR / f"feature{feature_id}" / "results.json"
                if results_path.exists():
                    rows.append({
                        "feature_index": feature_id,
                        "category": row.get("category", ""),
                        "subcategory": row.get("subcategory", ""),
                        "has_results_json": True
                    })
            except (ValueError, KeyError):
                continue

    # Take up to limit
    selected = rows[:limit]

    result = {
        "total_needing_data": len(rows),
        "selected_count": len(selected),
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def batch_update_feature_groups(updates_json: str) -> None:
    """Batch update Feature_groups.csv with multiple feature updates.

    Takes JSON string or file path of list of {feature, example, executive_summary} dicts.
    If updates_json starts with '@', it's treated as a file path to read JSON from.

    CRITICAL: Only updates cells that are CURRENTLY EMPTY.
    Never overwrites existing values.
    """
    # Check if it's a file path
    if updates_json.startswith("@"):
        file_path = updates_json[1:]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                updates = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(json.dumps({
                "status": "error",
                "error": f"Failed to read JSON from file: {e}"
            }))
            sys.exit(1)
    else:
        try:
            updates = json.loads(updates_json)
        except json.JSONDecodeError as e:
            print(json.dumps({
                "status": "error",
                "error": f"Invalid JSON: {e}"
            }))
            sys.exit(1)

    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    # Build lookup of updates by feature ID
    update_lookup = {u["feature"]: u for u in updates}

    # Read all rows
    rows = []
    fieldnames = None
    updated_count = 0
    skipped_count = 0
    updated_features = []
    skipped_features = []

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        for row in reader:
            try:
                feature_id = int(float(row["feature_index"]))
            except (ValueError, KeyError):
                rows.append(row)
                continue

            if feature_id in update_lookup:
                update = update_lookup[feature_id]
                updated_this_row = False

                # Only update example if current value is empty
                current_example = (row.get("example") or "").strip()
                new_example = (update.get("example") or "").strip()
                if not current_example and new_example:
                    row["example"] = new_example
                    updated_this_row = True

                # Only update executive_summary if current value is empty
                current_summary = (row.get("executive_summary") or "").strip()
                new_summary = (update.get("executive_summary") or "").strip()
                if not current_summary and new_summary:
                    row["executive_summary"] = new_summary
                    updated_this_row = True

                if updated_this_row:
                    updated_count += 1
                    updated_features.append(feature_id)
                else:
                    skipped_count += 1
                    skipped_features.append(feature_id)

            rows.append(row)

    # Write back
    with open(FEATURE_GROUPS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "updated_features": updated_features,
        "skipped_features": skipped_features
    }, indent=2))


def update_scores(updates_json: str) -> None:
    """Batch update Feature_groups.csv with paralinguistic_score and general_interest_score.

    Takes JSON string or file path of list of {feature, paralinguistic_score, general_interest_score} dicts.
    If updates_json starts with '@', it's treated as a file path to read JSON from.

    Adds the score columns if they don't exist.
    Only updates cells that are CURRENTLY EMPTY - never overwrites existing scores.
    """
    # Check if it's a file path
    if updates_json.startswith("@"):
        file_path = updates_json[1:]
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                updates = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(json.dumps({
                "status": "error",
                "error": f"Failed to read JSON from file: {e}"
            }))
            sys.exit(1)
    else:
        try:
            updates = json.loads(updates_json)
        except json.JSONDecodeError as e:
            print(json.dumps({
                "status": "error",
                "error": f"Invalid JSON: {e}"
            }))
            sys.exit(1)

    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    # Build lookup of updates by feature ID
    update_lookup = {u["feature"]: u for u in updates}

    # Read all rows
    rows = []
    fieldnames = None
    updated_count = 0
    skipped_count = 0
    updated_features = []
    skipped_features = []
    columns_added = []

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        # Add score columns if they don't exist (after executive_summary)
        if "paralinguistic_score" not in fieldnames:
            fieldnames.append("paralinguistic_score")
            columns_added.append("paralinguistic_score")
        if "general_interest_score" not in fieldnames:
            fieldnames.append("general_interest_score")
            columns_added.append("general_interest_score")

        for row in reader:
            # Ensure score columns exist in row
            if "paralinguistic_score" not in row:
                row["paralinguistic_score"] = ""
            if "general_interest_score" not in row:
                row["general_interest_score"] = ""

            try:
                feature_id = int(float(row["feature_index"]))
            except (ValueError, KeyError):
                rows.append(row)
                continue

            if feature_id in update_lookup:
                update = update_lookup[feature_id]
                updated_this_row = False

                # Only update paralinguistic_score if current value is empty
                current_para = (row.get("paralinguistic_score") or "").strip()
                new_para = update.get("paralinguistic_score")
                if not current_para and new_para is not None:
                    row["paralinguistic_score"] = str(new_para)
                    updated_this_row = True

                # Only update general_interest_score if current value is empty
                current_gi = (row.get("general_interest_score") or "").strip()
                new_gi = update.get("general_interest_score")
                if not current_gi and new_gi is not None:
                    row["general_interest_score"] = str(new_gi)
                    updated_this_row = True

                if updated_this_row:
                    updated_count += 1
                    updated_features.append(feature_id)
                else:
                    skipped_count += 1
                    skipped_features.append(feature_id)

            rows.append(row)

    # Write back
    with open(FEATURE_GROUPS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "updated_count": updated_count,
        "skipped_count": skipped_count,
        "updated_features": updated_features,
        "skipped_features": skipped_features,
        "columns_added": columns_added if columns_added else None
    }, indent=2))


def find_unscored(limit: int) -> None:
    """Find features with empty paralinguistic_score or general_interest_score.

    Returns features that need scoring, ordered by feature_index.
    """
    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_features = 0
    scored_count = 0

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_features += 1
            para_score = (row.get("paralinguistic_score") or "").strip()
            gi_score = (row.get("general_interest_score") or "").strip()

            # Check if either score is missing
            if not para_score or not gi_score:
                try:
                    feature_id = int(float(row["feature_index"]))
                    exec_summary = (row.get("executive_summary") or "")[:100]
                    rows.append({
                        "feature_index": feature_id,
                        "category": row.get("category", ""),
                        "has_paralinguistic": bool(para_score),
                        "has_general_interest": bool(gi_score),
                        "executive_summary_preview": exec_summary + "..." if len(exec_summary) == 100 else exec_summary
                    })
                except (ValueError, KeyError):
                    continue
            else:
                scored_count += 1

    # Sort by feature_index
    rows.sort(key=lambda x: x["feature_index"])

    # Take up to limit
    selected = rows[:limit]

    result = {
        "total_features": total_features,
        "scored_count": scored_count,
        "unscored_count": len(rows),
        "selected_count": len(selected),
        "features": [r["feature_index"] for r in selected],
        "details": selected
    }
    print(json.dumps(result, indent=2))


def get_unscored_batch(limit: int) -> None:
    """Get next batch of unscored features with FULL executive summaries.

    Outputs features in a format ready for scoring agents.
    """
    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    rows = []
    total_features = 0
    scored_count = 0

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_features += 1
            para_score = (row.get("paralinguistic_score") or "").strip()
            gi_score = (row.get("general_interest_score") or "").strip()

            if para_score and gi_score:
                scored_count += 1
                continue

            try:
                feature_id = int(float(row["feature_index"]))
                rows.append({
                    "feature_index": feature_id,
                    "executive_summary": row.get("executive_summary", "")
                })
            except (ValueError, KeyError):
                continue

    # Sort by feature_index and take limit
    rows.sort(key=lambda x: x["feature_index"])
    selected = rows[:limit]

    # Output in easy-to-parse format
    # Sanitize output for Windows console (cp1252 can't handle all Unicode)
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode('ascii'))

    safe_print(f"=== Progress: {scored_count}/{total_features} scored, {len(rows)} remaining ===")
    print()
    for item in selected:
        safe_print(f"=== Feature {item['feature_index']} ===")
        safe_print(item["executive_summary"])
        print()


def score_stats() -> None:
    """Show distribution of paralinguistic and general_interest scores."""
    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    para_counts = {}
    gi_counts = {}
    total = 0
    scored = 0

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            para = (row.get("paralinguistic_score") or "").strip()
            gi = (row.get("general_interest_score") or "").strip()

            if para:
                para_counts[int(para)] = para_counts.get(int(para), 0) + 1
            if gi:
                gi_counts[int(gi)] = gi_counts.get(int(gi), 0) + 1
            if para and gi:
                scored += 1

    result = {
        "total_features": total,
        "scored_count": scored,
        "unscored_count": total - scored,
        "paralinguistic_distribution": dict(sorted(para_counts.items())),
        "general_interest_distribution": dict(sorted(gi_counts.items()))
    }
    print(json.dumps(result, indent=2))


def clear_scores(features_json: str) -> None:
    """Clear paralinguistic_score and general_interest_score for specified features.

    Takes comma-separated feature IDs or JSON array.
    This allows re-scoring features (e.g., switching from haiku to opus).
    """
    # Parse feature list
    if features_json.startswith("["):
        try:
            feature_ids = json.loads(features_json)
        except json.JSONDecodeError as e:
            print(json.dumps({
                "status": "error",
                "error": f"Invalid JSON: {e}"
            }))
            sys.exit(1)
    else:
        # Comma-separated
        feature_ids = [int(x.strip()) for x in features_json.split(",")]

    if not FEATURE_GROUPS_CSV.exists():
        print(f"ERROR: CSV not found at {FEATURE_GROUPS_CSV}", file=sys.stderr)
        sys.exit(1)

    feature_set = set(feature_ids)
    rows = []
    fieldnames = None
    cleared_count = 0
    cleared_features = []
    not_found = []

    with open(FEATURE_GROUPS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)

        for row in reader:
            try:
                feature_id = int(float(row["feature_index"]))
            except (ValueError, KeyError):
                rows.append(row)
                continue

            if feature_id in feature_set:
                # Clear both score columns
                had_scores = bool((row.get("paralinguistic_score") or "").strip() or
                                  (row.get("general_interest_score") or "").strip())
                row["paralinguistic_score"] = ""
                row["general_interest_score"] = ""
                if had_scores:
                    cleared_count += 1
                    cleared_features.append(feature_id)
                feature_set.discard(feature_id)

            rows.append(row)

    not_found = list(feature_set)

    # Write back
    with open(FEATURE_GROUPS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "status": "success",
        "cleared_count": cleared_count,
        "cleared_features": sorted(cleared_features),
        "not_found": sorted(not_found)
    }, indent=2))


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

    # classify-feature
    classify_parser = subparsers.add_parser("classify-feature",
        help="Classify a feature based on description text analysis")
    classify_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to classify")

    # update-category
    category_parser = subparsers.add_parser("update-category",
        help="Update category column in CSV for a feature")
    category_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    category_parser.add_argument("--category", required=True,
        help="Category (e.g., 'Semantic > Temporal' or 'Lexical')")

    # force-update-category
    force_category_parser = subparsers.add_parser("force-update-category",
        help="Force update category (overwrites existing)")
    force_category_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to update")
    force_category_parser.add_argument("--category", required=True,
        help="Category (e.g., 'Semantic > Temporal' or 'Lexical')")

    # find-classifiable
    find_classifiable_parser = subparsers.add_parser("find-classifiable",
        help="Find features with results.json that can be classified")
    find_classifiable_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")
    find_classifiable_parser.add_argument("--include-classified", action="store_true",
        help="Include features that already have a category (for reclassification)")

    # find-by-category
    find_by_category_parser = subparsers.add_parser("find-by-category",
        help="Find features with a specific category for reclassification")
    find_by_category_parser.add_argument("--category", required=True,
        help="Category to search for (e.g., 'Lexical > Specific Tokens')")
    find_by_category_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")

    # clear-all-categories
    subparsers.add_parser("clear-all-categories",
        help="Clear the category column for all features")

    # category-summary
    subparsers.add_parser("category-summary",
        help="Generate summary statistics of category classifications")

    # stats
    subparsers.add_parser("stats", help="Show summary statistics for feature CSV")

    # extract-example-summary
    extract_example_parser = subparsers.add_parser("extract-example-summary",
        help="Extract example and executive_summary from results.json for a feature")
    extract_example_parser.add_argument("--feature", type=int, required=True,
        help="Feature ID to extract data for")

    # find-features-needing-example
    find_needing_example_parser = subparsers.add_parser("find-features-needing-example",
        help="Find features with empty example/executive_summary in Feature_groups.csv")
    find_needing_example_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")

    # batch-update-feature-groups
    batch_update_parser = subparsers.add_parser("batch-update-feature-groups",
        help="Batch update Feature_groups.csv with example and executive_summary")
    batch_update_parser.add_argument("--updates", required=True,
        help="JSON array of {feature, example, executive_summary} objects")

    # update-scores
    update_scores_parser = subparsers.add_parser("update-scores",
        help="Batch update Feature_groups.csv with paralinguistic_score and general_interest_score")
    update_scores_parser.add_argument("--updates", required=True,
        help="JSON array of {feature, paralinguistic_score, general_interest_score} objects")

    # find-unscored
    find_unscored_parser = subparsers.add_parser("find-unscored",
        help="Find features with empty scores in Feature_groups.csv")
    find_unscored_parser.add_argument("--limit", type=int, default=100,
        help="Maximum number of features to return (default: 100)")

    # get-unscored-batch
    get_batch_parser = subparsers.add_parser("get-unscored-batch",
        help="Get next batch of unscored features with full executive summaries")
    get_batch_parser.add_argument("--limit", type=int, default=5,
        help="Number of features to return (default: 5)")

    # score-stats
    subparsers.add_parser("score-stats",
        help="Show distribution of paralinguistic and general_interest scores")

    # clear-scores
    clear_scores_parser = subparsers.add_parser("clear-scores",
        help="Clear scores for specified features (to allow re-scoring)")
    clear_scores_parser.add_argument("--features", required=True,
        help="Comma-separated feature IDs or JSON array")

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
    elif args.command == "classify-feature":
        classify_feature(args.feature)
    elif args.command == "update-category":
        update_category(args.feature, args.category)
    elif args.command == "force-update-category":
        force_update_category(args.feature, args.category)
    elif args.command == "find-classifiable":
        find_classifiable(args.limit, getattr(args, 'include_classified', False))
    elif args.command == "find-by-category":
        find_by_category(args.category, args.limit)
    elif args.command == "clear-all-categories":
        clear_all_categories()
    elif args.command == "category-summary":
        category_summary()
    elif args.command == "stats":
        stats()
    elif args.command == "extract-example-summary":
        extract_example_summary(args.feature)
    elif args.command == "find-features-needing-example":
        find_features_needing_example(args.limit)
    elif args.command == "batch-update-feature-groups":
        batch_update_feature_groups(args.updates)
    elif args.command == "update-scores":
        update_scores(args.updates)
    elif args.command == "find-unscored":
        find_unscored(args.limit)
    elif args.command == "get-unscored-batch":
        get_unscored_batch(args.limit)
    elif args.command == "score-stats":
        score_stats()
    elif args.command == "clear-scores":
        clear_scores(args.features)


if __name__ == "__main__":
    main()
