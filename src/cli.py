"""CLI interface for SAE interpretability analysis.

Human-friendly command-line interface that calls the Modal backend.

Usage:
    python src/cli.py feature 16751 --examples 10
    python src/cli.py text "Why in the world would anyone eat here?"
    python src/cli.py batch 16751 11328 14292 --output features.csv
"""

import argparse
import json
import csv
import sys
from modal import Function


def get_interpreter():
    """Get reference to deployed Modal SAEInterpreter."""
    return Function.lookup("sae-interpretability", "SAEInterpreter")


def format_table(headers: list, rows: list, col_widths: list = None) -> str:
    """Format data as a simple ASCII table."""
    if not rows:
        return "(no data)"

    if col_widths is None:
        col_widths = [
            max(len(str(row[i])) for row in rows + [headers])
            for i in range(len(headers))
        ]

    # Header
    header_line = " | ".join(
        str(h).ljust(w) for h, w in zip(headers, col_widths)
    )
    separator = "-+-".join("-" * w for w in col_widths)

    # Rows
    row_lines = [
        " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        for row in rows
    ]

    return "\n".join([header_line, separator] + row_lines)


def cmd_feature(args):
    """Analyze a single feature."""
    interpreter = get_interpreter()
    feature_idx = args.feature_idx

    print(f"\n{'='*60}")
    print(f"Analyzing Feature {feature_idx}")
    print(f"{'='*60}")

    # Get stats
    print("\nFetching statistics...")
    stats = interpreter.get_feature_stats.remote(feature_idx)
    print(f"\nStatistics:")
    print(f"  Total activations: {stats['total_activations']:,}")
    print(f"  Activation rate: {stats['activation_rate']:.4%}")
    print(f"  Mean (when active): {stats['mean_when_active']:.3f}")
    print(f"  Max activation: {stats['max_activation']:.3f}")

    # Get top tokens
    print(f"\nFetching top tokens (scanning up to {args.max_samples:,} samples)...")
    top_tokens = interpreter.get_top_tokens.remote(
        feature_idx,
        top_k=args.top_k,
        max_samples=args.max_samples
    )

    print(f"\nTop {len(top_tokens)} Tokens:")
    headers = ["Token", "Count", "Mean Activation"]
    rows = [
        [f"'{t['token']}'", t['count'], f"{t['mean_activation']:.3f}"]
        for t in top_tokens
    ]
    print(format_table(headers, rows, [20, 10, 15]))

    # Get example contexts
    print(f"\nFetching {args.examples} example contexts...")
    contexts = interpreter.get_feature_contexts.remote(
        feature_idx,
        n_samples=args.examples
    )

    print(f"\nExample Contexts (sorted by activation):")
    for i, ctx in enumerate(contexts, 1):
        print(f"\n  [{i}] Activation: {ctx['activation']:.3f}")
        print(f"      Token: '{ctx['active_token']}'")
        print(f"      Context: {ctx['context'][:100]}...")
        print(f"      Review: {ctx['review_id']}")

    print(f"\n{'='*60}")


def cmd_text(args):
    """Analyze text through the GPT-2 → SAE pipeline."""
    interpreter = get_interpreter()
    text = args.text

    print(f"\n{'='*60}")
    print(f"Analyzing Text")
    print(f"{'='*60}")
    print(f"\nInput: {text[:100]}{'...' if len(text) > 100 else ''}")

    print("\nProcessing through GPT-2 → SAE pipeline...")
    result = interpreter.process_text.remote(text)

    print(f"\nTokens: {result['n_tokens']}")
    print(f"\nToken-by-token analysis:")

    for i, (token, features) in enumerate(zip(result['tokens'], result['top_features_per_token'])):
        if features:
            top_feat = features[0]
            print(f"  [{i:2d}] '{token}' → Feature {top_feat['feature_idx']} ({top_feat['activation']:.3f})")
        else:
            print(f"  [{i:2d}] '{token}' → (no active features)")

    # Summary of most active features across all tokens
    feature_counts = {}
    for token_features in result['top_features_per_token']:
        for feat in token_features:
            idx = feat['feature_idx']
            if idx not in feature_counts:
                feature_counts[idx] = {"count": 0, "max_act": 0}
            feature_counts[idx]["count"] += 1
            feature_counts[idx]["max_act"] = max(feature_counts[idx]["max_act"], feat["activation"])

    if feature_counts:
        print(f"\nMost Active Features (across all tokens):")
        sorted_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )[:10]
        headers = ["Feature", "Token Count", "Max Activation"]
        rows = [
            [idx, data["count"], f"{data['max_act']:.3f}"]
            for idx, data in sorted_features
        ]
        print(format_table(headers, rows))

    # If specific feature requested
    if args.feature:
        feat_str = str(args.feature)
        if feat_str in result['feature_activations']:
            activations = result['feature_activations'][feat_str]
            print(f"\nFeature {args.feature} activations per token:")
            for i, (token, act) in enumerate(zip(result['tokens'], activations)):
                if act > 0:
                    print(f"  [{i:2d}] '{token}': {act:.3f}")
        else:
            print(f"\nFeature {args.feature} did not activate on any token.")

    print(f"\n{'='*60}")


def cmd_batch(args):
    """Analyze multiple features and output to CSV."""
    interpreter = get_interpreter()
    feature_indices = args.feature_indices

    print(f"\n{'='*60}")
    print(f"Batch Analysis: {len(feature_indices)} features")
    print(f"{'='*60}")

    results = []

    for i, feature_idx in enumerate(feature_indices, 1):
        print(f"\n[{i}/{len(feature_indices)}] Analyzing feature {feature_idx}...")

        # Get stats
        stats = interpreter.get_feature_stats.remote(feature_idx)

        # Get top tokens
        top_tokens = interpreter.get_top_tokens.remote(feature_idx, top_k=5, max_samples=10000)

        # Format top tokens as string
        top_tokens_str = ", ".join([f"'{t['token']}'" for t in top_tokens[:3]])

        results.append({
            "feature_idx": feature_idx,
            "total_activations": stats["total_activations"],
            "activation_rate": f"{stats['activation_rate']:.4%}",
            "mean_when_active": f"{stats['mean_when_active']:.3f}",
            "max_activation": f"{stats['max_activation']:.3f}",
            "top_tokens": top_tokens_str,
            "label": "",  # To be filled by human/agent
            "notes": ""
        })

        print(f"    Total: {stats['total_activations']:,}, Top: {top_tokens_str}")

    # Output to CSV
    if args.output:
        with open(args.output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to: {args.output}")
    else:
        # Print as table
        print("\nResults:")
        headers = list(results[0].keys())
        rows = [list(r.values()) for r in results]
        print(format_table(headers, rows))

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SAE Interpretability CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py feature 16751 --examples 10
  python cli.py text "Why in the world would anyone eat here?"
  python cli.py text "Great food!" --feature 16751
  python cli.py batch 16751 11328 14292 --output features.csv
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Feature command
    feature_parser = subparsers.add_parser("feature", help="Analyze a single feature")
    feature_parser.add_argument("feature_idx", type=int, help="Feature index (0-24575)")
    feature_parser.add_argument("--examples", type=int, default=10, help="Number of example contexts")
    feature_parser.add_argument("--top-k", type=int, default=20, help="Number of top tokens to show")
    feature_parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to scan")

    # Text command
    text_parser = subparsers.add_parser("text", help="Analyze text through pipeline")
    text_parser.add_argument("text", type=str, help="Text to analyze")
    text_parser.add_argument("--feature", type=int, help="Specific feature to track")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Analyze multiple features")
    batch_parser.add_argument("feature_indices", type=int, nargs="+", help="Feature indices to analyze")
    batch_parser.add_argument("--output", "-o", type=str, help="Output CSV file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "feature":
        cmd_feature(args)
    elif args.command == "text":
        cmd_text(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()
