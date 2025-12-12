"""Agent runner for SAE interpretability analysis.

This script orchestrates Claude to analyze SAE features using the
hypothesis testing workflow.

Usage:
    python src/agent/run_agent.py --features 16751,11328 --output reports/
"""

import argparse
import asyncio
import os
from pathlib import Path

# System prompt for SAE analysis
SAE_ANALYSIS_PROMPT = """You are an expert in mechanistic interpretability analyzing Sparse Autoencoder (SAE) features extracted from GPT-2's residual stream (layer 8).

The SAE was trained on 432K Mexican restaurant reviews from Yelp. It has:
- 24,576 total features (768 dimensions × 32 expansion)
- Top-K=32 sparsity (exactly 32 features active per token)

For each feature you analyze, follow this hypothesis testing workflow:

## 1. GET OVERVIEW
Call `analyze_feature` to see:
- **Statistics**: How often and how strongly the feature activates
- **Top tokens**: Which words/subwords most commonly trigger this feature
- **Example contexts**: Real text examples where the feature fires

## 2. FORM HYPOTHESIS
Based on the patterns, hypothesize what the feature detects. Consider:
- **Semantic patterns**: Topics (food, service), sentiment, entities
- **Syntactic patterns**: Lists, punctuation, sentence structure
- **Lexical patterns**: Specific words, prefixes, suffixes, word boundaries

## 3. GENERATE TEST EXAMPLES
Create examples to test your hypothesis:

**Positive examples** (3-5): Text that SHOULD activate the feature
- If you think it detects "emphatic expressions", write emphatic sentences
- Vary the examples to test different aspects of your hypothesis

**Negative examples** (3-5): Text that should NOT activate the feature
- Similar context but lacking the key pattern
- This tests specificity of your interpretation

## 4. TEST EXAMPLES
Use `test_text` with `target_feature` set to check each example:
- Does the feature activate on positive examples? (It should)
- Does it stay inactive on negative examples? (It should)

## 5. ASSESS CONFIDENCE
Based on test results:
- **HIGH confidence**: Positive examples activate, negative don't
- **MEDIUM confidence**: Mostly works but some exceptions
- **LOW confidence**: Results contradict hypothesis → revise and repeat

## Report Format
For each feature, write a markdown report with:
1. Feature index and summary label (1-5 words)
2. Statistics (activation count, rate)
3. Your hypothesis
4. Test examples and results
5. Confidence level and reasoning
6. Notes on edge cases or related features

Be specific and evidence-based. If a hypothesis doesn't hold up to testing, revise it."""


async def analyze_features_with_sdk(feature_indices: list[int], output_dir: str):
    """
    Run Claude agent to analyze features using Claude Agent SDK.

    Args:
        feature_indices: List of feature indices to analyze
        output_dir: Directory to write reports
    """
    try:
        from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
        from agent.tools import create_mcp_server
    except ImportError:
        print("Error: claude_agent_sdk not installed.")
        print("Install with: pip install claude-agent-sdk")
        return

    # Create MCP server with our tools
    sae_server = create_mcp_server()
    if sae_server is None:
        print("Error: Could not create MCP server")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Configure agent
    options = ClaudeAgentOptions(
        system_prompt=SAE_ANALYSIS_PROMPT,
        mcp_servers={"sae": sae_server},
        allowed_tools=[
            "mcp__sae__analyze_feature",
            "mcp__sae__test_text",
            "Write"
        ],
        permission_mode="acceptEdits",
        cwd=output_dir
    )

    # Build the analysis prompt
    feature_list = ", ".join(str(f) for f in feature_indices)
    analysis_prompt = f"""Analyze these SAE features using the hypothesis testing workflow: {feature_list}

For each feature:
1. Call `analyze_feature` to get overview data
2. Form a hypothesis about what the feature detects
3. Generate 3-5 positive and 3-5 negative test examples
4. Test each example with `test_text` (set target_feature to the feature you're analyzing)
5. Assess confidence based on test results

Write each feature's analysis to: reports/feature_{{idx}}.md

After analyzing all features, create a summary CSV file at summary.csv with columns:
- feature_idx
- label (1-5 word description)
- confidence (HIGH/MEDIUM/LOW)
- top_3_tokens
- total_activations
- notes

Start with feature {feature_indices[0]}."""

    print(f"\n{'='*60}")
    print(f"SAE Feature Analysis Agent")
    print(f"{'='*60}")
    print(f"Features to analyze: {feature_list}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # Run the agent
    async with ClaudeSDKClient(options=options) as client:
        await client.query(analysis_prompt)

        async for message in client.receive_response():
            # Stream output to console
            if hasattr(message, 'content'):
                for block in message.content:
                    if hasattr(block, 'text'):
                        print(block.text, end='', flush=True)

    print(f"\n\n{'='*60}")
    print(f"Analysis complete!")
    print(f"Reports written to: {reports_dir}")
    print(f"{'='*60}")


async def analyze_features_standalone(feature_indices: list[int], output_dir: str):
    """
    Analyze features without Claude Agent SDK (direct Modal calls).

    This is a fallback/testing mode that doesn't use Claude for interpretation,
    just gathers the data.
    """
    from agent.tools import analyze_feature_impl, test_text_impl
    import json
    import csv

    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    results = []

    for feature_idx in feature_indices:
        print(f"\nAnalyzing feature {feature_idx}...")

        # Get feature data
        data = await analyze_feature_impl({
            "feature_idx": feature_idx,
            "n_examples": 10,
            "top_k": 15
        })
        feature_data = json.loads(data["content"][0]["text"])

        # Write raw data to file
        with open(os.path.join(reports_dir, f"feature_{feature_idx}_data.json"), 'w') as f:
            json.dump(feature_data, f, indent=2)

        # Add to results
        stats = feature_data["statistics"]
        top_tokens = feature_data["top_tokens"]
        results.append({
            "feature_idx": feature_idx,
            "total_activations": stats["total_activations"],
            "activation_rate": f"{stats['activation_rate']:.4%}",
            "top_3_tokens": ", ".join([f"'{t['token']}'" for t in top_tokens[:3]]),
            "label": "(needs interpretation)",
            "confidence": "",
            "notes": ""
        })

        print(f"  Total activations: {stats['total_activations']:,}")
        print(f"  Top tokens: {', '.join([t['token'] for t in top_tokens[:3]])}")

    # Write summary CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nData extraction complete!")
    print(f"Raw data written to: {reports_dir}")
    print(f"Summary CSV written to: {csv_path}")
    print(f"\nNote: Run with Claude Agent SDK for full interpretation.")


def main():
    parser = argparse.ArgumentParser(
        description="Run Claude agent to analyze SAE features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze specific features
    python src/agent/run_agent.py --features 16751,11328,14292 --output analysis/

    # Analyze with data extraction only (no Claude)
    python src/agent/run_agent.py --features 16751 --output analysis/ --no-agent
        """
    )

    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Comma-separated list of feature indices to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./sae_analysis",
        help="Output directory for reports and CSV"
    )
    parser.add_argument(
        "--no-agent",
        action="store_true",
        help="Skip Claude agent, just extract data"
    )

    args = parser.parse_args()

    # Parse feature indices
    feature_indices = [int(f.strip()) for f in args.features.split(",")]

    # Run analysis
    if args.no_agent:
        asyncio.run(analyze_features_standalone(feature_indices, args.output))
    else:
        asyncio.run(analyze_features_with_sdk(feature_indices, args.output))


if __name__ == "__main__":
    main()
