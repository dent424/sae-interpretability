"""Claude SDK tool definitions for SAE interpretability.

These tools allow Claude to interact with the Modal backend for
feature analysis and hypothesis testing.

The tools are bundled into an MCP server for use with Claude Agent SDK.
"""

import json
from typing import Any
from modal import Function

# Lazy load - tools get the interpreter when called
_interpreter = None


def get_interpreter():
    """Get reference to deployed Modal SAEInterpreter."""
    global _interpreter
    if _interpreter is None:
        _interpreter = Function.lookup("sae-interpretability", "SAEInterpreter")
    return _interpreter


# Tool definitions using claude_agent_sdk format
# These will be imported and registered by run_agent.py

async def analyze_feature_impl(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get comprehensive analysis of an SAE feature.

    Returns top tokens, example contexts, and statistics.
    This is the primary tool for understanding what a feature detects.
    """
    feature_idx = args["feature_idx"]
    n_examples = args.get("n_examples", 10)
    top_k = args.get("top_k", 15)
    max_samples = args.get("max_samples", 30000)

    interpreter = get_interpreter()

    # Call Modal methods
    stats = interpreter.get_feature_stats.remote(feature_idx)
    top_tokens = interpreter.get_top_tokens.remote(feature_idx, top_k=top_k, max_samples=max_samples)
    contexts = interpreter.get_feature_contexts.remote(feature_idx, n_samples=n_examples)

    result = {
        "feature_idx": feature_idx,
        "statistics": stats,
        "top_tokens": top_tokens,
        "example_contexts": contexts
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


async def test_text_impl(args: dict[str, Any]) -> dict[str, Any]:
    """
    Run text through GPT-2 → SAE pipeline to see which features activate.

    Use this to test hypotheses about what a feature detects.
    Provide positive examples (should activate) and negative examples
    (should not activate) to verify your interpretation.
    """
    text = args["text"]
    target_feature = args.get("target_feature")  # Optional: highlight specific feature

    interpreter = get_interpreter()
    result = interpreter.process_text.remote(text)

    # If target feature specified, add focused analysis
    if target_feature is not None:
        target_str = str(target_feature)
        if target_str in result.get("feature_activations", {}):
            activations = result["feature_activations"][target_str]
            max_activation = max(activations) if activations else 0
            active_tokens = [
                {"token": t, "activation": a}
                for t, a in zip(result["tokens"], activations)
                if a > 0
            ]
            result["target_feature_analysis"] = {
                "feature_idx": target_feature,
                "max_activation": max_activation,
                "activates": max_activation > 0,
                "active_tokens": active_tokens
            }
        else:
            result["target_feature_analysis"] = {
                "feature_idx": target_feature,
                "max_activation": 0,
                "activates": False,
                "active_tokens": []
            }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


# Tool definitions in the format expected by claude_agent_sdk
TOOL_DEFINITIONS = [
    {
        "name": "analyze_feature",
        "description": """Get comprehensive analysis of an SAE feature including top activating tokens, example contexts, and statistics.

Use this tool as your first step to understand what a feature detects. The output includes:
- statistics: activation count, rate, mean, max
- top_tokens: most common tokens that activate this feature
- example_contexts: real text examples where the feature fires, with the active token marked

Based on this analysis, form a hypothesis about what pattern the feature detects.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_idx": {
                    "type": "integer",
                    "description": "Feature index (0-24575)"
                },
                "n_examples": {
                    "type": "integer",
                    "description": "Number of example contexts to return (default: 10)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top tokens to return (default: 15)"
                }
            },
            "required": ["feature_idx"]
        },
        "handler": analyze_feature_impl
    },
    {
        "name": "test_text",
        "description": """Run text through the GPT-2 → SAE pipeline to see which features activate.

Use this to test your hypothesis about what a feature detects:
1. Create POSITIVE examples that SHOULD activate the feature
2. Create NEGATIVE examples that should NOT activate
3. Check if the feature behaves as expected

If testing a specific feature, set target_feature to get focused analysis.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze"
                },
                "target_feature": {
                    "type": "integer",
                    "description": "Optional: specific feature index to focus on"
                }
            },
            "required": ["text"]
        },
        "handler": test_text_impl
    }
]


def create_mcp_server():
    """Create MCP server with SAE tools for Claude Agent SDK."""
    try:
        from claude_agent_sdk import tool, create_sdk_mcp_server
    except ImportError:
        print("Warning: claude_agent_sdk not installed. Tools defined but MCP server not created.")
        return None

    # Create decorated tool functions
    @tool(
        "analyze_feature",
        TOOL_DEFINITIONS[0]["description"],
        {"feature_idx": int, "n_examples": int, "top_k": int}
    )
    async def analyze_feature(args: dict[str, Any]) -> dict[str, Any]:
        return await analyze_feature_impl(args)

    @tool(
        "test_text",
        TOOL_DEFINITIONS[1]["description"],
        {"text": str, "target_feature": int}
    )
    async def test_text(args: dict[str, Any]) -> dict[str, Any]:
        return await test_text_impl(args)

    return create_sdk_mcp_server(
        name="sae",
        version="1.0.0",
        tools=[analyze_feature, test_text]
    )


# For direct testing
if __name__ == "__main__":
    import asyncio

    async def test_tools():
        print("Testing analyze_feature...")
        result = await analyze_feature_impl({"feature_idx": 16751, "n_examples": 3})
        print(result["content"][0]["text"][:500])

        print("\nTesting test_text...")
        result = await test_text_impl({
            "text": "Why in the world would anyone eat here?",
            "target_feature": 16751
        })
        print(result["content"][0]["text"][:500])

    asyncio.run(test_tools())
