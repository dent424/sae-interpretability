# Claude Agent SDK Reference

Reference for building AI agents with the Claude Agent SDK.

## Installation

```bash
pip install claude-agent-sdk
```

## Core Interfaces

### Option 1: `query()` - One-off Tasks

For independent analysis tasks (no conversation memory):

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Analyze this SAE feature",
    options=ClaudeAgentOptions(...)
):
    print(message)
```

### Option 2: `ClaudeSDKClient` - Conversations

For multi-turn interactions with context:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async with ClaudeSDKClient(options=options) as client:
    await client.query("Analyze SAE feature 16751")
    async for message in client.receive_response():
        print(message)

    # Follow-up - Claude remembers context
    await client.query("Compare it to feature 11328")
    async for message in client.receive_response():
        print(message)
```

---

## Defining Tools

Use the `@tool` decorator to create custom tools:

```python
from claude_agent_sdk import tool
from typing import Any

@tool(
    "analyze_feature",                          # Tool name
    "Analyze SAE feature properties",           # Description
    {"feature_id": str, "top_k": int}           # Input schema
)
async def analyze_feature(args: dict[str, Any]) -> dict[str, Any]:
    feature_id = args["feature_id"]
    top_k = args.get("top_k", 10)

    # Your analysis logic here
    results = do_analysis(feature_id, top_k)

    return {
        "content": [{
            "type": "text",
            "text": f"Analysis results: {results}"
        }]
    }
```

### Tool Return Format

Tools must return a dict with `content` list:

```python
return {
    "content": [
        {"type": "text", "text": "Plain text result"},
        # Or for structured data:
        {"type": "text", "text": json.dumps(data)}
    ]
}
```

---

## MCP Server Creation

Bundle tools into an MCP server:

```python
from claude_agent_sdk import create_sdk_mcp_server

sae_server = create_sdk_mcp_server(
    name="sae_analyzer",
    version="1.0.0",
    tools=[analyze_feature, compare_features, get_examples]
)
```

---

## Agent Configuration

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    # System prompt
    system_prompt="You are an expert in SAE interpretability...",

    # Or use preset with additions
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "Focus on mechanistic interpretability..."
    },

    # Register MCP servers
    mcp_servers={"sae": sae_server},

    # Allowed tools (MCP tools use mcp__<server>__<tool> format)
    allowed_tools=[
        "mcp__sae__analyze_feature",
        "mcp__sae__compare_features",
        "Read",
        "Write",
        "Bash"
    ],

    # Working directory
    cwd="/path/to/project",

    # Permission mode
    permission_mode="acceptEdits",  # Auto-approve file operations

    # Model selection
    model="claude-sonnet-4-20250514"
)
```

### Permission Modes

| Mode | Behavior |
|------|----------|
| `"default"` | Ask user for approval |
| `"acceptEdits"` | Auto-approve file operations |
| `"bypassPermissions"` | Skip all permission checks |

---

## Message Types

```python
from claude_agent_sdk import (
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock
)

async for message in client.receive_response():
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(f"Text: {block.text}")
            elif isinstance(block, ToolUseBlock):
                print(f"Tool call: {block.name}({block.input})")
```

---

## Complete SAE Agent Example

```python
import asyncio
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    AssistantMessage,
    TextBlock
)
from typing import Any
import json

# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@tool(
    "analyze_feature",
    "Get activation patterns and top tokens for an SAE feature",
    {"feature_idx": int, "n_examples": int}
)
async def analyze_feature(args: dict[str, Any]) -> dict[str, Any]:
    """Analyze a single SAE feature"""
    feature_idx = args["feature_idx"]
    n_examples = args.get("n_examples", 10)

    # Call your Modal function or local analysis
    # results = await modal_interpreter.get_feature_examples.remote(feature_idx, n_examples)

    results = {
        "feature": feature_idx,
        "top_tokens": ["example", "tokens"],
        "activation_count": 1234
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(results, indent=2)
        }]
    }


@tool(
    "process_text",
    "Run text through GPT-2 + SAE pipeline and get activations",
    {"text": str, "feature_indices": list}
)
async def process_text(args: dict[str, Any]) -> dict[str, Any]:
    """Process arbitrary text and extract feature activations"""
    text = args["text"]
    features = args.get("feature_indices", [])

    # Call your Modal function
    # results = await modal_interpreter.process_text.remote(text, features)

    results = {
        "tokens": ["example"],
        "activations": {str(f): [0.0] for f in features}
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(results, indent=2)
        }]
    }


@tool(
    "compare_features",
    "Compare activation patterns between two features",
    {"feature1": int, "feature2": int}
)
async def compare_features(args: dict[str, Any]) -> dict[str, Any]:
    """Compare two SAE features"""
    f1, f2 = args["feature1"], args["feature2"]

    comparison = {
        "feature1": f1,
        "feature2": f2,
        "similarity": 0.85,
        "shared_tokens": ["the", "a"]
    }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(comparison, indent=2)
        }]
    }


# =============================================================================
# MCP SERVER
# =============================================================================

sae_server = create_sdk_mcp_server(
    name="sae",
    version="1.0.0",
    tools=[analyze_feature, process_text, compare_features]
)


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

options = ClaudeAgentOptions(
    system_prompt="""You are an expert in mechanistic interpretability and Sparse Autoencoders (SAEs).

Your role is to help analyze SAE features from a GPT-2 model trained on restaurant reviews.

When analyzing features:
1. Look at top activating tokens to understand the pattern
2. Examine example contexts to verify semantic meaning
3. Consider both syntactic and semantic interpretations
4. Compare with related features when relevant

The SAE has 24,576 features (768 * 32 expansion) with Top-K=32 sparsity.""",

    mcp_servers={"sae": sae_server},

    allowed_tools=[
        "mcp__sae__analyze_feature",
        "mcp__sae__process_text",
        "mcp__sae__compare_features",
        "Read",
        "Write"
    ],

    permission_mode="acceptEdits"
)


# =============================================================================
# AGENT EXECUTION
# =============================================================================

async def run_analysis():
    """Run an interactive SAE analysis session"""
    async with ClaudeSDKClient(options=options) as client:
        # Initial query
        await client.query(
            "Analyze feature 16751 and explain what semantic pattern it detects"
        )

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

        # Follow-up
        await client.query(
            "Now test it on this text: 'Why in the world would anyone eat here?'"
        )

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


if __name__ == "__main__":
    asyncio.run(run_analysis())
```

---

## Agentic Patterns

### Pattern 1: Exploration Loop

```python
async def explore_features():
    async with ClaudeSDKClient(options=options) as client:
        await client.query("What are the most interesting features in this SAE?")
        async for msg in client.receive_response():
            pass

        await client.query("Deep dive into the top 3 you identified")
        async for msg in client.receive_response():
            pass
```

### Pattern 2: Hypothesis Testing

```python
async def test_hypothesis():
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "I think feature 11328 detects formatted lists. "
            "Test this hypothesis by analyzing its activations."
        )
        async for msg in client.receive_response():
            pass
```

### Pattern 3: Report Generation

```python
async def generate_report():
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "Analyze features 16751, 11328, and 14292. "
            "Write a markdown report to feature_analysis.md"
        )
        async for msg in client.receive_response():
            pass
```

---

## Connecting to Modal Functions

Your tools can call Modal remote functions:

```python
from modal import Function

# Get reference to deployed Modal function
modal_process_text = Function.lookup("sae-interpretability", "SAEInterpreter.process_text")

@tool("process_text", "Run text through SAE", {"text": str})
async def process_text(args: dict[str, Any]) -> dict[str, Any]:
    # Call Modal function
    result = modal_process_text.remote(args["text"])

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result)
        }]
    }
```

---

## Custom Permission Logic

```python
async def can_use_tool(tool_name: str, input_data: dict, context: dict) -> dict:
    # Block certain operations
    if tool_name == "Bash" and "rm" in input_data.get("command", ""):
        return {"behavior": "deny", "message": "Delete operations not allowed"}

    # Allow everything else
    return {"behavior": "allow"}

options = ClaudeAgentOptions(
    can_use_tool=can_use_tool,
    ...
)
```

---

## Resources

- [Agent SDK Overview](https://platform.claude.com/docs/en/agent-sdk/overview)
- [Python SDK Reference](https://platform.claude.com/docs/en/agent-sdk/python)
- [Tool Use Guide](https://platform.claude.com/docs/en/build-with-claude/tool-use)
- [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook)
- [Claude Quickstarts](https://github.com/anthropics/anthropic-quickstarts)
