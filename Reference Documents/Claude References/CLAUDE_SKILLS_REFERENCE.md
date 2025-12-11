# Claude Skills Reference

Reference for Claude Skills - folder-based capability packages that enhance Claude's performance on specific tasks.

## What Are Skills?

Skills are specialized capability packages that Claude can load when needed. They consist of:
- Instructions and guidance documentation
- Executable scripts and code
- Supporting resources and files
- A SKILL.md manifest file

**Key Characteristics:**
| Property | Description |
|----------|-------------|
| Composable | Multiple skills work together automatically |
| Portable | Same format across Claude Apps, Claude Code, and API |
| Efficient | Loaded only when relevant to the task |
| Powerful | Can include executable code for specialized tasks |

---

## Skill Structure

Skills are organized as folders:

```
skill-name/
├── SKILL.md          # Manifest with instructions
├── scripts/          # Executable code (optional)
│   └── process.py
└── resources/        # Supporting files (optional)
    └── templates/
```

The `SKILL.md` file contains:
- Skill description and purpose
- Instructions for Claude on how to use the skill
- Configuration and parameters

---

## Using Skills in Claude Code

### Installation Location

```
~/.claude/skills/
```

### Installing from Marketplace

Skills can be installed from the Anthropic skills marketplace:
- Repository: `anthropics/skills`

### Auto-Invocation

Claude automatically loads relevant skills based on task context. No manual invocation needed - Claude detects when a skill is useful and loads it.

---

## Built-in Skills (Anthropic-Provided)

| Skill | Purpose |
|-------|---------|
| Excel | Create and manipulate spreadsheets with formulas |
| PowerPoint | Generate presentations |
| Word | Create formatted documents |
| PDF | Create fillable PDF forms |

These are available in Claude Apps (Pro, Max, Team, Enterprise) and can be enabled in settings.

---

## Creating Custom Skills

Use the **skill-creator** tool to create skills:

1. Describe your use case
2. Answer clarifying questions
3. Generates folder structure automatically
4. Formats SKILL.md manifest
5. Bundles necessary resources

No manual file editing required.

---

## Skills API (Beta)

For programmatic skill management via the API:

**Endpoint:** `POST /v1/skills`

**Requirements:**
- Beta header: `anthropic-beta: skills-2025-10-02`
- Code Execution Tool beta enabled

```python
import anthropic

client = anthropic.Anthropic(
    api_key="your-api-key",
    default_headers={
        "anthropic-beta": "skills-2025-10-02"
    }
)

# List available skills
skills = client.skills.list(limit=20)

# Filter by source
custom_skills = client.skills.list(source="custom")
anthropic_skills = client.skills.list(source="anthropic")
```

---

## Skills vs Tools

| Aspect | Skills | Tools |
|--------|--------|-------|
| Format | Folder with SKILL.md | Function definition |
| Invocation | Auto-loaded when relevant | Explicitly called |
| Scope | Complex, multi-step capabilities | Single operations |
| Portability | Cross-platform (Apps, Code, API) | Per-integration |
| Content | Instructions + scripts + resources | Code only |

---

## Resources

- [Claude Skills Blog Post](https://claude.com/blog/skills)
- [Agent Skills Overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview)
- [Skills Marketplace](https://github.com/anthropics/skills)
