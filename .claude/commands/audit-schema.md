# DEPRECATED: Use /final_audit instead

This command has been merged into `/final_audit`.

**Use `/final_audit` for:**
- Schema validation (required fields with correct types)
- Content validation (substantive hypotheses, tests, verdicts)
- Process validation (audit.jsonl step logging)
- `--fix` mode to auto-repair missing top-level fields

**Examples:**
```
/final_audit                    # Audit all features
/final_audit 8134               # Audit single feature
/final_audit --fix              # Audit and fix issues
/final_audit --limit 10         # Audit first 10 only
```

**Archived version:** `.claude/commands/archive/audit-schema-v1.md`

---

**Stop.** Inform the user that `/audit-schema` is deprecated and they should use `/final_audit` instead.
