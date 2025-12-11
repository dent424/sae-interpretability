# Feature Catalog

Catalog of explored SAE features from the Mexican restaurant review dataset.

**SAE Configuration:** 24,576 features (768 x 32 expansion), Top-K=32

---

## Explored Features

| Feature | Description | Date |
|---------|-------------|------|
| 208 | Summary and update markers ("UPDATE:") | 8/29/2025 |
| 1087 | Dashes used for emphasis or appositives | - |
| 1893 | "They have [determiner]" patterns | - |
| 3223 | Profanity | - |
| 7797 | Forward-looking language | - |
| 10998 | Directive language with "Your" | 8/18/2025 |
| 11328 | Noun lists where items separated by line breaks (stronger with longer lists) | 8/18/2025 |
| 11579 | Last name detector | - |
| 12827 | Mexican + regional comparison | - |
| 13627 | Simple evaluative phrases about restaurants | - |
| 14292 | Demonstrative-initiated descriptive phrase detector | - |
| 16751 | Emphatic expressions ("why in the world", "never in my life", "how in the world") | - |
| 18019 | Words like "night", "evening" | - |
| 20379 | Token detector for "Gum" as in "Gumbo" | - |
| 20687 | Locations (theater, restaurant, market) | - |
| 21302 | Explosions / explosive language | - |
| 22080 | This/that/it after all forms of "Give" | 8/19/2025 |
| 23016 | (Analysis pending) | - |

---

## Feature Categories

### Linguistic Patterns
- **208**: Discourse markers (updates, edits)
- **1087**: Punctuation patterns (dashes)
- **1893**: Syntactic patterns ("They have...")
- **14292**: Demonstrative phrases
- **22080**: Give + pronoun constructions

### Sentiment/Emphasis
- **16751**: Emphatic expressions
- **3223**: Profanity
- **21302**: Explosive/emphatic language

### Temporal
- **18019**: Evening/night references
- **7797**: Future-oriented language

### Entity/Topic Detection
- **11579**: Names (last names)
- **20379**: Specific words ("Gumbo")
- **20687**: Location types
- **12827**: Mexican food + regional context

### Review Structure
- **11328**: List formatting (line breaks)
- **10998**: Direct address ("Your...")
- **13627**: Restaurant evaluations

---

## Adding New Features

When exploring a new feature, add an entry with:
1. Feature index
2. Brief description (what activates it)
3. Date explored (optional)

Use `analyze_feature_tokens_with_text()` or `extract_feature_data_adapted()` from the notebook to explore features.

---

## Notes

- Features were identified through manual exploration of top activations
- Descriptions are hypotheses based on observed activation patterns
- Some features may have multiple interpretations or edge cases
