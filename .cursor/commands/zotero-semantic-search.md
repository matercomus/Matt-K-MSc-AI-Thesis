# Zotero Semantic Search

Search the Zotero library using AI-powered semantic search to find papers by conceptual meaning rather than exact keywords.

## Instructions

1. **Use the `zotero_semantic_search` MCP tool** to search the local Zotero library
2. Formulate queries as complete questions or topic descriptions
3. Return results with similarity scores and relevant metadata
4. Provide citation keys for papers that match the search

## Query Guidelines

### Good Query Examples
- "Find research on privacy-preserving trajectory generation"
- "Papers about anomaly detection in time series data using deep learning"
- "Studies on differential privacy mechanisms in machine learning"
- "Research similar to: [paste abstract or description]"
- "Work related to synthetic data generation for transportation systems"

### Query Construction
- Use complete, descriptive questions
- Include domain context (e.g., "in transportation", "for privacy")
- Specify key concepts clearly
- Can paste abstracts or paper descriptions for similarity search

## Expected Output

For each matching paper provide:
- **Title** and **Authors**
- **Similarity score** (indicates relevance)
- **Citation key** (for use in LaTeX with `\cite{key}`)
- **Abstract** or brief description
- **Year** and publication venue if available
- **Tags** and collections (if relevant)

## Usage Context

Use this command when:
- Exploring related work for literature review
- Finding papers on a specific research concept
- Discovering relevant citations for a section
- Investigating papers similar to a known work

## Follow-up Actions

After finding relevant papers:
1. Read full text if needed using `zotero_get_item_fulltext`
2. Export citations to verify citation keys
3. Add citation to thesis using `\cite{key}`
4. Update `references_new.bib` from Zotero if new papers were added

## Important Notes

- Zotero desktop must be running for this to work
- Results depend on database being up-to-date (see `zotero-update-db` command)
- More comprehensive results if full-text indexing is enabled
- Semantic search finds papers by meaning, not just keyword matching

