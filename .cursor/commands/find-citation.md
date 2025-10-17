# Find Citation in Zotero

Search for papers in Zotero library and retrieve citation information for use in the thesis.

## Instructions

1. **Search for the paper** using appropriate Zotero MCP tool:
   - Use `zotero_semantic_search` for conceptual searches
   - Use `zotero_search_items` for keyword/title searches
   - Use `zotero_search_by_tag` for tag-based searches

2. **Get citation metadata** using `zotero_get_item_metadata` with `format="bibtex"`

3. **Verify citation key** matches entry in `references_new.bib`

4. **Provide citation key** for use in LaTeX with `\cite{key}`

## Search Strategy

### For Known Papers
Use standard search with title or author:
```
zotero_search_items(query="trajectory generation", ...)
```

### For Topic-Based Search
Use semantic search with concept description:
```
zotero_semantic_search(query="privacy-preserving machine learning techniques", ...)
```

### For Organized Topics
Use tag-based search if library is organized:
```
zotero_search_by_tag(include_tags=["#Privacy", "#Trajectory"], ...)
```

## Expected Output

Provide the following for each relevant paper:

1. **Paper Information**:
   - Title
   - Authors
   - Year
   - Publication venue
   - Abstract (if helpful)

2. **Citation Key**: The BibTeX citation key (e.g., `smith2023privacy`)

3. **Usage Example**: How to cite in LaTeX: `\cite{smith2023privacy}`

4. **Verification**: Confirm the citation key exists in `references_new.bib`

## Example Workflow

**User Request**: "Find citation for differential privacy in trajectory data"

**AI Response**:
1. Search Zotero: `zotero_semantic_search("differential privacy mechanisms for trajectory data")`
2. Find relevant papers with high similarity scores
3. Get metadata for top matches with `format="bibtex"`
4. Extract citation keys
5. Verify keys exist in `references_new.bib`
6. Provide citation keys and usage instructions

## Important Notes

### Reference Management Rules
- **NEVER manually edit `references_new.bib`** - it's managed by Zotero
- All citations must exist in Zotero library first
- Citation keys are assigned by Zotero (often via Better BibTeX plugin)
- If paper not in library, user must add it to Zotero first

### Citation Style Guidelines
- Use inline citations: `\cite{key}`
- No author names in text
- No paper summaries in text
- No special formatting for citations
- Let context provide meaning, citations provide support

## If Paper Not Found

If the requested paper is not in Zotero:

1. **Search ArXiv** using `arxiv-search` command if it's a recent paper
2. **Suggest to user**: "This paper is not in your Zotero library. Would you like me to search ArXiv or shall we add it to Zotero first?"
3. **Workflow to add**:
   - Find paper on ArXiv or other source
   - User adds to Zotero (via DOI, arXiv ID, or manual entry)
   - User exports updated bibliography to `references_new.bib`
   - User updates Zotero MCP database: `zotero-mcp update-db`
   - Then retry citation search

## Multiple Citations

When providing multiple citations for a topic:
- List all relevant papers with citation keys
- Order by relevance (similarity score or importance)
- Explain which papers address which aspects
- Show how to cite multiple papers: `\cite{key1,key2,key3}`

## Follow-up Actions

After finding citations:
1. Verify citation keys in `references_new.bib`
2. Add citations to appropriate thesis sections
3. Rebuild document to verify citations work
4. Check for "undefined citation" warnings in compilation

