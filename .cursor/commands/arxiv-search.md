# Search and Download ArXiv Papers

Search ArXiv repository for papers and download them for review before adding to Zotero library.

## Instructions

1. **Search ArXiv** using `search_papers` MCP tool
2. **Filter results** by date range and categories if needed
3. **Download promising papers** using `download_paper` with arXiv ID
4. **Review papers** using `read_paper` or `deep-paper-analysis` prompt
5. **Suggest adding to Zotero** if paper is relevant to thesis

## Search Parameters

### Query
Describe the research topic clearly:
- "trajectory generation using diffusion models"
- "anomaly detection in time series with deep learning"
- "differential privacy for location data"

### Categories (Optional)
Filter by arXiv categories relevant to AI thesis:
- `cs.AI` - Artificial Intelligence
- `cs.LG` - Machine Learning
- `cs.CV` - Computer Vision
- `cs.CR` - Cryptography and Security
- `stat.ML` - Machine Learning (Statistics)
- `cs.DB` - Databases (for trajectory data)

### Date Range (Optional)
Filter by publication date:
- `date_from="2023-01-01"` - Papers from 2023 onwards
- Useful for finding recent work

### Max Results
Default to 10, adjust based on need:
- `max_results=10` - Good for initial exploration
- `max_results=20` - For comprehensive searches
- `max_results=5` - For very specific queries

## Example Searches

### General Topic Search
```
search_papers(
    query="privacy-preserving synthetic trajectory generation",
    max_results=10
)
```

### Recent Papers in Specific Area
```
search_papers(
    query="diffusion models for trajectory data",
    date_from="2024-01-01",
    categories=["cs.LG", "cs.AI"],
    max_results=15
)
```

### Focused Search
```
search_papers(
    query="LM-TAD anomaly detection",
    max_results=5
)
```

## Downloading Papers

For each relevant paper found:
1. **Download**: Use `download_paper(paper_id="2401.12345")`
2. **Papers saved to**: `~/Documents/arxiv-mcp-papers/`
3. **Read content**: Use `read_paper(paper_id="2401.12345")`

## Paper Review Options

### Quick Review
Use `read_paper` to access paper content and summarize key points

### Deep Analysis
Use `deep-paper-analysis` prompt for comprehensive analysis:
- Executive summary
- Research context and motivation
- Methodology analysis
- Results evaluation
- Implications for thesis research
- Future research directions

## Integration with Zotero

After reviewing ArXiv papers:

1. **If paper is relevant**:
   - Suggest user add to Zotero library
   - Provide arXiv ID or DOI for easy import
   - Recommend appropriate tags/collections

2. **After user adds to Zotero**:
   - User exports updated bibliography to `references_new.bib`
   - User updates Zotero MCP database: `zotero-mcp update-db --fulltext`
   - Paper becomes searchable via Zotero semantic search
   - Citation key available for use in thesis

3. **Workflow reminder**:
   ```
   ArXiv search → Download → Review → Add to Zotero → Export bib → Update db → Cite in thesis
   ```

## Expected Output

For search results, provide:
- **Paper title** and **authors**
- **ArXiv ID** (e.g., 2401.12345)
- **Publication date**
- **Abstract** or brief description
- **Relevance** to user's query
- **Recommendation** on whether to download

For downloaded papers, provide:
- **Summary** of key contributions
- **Relevance** to thesis research
- **Recommendation** on adding to Zotero
- **Suggested tags** or collections for organization

## Managing ArXiv Papers

### Local Storage
- Papers stored at: `~/Documents/arxiv-mcp-papers/`
- Organized by arXiv ID
- Can list all downloaded papers: `list_papers()`

### Storage Management
- Periodically clean up papers after adding important ones to Zotero
- Keep papers under review separate from Zotero library
- ArXiv storage is for temporary review, Zotero is permanent library

## Common Use Cases

### Finding Recent Work
When starting a new thesis section or updating literature review:
1. Search ArXiv for recent papers in topic area
2. Download and review promising papers
3. Add most relevant papers to Zotero
4. Update bibliography and cite in thesis

### Exploring Unfamiliar Topics
When expanding into new research area:
1. Broad ArXiv search on topic
2. Download several papers for overview
3. Use deep analysis to understand landscape
4. Add foundational papers to Zotero

### Tracking Specific Research
When following up on specific approach or method:
1. Targeted search with specific keywords
2. Filter by recent dates
3. Download papers on specific technique
4. Compare with existing work in Zotero library

## Important Notes

- **ArXiv IDs**: Format is YYMM.NNNNN (e.g., 2401.12345)
- **Storage location**: `~/Documents/arxiv-mcp-papers/` (configured in `.cursor/mcp.json`)
- **Internet required**: ArXiv search and download need internet connection
- **Not a replacement for Zotero**: Use ArXiv MCP for discovery, Zotero for permanent library
- **Citation workflow**: Papers must be in Zotero to cite in thesis

## Troubleshooting

**Search returns no results:**
- Broaden search query
- Remove category filters
- Check internet connection

**Download fails:**
- Verify arXiv ID format (no "arXiv:" prefix)
- Check internet connection
- Verify storage path exists

**Can't read downloaded paper:**
- Verify paper was downloaded successfully
- Check paper ID matches downloaded paper
- Use `list_papers()` to see available papers

