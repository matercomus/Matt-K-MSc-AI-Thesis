# Cursor Configuration for LaTeX Thesis

This directory contains Cursor IDE configuration specifically tailored for writing a Master's thesis in LaTeX with integrated research tools.

## Structure

```
.cursor/
├── mcp.json                           # MCP server configuration (Zotero, ArXiv)
├── rules/                             # Context rules (always applied)
│   ├── git-best-practices.mdc        # Git workflow for thesis writing
│   ├── thesis-latex-workflow.mdc     # LaTeX and reference management
│   └── mcp-research-tools.mdc        # Zotero and ArXiv MCP usage
├── commands/                          # Custom Cursor commands
│   ├── zotero-semantic-search.md     # Search Zotero by concept
│   ├── zotero-update-db.md           # Update Zotero search database
│   ├── find-citation.md              # Find and export citations
│   └── arxiv-search.md               # Search and download ArXiv papers
└── README.md                          # This file
```

## Configuration Files

### `mcp.json`
Configures Model Context Protocol (MCP) servers for AI-assisted research:
- **Zotero MCP**: Local Zotero library access with semantic search
- **ArXiv MCP**: ArXiv paper search and download (storage: `~/Documents/arxiv-mcp-papers/`)

### Rules (`rules/*.mdc`)

Rules are automatically applied to all AI interactions in this workspace.

#### `git-best-practices.mdc`
Defines git workflow adapted for academic thesis writing:
- Commit types and gitmojis for thesis content (content, cite, figure, edit, etc.)
- Branch management for chapters and revisions
- Pre-commit checks for LaTeX projects
- Milestone tagging for thesis versions

#### `thesis-latex-workflow.mdc`
Core workflow rules for LaTeX thesis development:
- **Reference management**: Never edit `references_new.bib` manually (Zotero is source of truth)
- **Citation style**: Inline citations, no author mentions, no summaries
- **Build configuration**: Artifacts in `build/` directory
- **LaTeX best practices**: Figures, tables, cross-references
- **Quality checks**: Pre-commit verification steps

#### `mcp-research-tools.mdc`
Comprehensive guide to using research MCP tools:
- **Zotero MCP**: Semantic search, database management, citation export, full-text access
- **ArXiv MCP**: Paper search, download, analysis
- **Complete workflows**: Finding papers → Adding to Zotero → Citing in thesis
- **Troubleshooting**: Common issues and solutions

## Commands (`commands/*.md`)

Commands provide specific task instructions for the AI assistant. Invoke with `@command-name` in chat.

### `@zotero-semantic-search`
Search Zotero library by conceptual meaning using AI-powered semantic search.

**Use when**: Exploring related work, finding papers on concepts, literature review

**Example**: "@zotero-semantic-search privacy-preserving trajectory generation techniques"

### `@zotero-update-db`
Update Zotero MCP semantic search database after adding papers to library.

**Use when**: After batch-adding papers, before research sessions, weekly maintenance

**Options**: Metadata-only (fast) or full-text indexing (comprehensive)

### `@find-citation`
Search for papers in Zotero and get citation keys for use in thesis.

**Use when**: Need to cite specific papers, building reference list, verifying citation keys

**Output**: Paper info, citation key, usage example (`\cite{key}`)

### `@arxiv-search`
Search ArXiv for papers and download for review before adding to Zotero.

**Use when**: Finding recent work, exploring new topics, tracking specific research

**Workflow**: Search → Download → Review → Add to Zotero → Cite

## Quick Start

### First-Time Setup

1. **Verify MCP servers are configured**:
   - Zotero: Ensure Zotero desktop is running with local API enabled
   - ArXiv: Create storage directory: `mkdir -p ~/Documents/arxiv-mcp-papers/`

2. **Initialize Zotero semantic search**:
   ```bash
   zotero-mcp update-db --fulltext
   ```

3. **Test configuration**:
   - Use `@zotero-semantic-search` to search your library
   - Use `@arxiv-search` to find papers on ArXiv

### Daily Workflow

1. **Before research session**:
   ```bash
   zotero-mcp update-db  # Quick metadata update
   ```

2. **Find papers**:
   - Use `@zotero-semantic-search` for papers in your library
   - Use `@arxiv-search` for new papers
   - Use `@find-citation` to get citation keys

3. **Add citations to thesis**:
   - Use citation keys with `\cite{key}` in LaTeX
   - Never manually edit `references_new.bib`

4. **Commit changes**:
   ```bash
   git add main.tex
   git commit -m "✍️ content: add related work section with citations"
   ```

### Weekly Maintenance

1. **Update Zotero database with full-text**:
   ```bash
   zotero-mcp update-db --fulltext
   ```

2. **Export bibliography from Zotero** to `references_new.bib`

3. **Commit bibliography updates**:
   ```bash
   git add references_new.bib
   git commit -m "🔗 cite: sync bibliography with Zotero library"
   ```

## Research Workflow

### Complete Paper Integration Flow

1. **Discovery**:
   - Search existing library: `@zotero-semantic-search [topic]`
   - Search for new papers: `@arxiv-search [topic]`

2. **Evaluation**:
   - Download promising papers
   - Review using deep analysis
   - Take notes in `notes/` directory

3. **Integration**:
   - Add relevant papers to Zotero
   - Add tags and organize into collections
   - Update Zotero MCP database: `zotero-mcp update-db --fulltext`
   - Export updated bibliography to thesis

4. **Citation**:
   - Find citation keys: `@find-citation [paper topic]`
   - Add citations to thesis: `\cite{key}`
   - Verify citations compile correctly

## Key Principles

### Reference Management
- **Zotero is the single source of truth** for all references
- `references_new.bib` is automatically synced from Zotero
- Never manually edit the bibliography file
- All citation changes happen in Zotero first

### Citation Style
- Use inline citations throughout thesis
- No author name mentions in text
- No paper summaries or descriptions
- No special formatting for citations
- Let context provide meaning, citations provide support

### Git Workflow
- Thesis-specific commit types and gitmojis
- Atomic commits (one logical change per commit)
- Chapter-based branching for major work
- Tag milestones for supervisor reviews

### MCP Tools
- Zotero MCP for library management and semantic search
- ArXiv MCP for discovering new papers
- Complete workflows from discovery to citation
- Database maintenance for optimal search results

## Troubleshooting

### Zotero MCP Issues

**Semantic search not working:**
1. Ensure Zotero is running
2. Check Zotero preferences: Allow local API access
3. Update database: `zotero-mcp update-db --fulltext`
4. Check status: `zotero-mcp db-status`

**Citations not found:**
1. Verify paper is in Zotero library
2. Export updated bibliography from Zotero
3. Check citation key spelling in LaTeX

### ArXiv MCP Issues

**Papers not downloading:**
1. Check internet connection
2. Verify arXiv ID format (no "arXiv:" prefix)
3. Create storage directory: `mkdir -p ~/Documents/arxiv-mcp-papers/`

### LaTeX Issues

**Undefined citations:**
1. Verify citation key exists in `references_new.bib`
2. Run `latexmk` multiple times to resolve references
3. Check for typos in citation keys

**Build failures:**
1. Read error messages for line numbers
2. Check recent changes
3. Verify all required packages are installed

## Additional Resources

- **Zotero MCP Documentation**: See `rules/mcp-research-tools.mdc`
- **Git Best Practices**: See `rules/git-best-practices.mdc`
- **LaTeX Workflow**: See `rules/thesis-latex-workflow.mdc`
- **Project README**: `/home/matt/Dev/Matt-K-MSc-AI-Thesis/README.md`

## Updates and Maintenance

This configuration is maintained as part of the thesis project. Updates should:
- Preserve MCP server configurations
- Maintain rule consistency across files
- Update commands based on workflow improvements
- Document any changes in git commits

Last Updated: October 2025

