# Zotero MCP Setup and Configuration

Initial setup and configuration verification for Zotero MCP semantic search integration.

## Prerequisites

Before setting up Zotero MCP, ensure:

1. **Zotero Desktop is installed** (version 7+ recommended)
2. **Zotero is running** with papers in your library
3. **Local API is enabled** in Zotero:
   - Open Zotero → Preferences → Advanced → General
   - Check: "Allow other applications on this computer to communicate with Zotero"

## Current Configuration

The system is already configured to use:
- **Embedding Model**: Local embeddings (default, free)
- **API Keys**: None required for basic functionality
- **Access Method**: Local Zotero API (`ZOTERO_LOCAL=true`)
- **MCP Config**: `.cursor/mcp.json` (gitignored)

### Configuration File

`.cursor/mcp.json` should contain:
```json
{
  "mcpServers": {
    "zotero": {
      "command": "zotero-mcp",
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```

**Important**: Keep this simple. Do not add `ZOTERO_EMBEDDING_MODEL` or `OPENAI_API_KEY` unless you specifically need OpenAI embeddings.

## Library Access Setup

### My Library (Default)

By default, the local API indexes "My Library":
- All papers in your personal Zotero library
- Papers you've added or imported
- Automatically synced with semantic search

### Group Libraries

To include group library papers:

1. **Open Zotero desktop**
2. **Locate your group library** in the left sidebar (e.g., "Matt-AI-Master-Thesis")
3. **Drag collections** from the group library to "My Library"
   - This creates copies in your personal library
   - Semantic search will index these copies
4. **Update the database** (see below)

**Note**: This creates duplicates but ensures all papers are searchable. For explicit group library access without duplication, use web API setup (requires API key).

## Initial Database Build

After configuration, build the semantic search database:

### Quick Setup (Metadata Only)
```bash
zotero-mcp update-db
```
- Fast (~30 seconds for 293 items)
- Indexes titles, abstracts, metadata
- Good for testing setup

### Comprehensive Setup (Full-Text)
```bash
zotero-mcp update-db --fulltext
```
- Slower (~2-5 minutes for 293 items)
- Indexes complete paper content from PDFs
- Better semantic search results
- **Recommended for production use**

## Verification Steps

### 1. Check Database Status

```bash
zotero-mcp db-status
```

Expected output:
- Collection name: `zotero_library`
- Document count: Number matching your library size (e.g., 293)
- Embedding model: `default`
- Database path: `~/.config/zotero-mcp/chroma_db`

### 2. Test Semantic Search

Use the `@zotero-semantic-search` command or MCP tool:

```
Query: "privacy-preserving trajectory generation using differential privacy"
```

Expected results:
- List of relevant papers with similarity scores
- Higher scores (>0.3) indicate strong relevance
- Papers should be from your Zotero library

### 3. Test Standard Search

```
Query: "DiffTraj: Generating GPS Trajectory"
```

Should find papers with matching title/keywords.

### 4. Verify Citation Keys

Search for a known paper and verify the citation key format matches entries in `references_new.bib`.

## Common Setup Issues

### Zotero Not Running
**Symptom**: Connection errors, "Cannot connect to Zotero"
**Solution**: Start Zotero desktop application

### Local API Not Enabled
**Symptom**: Permission errors, API access denied
**Solution**: Enable in Zotero Preferences → Advanced → General

### Empty Database
**Symptom**: 0 documents indexed
**Solution**: 
1. Ensure papers are in "My Library" (not just group libraries)
2. Run `zotero-mcp update-db`
3. Check Zotero has papers loaded

### Embedding Dimension Mismatch
**Symptom**: "Collection expecting embedding with dimension of 1536, got 384"
**Solution**:
1. Check `.cursor/mcp.json` only has `ZOTERO_LOCAL=true`
2. Remove any `ZOTERO_EMBEDDING_MODEL` or `OPENAI_API_KEY` variables
3. Restart Cursor completely
4. Rebuild database: `rm -rf ~/.config/zotero-mcp/chroma_db && zotero-mcp update-db`

### Group Library Papers Missing
**Symptom**: Papers from group libraries don't appear in search
**Solution**:
1. Copy collections from group library to "My Library" in Zotero
2. Run `zotero-mcp update-db`

## Database Maintenance

### Regular Updates

**Daily** (before research sessions):
```bash
zotero-mcp update-db
```

**Weekly** (comprehensive):
```bash
zotero-mcp update-db --fulltext
```

### Force Rebuild

If database becomes corrupted or configuration changes:
```bash
zotero-mcp update-db --force-rebuild
```

Or with full-text:
```bash
zotero-mcp update-db --fulltext --force-rebuild
```

### Complete Reset

If all else fails:
```bash
rm -rf ~/.config/zotero-mcp/chroma_db
zotero-mcp update-db --fulltext
```

## Advanced Configuration

### OpenAI Embeddings (Optional)

For better semantic search quality (requires API key and costs money):

1. **Edit `.cursor/mcp.json`**:
```json
{
  "mcpServers": {
    "zotero": {
      "command": "zotero-mcp",
      "env": {
        "ZOTERO_LOCAL": "true",
        "ZOTERO_EMBEDDING_MODEL": "openai",
        "OPENAI_API_KEY": "your-api-key-here",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small"
      }
    }
  }
}
```

2. **Rebuild database**:
```bash
rm -rf ~/.config/zotero-mcp/chroma_db
zotero-mcp update-db --fulltext --force-rebuild
```

3. **Restart Cursor**

**Note**: This adds API costs and potential token limit errors. Not recommended unless local embeddings are insufficient.

### Web API Setup (For Group Libraries)

For explicit group library access:

```bash
zotero-mcp setup --no-local --api-key YOUR_KEY --library-id GROUP_ID --library-type group
```

Requires:
- Zotero API key from https://www.zotero.org/settings/keys/new
- Group library ID (from Zotero group settings)

## Testing the Setup

After setup, test with these queries:

1. **Semantic search**:
   - "@zotero-semantic-search privacy-preserving data generation"
   - Should return relevant papers with similarity scores

2. **Keyword search**:
   - "@find-citation DiffTraj"
   - Should find papers with "DiffTraj" in title

3. **Recent papers**:
   - Use `zotero_get_recent` tool
   - Should show recently added items

4. **Database status**:
   - Run `zotero-mcp db-status`
   - Should show correct item count and configuration

## Next Steps

After successful setup:

1. **Organize library**: Use collections and tags in Zotero
2. **Regular updates**: Run `zotero-mcp update-db` after adding papers
3. **Integrate workflow**: Use semantic search for literature review
4. **Export bibliography**: Keep `references_new.bib` synced with Zotero
5. **Monitor database**: Check status periodically with `zotero-mcp db-status`

## Getting Help

For issues not covered here:
- See detailed troubleshooting in `.cursor/rules/mcp-research-tools.mdc`
- Check Zotero MCP GitHub: https://github.com/54yyyu/zotero-mcp
- Verify Zotero desktop is running and accessible
- Try rebuilding database with `--force-rebuild` flag

