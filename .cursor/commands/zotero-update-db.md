# Update Zotero Semantic Search Database

Update the Zotero MCP semantic search database to index new papers and changes to the library.

## Instructions

1. **Run the command-line tool** `zotero-mcp update-db` to update the vector database
2. Choose between metadata-only (fast) or full-text indexing (comprehensive)
3. Verify database update completed successfully
4. Optionally check database status after update

## When to Update

Update the database when:
- **After adding new papers to Zotero** (batch of papers added)
- **Before starting research sessions** (quick metadata update)
- **Weekly for comprehensive indexing** (full-text update)
- **After significant library changes** (tags, annotations, reorganization)
- **When search results seem outdated or incomplete**

## Update Options

### Quick Metadata Update (Fast)
```bash
zotero-mcp update-db
```
- Indexes titles, abstracts, and basic metadata
- Takes seconds to complete
- Good for quick updates before research sessions

### Full-Text Update (Comprehensive)
```bash
zotero-mcp update-db --fulltext
```
- Indexes complete paper content from PDFs
- Takes longer but provides better search results
- **Recommended weekly** for comprehensive coverage
- Requires Zotero 7+ with PDF attachments

### Force Rebuild
```bash
zotero-mcp update-db --force-rebuild
```
- Completely rebuilds database from scratch
- Use if database seems corrupted or out of sync
- Can combine with `--fulltext`: `zotero-mcp update-db --fulltext --force-rebuild`

## Check Database Status

After updating, verify the database:
```bash
zotero-mcp db-status
```

This shows:
- Number of items indexed
- Last update time
- Embedding model used
- Database configuration

## Recommended Workflow

### Daily
- Quick metadata update before research: `zotero-mcp update-db`

### Weekly
- Comprehensive full-text indexing: `zotero-mcp update-db --fulltext`
- Check database status to verify completeness

### As Needed
- After batch-adding papers from conferences or reading lists
- Before starting literature review sessions
- When semantic search returns unexpected results

## Expected Output

The command should output:
- Number of items processed
- Update duration
- Any errors or warnings
- Confirmation of successful update

## Important Notes

- **Zotero must be running** with local API enabled
- Zotero preference must allow: "Allow other applications on this computer to communicate with Zotero"
- Full-text indexing requires PDF attachments in Zotero
- Database updates are incremental (only new/changed items processed)
- Force rebuild processes entire library (use sparingly)

## Troubleshooting

**If update fails:**
1. Verify Zotero is running
2. Check Zotero API settings (Preferences → Advanced → General)
3. Try force rebuild: `zotero-mcp update-db --force-rebuild`
4. Check for error messages in output

**If semantic search still has issues after update:**
1. Verify database status: `zotero-mcp db-status`
2. Try full-text rebuild: `zotero-mcp update-db --fulltext --force-rebuild`
3. Check Zotero library has PDFs attached to items

