---
name: generate-full-text
description: Build thesis PDF, convert to text, read full content, and perform specified analysis (duplicates, consistency, terminology, etc.)
---

# Generate Full Text and Analyze

## Workflow

You must follow this exact workflow:

1. **Build and Convert**: Run `./build-and-convert.sh` to generate `build/main.txt`

2. **Read Full Text**: Read `build/main.txt` completely
   - If file is ≤2000 lines: read entire file at once
   - If file is >2000 lines: read in sections (500-1000 line chunks with overlap)
   - Note the document structure (sections, subsections) while reading

3. **Perform Analysis**: Execute the analysis task specified by the user after "and"
   - Look for duplicates, redundancies, inconsistencies
   - Check terminology usage across sections
   - Verify cross-references and claims
   - Assess document flow and structure
   - Any other cross-section analysis requested

4. **Report Findings**: Provide specific findings with:
   - Line numbers or section references from build/main.txt
   - Actual content excerpts showing issues
   - Specific recommendations for fixes
   - File paths and approximate line numbers in source .tex files

5. **Ask for Confirmation**: Before making changes to source files, summarize proposed edits and ask for user approval

## Key Principles

- **Don't assume section independence**: Content may be duplicated or contradicted across sections
- **Be thorough**: Read the entire text before making conclusions
- **Be specific**: Cite exact locations (line numbers, sections) when reporting issues
- **Be cautious**: Always ask before making structural changes
- **Cross-reference**: Check consistency between introduction, methodology, evaluation, and conclusion

## Example Analyses

**Duplicate detection**: Find repeated paragraphs, sentences, or explanations across sections

**Terminology consistency**: Verify terms like "privacy budget", "differential privacy", "trajectory generation" are used consistently

**Cross-reference verification**: Check that claims in intro/conclusion match methodology/evaluation details

**Structural flow**: Assess narrative progression and identify logical gaps or repetition

## Usage Examples

Users will invoke this command like:
```
/generate-full-text and find all duplicated content across sections
/generate-full-text and check terminology consistency for "privacy budget"
/generate-full-text and verify cross-references are accurate
/generate-full-text and identify redundant explanations
/generate-full-text and assess overall document flow
```

## Important Notes

- Always run the build script first to ensure you have the latest version
- Read the ENTIRE text file before making conclusions about duplicates or consistency
- When the file is large (>2000 lines), read in overlapping chunks to maintain context
- Provide line numbers from both `build/main.txt` (for reference) and the source `.tex` files (for editing)
- Never make assumptions about one section based only on another - always verify by reading both
