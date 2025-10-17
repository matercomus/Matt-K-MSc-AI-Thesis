# Privacy-Preserving Synthetic Trajectory Generation for Taxi Route Anomaly Detection

This repository contains the LaTeX source code for my Master's thesis in Artificial Intelligence at Vrije Universiteit Amsterdam.

## Research Overview

This thesis addresses the challenge of generating synthetic taxi trajectory datasets that preserve essential characteristics for anomaly detection research while ensuring passenger privacy protection. The proposed framework integrates DiffTraj-based synthetic generation with LM-TAD anomaly detection to enable privacy-preserving research in urban transportation systems.

## Repository Structure

```
.
├── main.tex                    # Main LaTeX document
├── main.pdf                    # Compiled thesis (generated)
├── custom-commands.tex         # Custom LaTeX commands and formatting
├── title_page.tex             # Thesis title page
├── references_new.bib         # Bibliography file (managed via Zotero)
├── llncs/                     # Springer LNCS template files
│   ├── llncs.cls              # Document class
│   ├── splncs04.bst           # Bibliography style
│   └── ...                    # Sample files and documentation
├── assets/                    # Images and figures
│   ├── vu/                    # University branding
│   │   └── VU_logo.pdf
│   └── plots/                 # Generated plots and figures
├── build/                     # LaTeX build artifacts (auto-generated)
├── notes/                     # Markdown notes and documentation
│   └── SETUP.md              # Development setup guide
└── .cursor/                   # Cursor IDE configuration
```

## Compilation

The project uses `latexmk` for automated building with artifacts directed to the `build/` directory.

### Prerequisites

- TeX Live or MiKTeX distribution
- LaTeX Workshop extension (for VS Code/Cursor)
- ChkTeX (for linting)
- latexindent (for formatting)

See `notes/SETUP.md` for detailed installation instructions.

### Building the Document

**Using LaTeX Workshop (VS Code/Cursor):**
- Press `Ctrl+Alt+B` to build
- Build artifacts will be placed in `build/` directory

**Using Command Line:**
```bash
latexmk -pdf main.tex
```

The compiled PDF will be generated as `build/main.pdf` (or `main.pdf` in root if building manually).

## Bibliography Management

References are managed through **Zotero** using the MCP (Model Context Protocol) integration. The `references_new.bib` file is exported from Zotero and should not be manually edited.

To update citations:
1. Add/modify references in Zotero
2. Export updated bibliography to `references_new.bib`
3. Rebuild the document

## Development Workflow

### Linting and Formatting

The repository includes configuration files for code quality:
- `.chktexrc` - ChkTeX linting rules
- `.latexindent.yaml` - Code formatting preferences
- `.latexmkrc` - Build automation configuration

### Git Workflow

This project follows conventional commit conventions with gitmojis. See `.cursor/rules/git-best-practices.mdc` for details.

## Project Status

**Current Stage:** Writing and methodology development  
**Target Completion:** June 2025  
**Institution:** Vrije Universiteit Amsterdam

## Author

**Mateusz Kędzia**  
MSc Artificial Intelligence  
Vrije Universiteit Amsterdam

## License

This thesis is submitted in partial fulfillment of the requirements for the VU degree of Master of Science in Artificial Intelligence. All rights reserved.

