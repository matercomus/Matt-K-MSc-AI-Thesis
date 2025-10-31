# Knowledge Distillation for Map-Matched Trajectory Prediction

This repository contains the LaTeX source code for my Master's thesis in Artificial Intelligence at Vrije Universiteit Amsterdam.

## Research Overview

This thesis addresses the challenge of improving fast trajectory prediction models through cross-task knowledge distillation. We transfer spatial understanding from LM-TAD (a trajectory anomaly detection model) to HOSER (a fast zone-based prediction model), achieving 85-89% path completion success (47-74× improvement over baseline) while maintaining fast inference speed (~0.1ms per trajectory). The work demonstrates that "normal trajectory" knowledge learned by anomaly detection models can dramatically improve route prediction without inference-time overhead, enabling practical deployment in real-time traffic management systems, urban digital twins, and large-scale simulations.

## Repository Structure

```
.
├── main.tex                    # Main LaTeX document
├── custom-commands.tex         # Custom LaTeX commands and formatting
├── references.bib              # Bibliography file (managed via Zotero)
├── sections/                   # Thesis content sections
│   ├── 00-title-page.tex       # Title page
│   ├── 00-abstract.tex         # Abstract
│   ├── 01-introduction.tex     # Introduction
│   ├── 02-related-work.tex     # Literature review
│   ├── 03-methodology.tex      # Methodology
│   ├── 04-implementation-details.tex  # Implementation
│   ├── 05-data-preprocessing.tex      # Data preprocessing
│   ├── 06-evaluation.tex       # Evaluation and results
│   ├── 07-conclusion.tex       # Conclusion
│   ├── 08-appendix.tex         # Appendix
│   └── figures/                # TikZ figure definitions
├── llncs/                      # Springer LNCS template files
│   ├── llncs.cls               # Document class
│   ├── splncs04.bst            # Bibliography style
│   └── ...                     # Sample files and documentation
├── assets/plots/               # Generated plots and figures
├── build/                      # LaTeX build artifacts (generated)
│   ├── main.pdf                # Compiled thesis
│   └── main.txt                # Text version for review
└── notes/                      # Personal research notes (not tracked)
```

## Compilation

The project uses **LuaLaTeX** for compilation via `latexmk`, with build artifacts directed to the `build/` directory.

### Prerequisites

- TeX Live or MiKTeX distribution (with LuaTeX support)
- ChkTeX (for linting)
- latexindent (for formatting)
- `texlive-emoji` package (for emoji support)
- `pdftotext` (for generating text version via `build-and-convert.sh`)

### Building the Document

```bash
latexmk main.tex
```

The compiled PDF will be generated as `build/main.pdf`.

**Important:** Do not use the `-pdf` flag, as it forces pdfLaTeX mode. This project requires **LuaLaTeX** for:
- Native UTF-8 support via `fontspec` package
- Color emoji rendering in figures (🔥❄️)
- Modern font handling

The `.latexmkrc` configuration automatically selects LuaLaTeX (`$pdf_mode = 4`) when building.

### Alternative: Build and Convert to Text

For full-document review and analysis:

```bash
./build-and-convert.sh
```

This builds the thesis and converts it to `build/main.txt` for easier searching and cross-section analysis.

## Bibliography Management

References are managed through **Zotero**. The `references.bib` file is exported from Zotero and should not be manually edited.

To update citations:
1. Add/modify references in Zotero library
2. Export updated bibliography to `references.bib`
3. Rebuild the document with `latexmk main.tex`

## Development Workflow

### Linting and Formatting

The repository includes configuration files for code quality:
- `.chktexrc` - ChkTeX linting rules
- `.latexindent.yaml` - Code formatting preferences
- `.latexmkrc` - Build automation configuration

### Git Workflow

This project follows conventional commit conventions with gitmojis for clear version history.

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

