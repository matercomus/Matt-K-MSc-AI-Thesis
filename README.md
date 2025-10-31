# Knowledge Distillation for Map-Matched Trajectory Prediction

This repository contains the LaTeX source code for my Master's thesis in Artificial Intelligence at Vrije Universiteit Amsterdam.

> [!WARNING]
> **Work in Progress** - This thesis is currently under active development. Content, structure, and results may change as the research progresses.

## 📄 View Thesis

**[Read the thesis online (PDF)](https://matercomus.github.io/Matt-K-MSc-AI-Thesis/thesis.pdf)** - Automatically updated with each commit to main branch

## Research Overview

This thesis addresses a fundamental challenge in urban trajectory prediction: how to achieve transformer-level spatial reasoning while maintaining millisecond-scale inference speeds required for real-time traffic management.

**The Problem:** Existing fast prediction models suffer from poor route completion rates (12-18%), limiting their practical deployment. While sophisticated models like LM-TAD achieve superior spatial reasoning, their computational overhead (~3.4ms per trajectory vs ~0.1ms for fast models) prevents real-time application in city-wide traffic management systems.

**The Solution:** Through training-time knowledge distillation, we transfer spatial understanding from LM-TAD (a trajectory anomaly detection model) to HOSER (a fast zone-based prediction model). We demonstrate that repurposing the "normal trajectory" knowledge learned by anomaly detection models enables dramatic improvements in route prediction without inference-time overhead.

**Key Results on Beijing Dataset (40,060 roads, 629,380 trajectories):**
- **85-89% path completion success** (47-74× improvement over vanilla baseline's 12-18%)
- **87% better distance distribution matching** (JSD: 0.016-0.022 vs 0.145-0.153)
- **98% better spatial pattern fidelity** (radius JSD: 0.003-0.004 vs 0.198-0.206)
- **Realistic trip lengths** (6.4 km vs vanilla's 2.4 km, real: 5.2 km)
- **Maintained fast inference speed** (~0.1ms per trajectory, enabling real-time deployment)

**Novel Findings:** Hyperparameter optimization reveals that minimal distillation weight (λ=0.0014) with high temperature (τ=4.37) enables effective knowledge transfer. This counter-intuitive result suggests that subtle distributional guidance is more effective than aggressive knowledge transfer, allowing the student model to integrate spatial priors while preserving its architectural strengths.

**Impact:** The resulting system enables practical deployment for policy makers and traffic regulators, supporting applications in real-time traffic signal optimization, infrastructure planning, urban digital twins, agent-based traffic simulation, and high-quality synthetic trajectory data generation. This work demonstrates the viability of cross-task knowledge distillation for trajectory prediction and provides a scalable framework for integrating AI-based route prediction into operational traffic management systems.

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

## Citation

If you found this work useful, please consider citing it:

```bibtex
@mastersthesis{kedzia2025knowledge,
  title     = {Knowledge Distillation for Map-Matched Trajectory Prediction: Improving Urban Route Prediction through Cross-Task Knowledge Transfer},
  author    = {K{\k{e}}dzia, Mateusz},
  school    = {Vrije Universiteit Amsterdam},
  year      = {2025},
  type      = {Master's thesis},
  note      = {MSc in Artificial Intelligence}
}
```

