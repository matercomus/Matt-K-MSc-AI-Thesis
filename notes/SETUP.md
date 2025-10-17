# LaTeX Development Setup

## Required Tools

LaTeX Workshop extension uses ChkTeX for linting and latexindent for formatting.

### Install chktex (LaTeX linter)
```bash
sudo apt-get install chktex  # Debian/Ubuntu
```

### Install latexindent (LaTeX formatter)
```bash
sudo apt-get install texlive-extra-utils  # Includes latexindent
# Or install from CPAN for latest version:
# cpan App::latexindent
```

## LaTeX Workshop Configuration

LaTeX Workshop automatically detects ChkTeX and latexindent if installed.

### Configuration Files
- `.chktexrc` - ChkTeX linting rules (auto-detected)
- `.latexindent.yaml` - latexindent formatting rules (auto-detected)
- `.latexmkrc` - Build configuration with output directory

### LaTeX Workshop Features
- **Linting**: ChkTeX runs automatically (configurable via settings)
- **Formatting**: Use Shift+Alt+F or Cmd+Shift+I
- **Build**: Ctrl+Alt+B or auto-build on save
- **Intellisense**: Auto-completion for commands, citations, references
- **SyncTeX**: Click PDF to jump to source and vice versa

### Workspace Settings

LaTeX Workshop is configured to use the `build/` directory for all output files.

**IMPORTANT:** If you need to recreate `.vscode/settings.json`:
```bash
mkdir -p .vscode
cp latex-workshop-config.json.template .vscode/settings.json
```

Key settings configured:
- `latex-workshop.latex.outDir`: "build" - All build artifacts go to build/ directory
- `latex-workshop.latex.recipe.default`: "latexmk" - Uses .latexmkrc configuration
- `latex-workshop.linting.chktex.enabled`: Enable/disable ChkTeX linting
- `latex-workshop.latex.autoBuild.run`: Auto-build behavior

## Verification

1. Open main.tex
2. Check status bar for LaTeX icon
3. Try building: Ctrl+Alt+B
4. Check that build artifacts go to build/ directory
5. Test formatting: Shift+Alt+F on a .tex file

