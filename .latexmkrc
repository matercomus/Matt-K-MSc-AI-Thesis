# Output directory for build artifacts
$out_dir = 'build';

# Ensure LaTeX can find class files in llncs/ subdirectory
ensure_path('TEXINPUTS', './llncs//');

# Ensure BibTeX can find style files in llncs/ subdirectory
ensure_path('BSTINPUTS', './llncs//');

# PDF generation using pdflatex
$pdf_mode = 1;

# Automatically invoke bibtex when needed
$bibtex_use = 2;

