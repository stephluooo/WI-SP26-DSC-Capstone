NeurIPS 2026 LaTeX paper
========================

1. Download the official NeurIPS 2026 template from:
   https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/
2. Copy `neurips_2026.sty` into this directory (next to main.tex).
3. Build the PDF:
     pdflatex main.tex
     bibtex main
     pdflatex main.tex
     pdflatex main.tex

Figures copied: phase1_overview.png, phase3_roc_curve.png, phase4_cv_metrics.png, phase4_top20_predictions.png, phase4_precision_at_k.png, phase4_signal_heatmap.png
