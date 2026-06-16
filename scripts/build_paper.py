"""Build the SENTINEL IEEE paper.

If a LaTeX toolchain (pdflatex+bibtex) is present, compiles
research_paper/sentinel_becithcon2026.tex via pdflatex -> bibtex -> pdflatex x2.
Otherwise prints the Overleaf upload instructions (this machine has no MiKTeX).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1] / "research_paper"
STEM = "sentinel_becithcon2026"


def main() -> int:
    if not (PAPER_DIR / f"{STEM}.tex").exists():
        print(f"missing {STEM}.tex in {PAPER_DIR}")
        return 1
    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")
    if not (pdflatex and bibtex):
        print("No local LaTeX toolchain (pdflatex/bibtex) found.\n"
              "Overleaf build: create a project and upload\n"
              f"  - research_paper/{STEM}.tex   (set as main document)\n"
              "  - research_paper/references.bib\n"
              "  - research_paper/figures/        (the .pdf figures)\n"
              "  - research_paper/IEEEtran.cls and IEEEtran.bst (or rely on Overleaf's built-in)\n"
              "Overleaf compiles IEEEtran out of the box; just press Recompile.")
        return 0
    seq = [[pdflatex, "-interaction=nonstopmode", STEM],
           [bibtex, STEM],
           [pdflatex, "-interaction=nonstopmode", STEM],
           [pdflatex, "-interaction=nonstopmode", STEM]]
    for cmd in seq:
        print("  $", " ".join(Path(c).name if i == 0 else c for i, c in enumerate(cmd)))
        r = subprocess.run(cmd, cwd=PAPER_DIR, capture_output=True, text=True)
        if r.returncode != 0 and cmd[0] == pdflatex:
            tail = "\n".join(r.stdout.splitlines()[-25:])
            print(f"pdflatex error:\n{tail}")
            return r.returncode
    pdf = PAPER_DIR / f"{STEM}.pdf"
    print(f"OK -> {pdf}" if pdf.exists() else "build finished but no PDF produced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
