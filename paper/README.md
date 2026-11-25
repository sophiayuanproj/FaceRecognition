# Scene-gated face detection on a wearable microcontroller

P04 final paper source for UCLA ECE M202A / CS M213A, revised November 24, 2026. It accompanies the final web report. The course has not specified a final-paper page limit or mandatory paper template; its final web report and reproducibility package are due December 11. Check the current course announcement for submission details.

## Open in Overleaf

The existing team project is [P04 on Overleaf](https://www.overleaf.com/project/6af8b6e054408ef325e5a19d). Update its source from this package rather than creating another project. Use **main.tex** as the main document and **pdfLaTeX** as the compiler. The document uses standard LaTeX packages, embedded PDF figures, and a BibTeX bibliography. Local compilation also works with `latexmk -pdf main.tex` or `tectonic main.tex`.

## Project files

- **main.tex** and **references.bib** contain the manuscript and references.
- **figures/** contains the five embedded result plots and an optional inference-work plot retained with its source data.
- **data/** contains the result tables behind the plots.
- **analysis/** contains figure-generation code, the numeric/provenance checker, and the scene-gate trigger and workload-order audit. Its readme gives reproduction commands.

[Public code and setup](https://github.com/sophiayuanproj/FaceRecognition) · [Saved models and raw power evidence](https://github.com/sophiayuanproj/FaceRecognition/releases/tag/research-assets-2026-11-22) · [Dataset provenance](https://github.com/sophiayuanproj/FaceRecognition/blob/main/DATA.md)

The primary result concerns fixed versus scene-gated scheduling on R19 with a native Haar detector and known-person acceptance disabled. The host recognition and compression studies are separate. USB-powered capture without an open serial host has been verified; cold boot on a physical power bank has not.
