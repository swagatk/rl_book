#! /bin/bash
pdflatex main.tex
bibtex main
makeindex main 
makeglossaries main 
pythontex main 
pdflatex main 
pdflatex main
