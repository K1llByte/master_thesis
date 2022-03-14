.ONESHELL:

all:
	cd src
	latexmk -f -synctex=1 -interaction=nonstopmode -file-line-error -pdf -outdir=./ dissertation
	cd ..


clean:
	rm -f src/dissertation.acn
	rm -f src/dissertation.aux
	rm -f src/dissertation.bbl
	rm -f src/dissertation.bcf
	rm -f src/dissertation.blg
	rm -f src/dissertation.dvi
	rm -f src/dissertation.fdb_latexmk
	rm -f src/dissertation.fls
	rm -f src/dissertation.glo
	rm -f src/dissertation.idx
	rm -f src/dissertation.ilg
	rm -f src/dissertation.ind
	rm -f src/dissertation.log
	rm -f src/dissertation.out.ps
	rm -f src/dissertation.run.xml
	rm -f src/dissertation.synctex.gz
	rm -f src/dissertation.toc