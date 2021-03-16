.PHONY: help clean html 

all: html

notebooks := $(wildcard notebooks/*.ipynb)
sources := $(wildcard *.rst)

html: $(notebooks) $(sources)
	sphinx-build -b html .  _build/html

clean:
	rm -r _build 
