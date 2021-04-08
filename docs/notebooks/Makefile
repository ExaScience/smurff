
NOTEBOOKS = $(wildcard *.ipynb) 
TESTS_PY = $(patsubst %.ipynb,%.py,$(NOTEBOOKS))
TESTS_OK = $(patsubst %.ipynb,%.ok,$(NOTEBOOKS))


%.py: %.ipynb
	jupyter nbconvert --to=python $?

%.ok: %.py
	ipython $? && touch $@	

.PHONY: test clean clearoutput

test:  $(TESTS_OK)

clearoutput:
	jupyter nbconvert --clear-output $(NOTEBOOKS)

clean:
	rm -f $(TESTS_PY)
	rm -f $(TESTS_OK)
