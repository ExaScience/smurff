
NOTEBOOKS = $(wildcard *.ipynb) 
TESTS_PY = $(patsubst %.ipynb,%.py,$(NOTEBOOKS))
TESTS_OK = $(patsubst %.ipynb,%.ok,$(NOTEBOOKS))


%.py: %.ipynb
	jupyter nbconvert --to=python $?

%.ok: %.py
	ipython $? && touch $@	

test:  $(TESTS_OK)


clean:
	rm -f $(TESTS_PY)
	rm -f $(TESTS_OK)
