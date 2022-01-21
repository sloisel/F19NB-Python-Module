all: dist

.PHONY: all clean pypi

clean:
	rm -rf dist *~ */*~

dist/.mark: setup.py
	rm -rf dist
	python3 setup.py sdist
	touch dist/.mark

dist: dist/.mark F19NB/__init__.py

pypi: all
	twine upload dist/*


