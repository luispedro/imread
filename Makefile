debug: imread/*.cpp imread/lib/*.cpp imread/lib/*.h
	DEBUG=2 python setup.py build --build-lib=.

fast: imread/*.cpp imread/lib/*.h imread/lib/*.cpp
	python setup.py build --build-lib=.

clean:
	rm -rf build imread/*.so imread/lib/*.so

tests: debug
	python -m pytest

docs:
	rm -rf build/docs
	cd docs && make html && cp -r build/html ../build/docs
	@echo python setup.py upload_docs

.PHONY: clean docs tests fast debug

