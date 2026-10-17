PIP_EDITABLE = python -m pip install --editable . --no-deps --no-build-isolation

debug: imread/*.cpp imread/lib/*.cpp imread/lib/*.h
	$(PIP_EDITABLE) --config-settings=build-dir=build/debug --config-settings=setup-args=-Dbuildtype=release --config-settings=setup-args=-Db_ndebug=false --config-settings=setup-args=-Dglibcpp_debug=true

fast: imread/*.cpp imread/lib/*.h imread/lib/*.cpp
	$(PIP_EDITABLE) --config-settings=build-dir=build/fast --config-settings=setup-args=-Dbuildtype=release

install:
	python -m pip install .

clean:
	rm -rf build dist .mesonpy-* imread/*.so imread/lib/*.so

tests: debug
	python -m pytest -v

docs:
	rm -rf build/docs
	sphinx-build -b html docs build/docs
	@echo python -m build

.PHONY: clean docs tests fast debug install
