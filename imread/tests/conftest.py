import pytest

@pytest.fixture(autouse=True)
def _run_in_tmp_path(tmp_path, monkeypatch):
    # Tests that write files use fixed relative names, so run each test in its
    # own temporary directory. pytest deletes old ones, and nothing is left in
    # the working directory.
    monkeypatch.chdir(tmp_path)
