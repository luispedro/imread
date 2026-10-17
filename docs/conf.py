# Sphinx configuration for the imread documentation.
#
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# imread is not installed in the Read the Docs environment (building it needs
# the C image libraries), so the version is read from the source tree instead
# of importing it.
with open(os.path.join(_root, 'imread', 'imread_version.py'), encoding='utf-8') as f:
    imread_version = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read()).group(1)

# autodoc imports imread from the source tree. The compiled extension and NumPy
# are mocked (see autodoc_mock_imports below), so neither needs to be installed.
sys.path.insert(0, _root)

# -- Project information -------------------------------------------------------

project = 'imread'
author = 'Luis Pedro Coelho'
copyright = '2012-2026, Luis Pedro Coelho'

# The short X.Y version
version = '.'.join(imread_version.split('.')[:2])
# The full version, including alpha/beta/rc tags
release = imread_version

# -- General configuration -----------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    ]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build']

myst_enable_extensions = ['deflist']

autodoc_mock_imports = ['numpy', 'imread._imread']
autodoc_member_order = 'bysource'

# The docstrings are in NumPy style
napoleon_google_docstring = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'ndarray': 'numpy.ndarray',
    'file-like': ':term:`file-like <python:file-like object>`',
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

pygments_style = 'sphinx'

# -- Options for HTML output ---------------------------------------------------

html_theme = 'nature'

html_sidebars = {
    '*': ['searchbox.html', 'sidebar.html'],
}
