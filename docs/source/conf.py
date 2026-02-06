# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add the package to the path for autodoc
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'spatialtissuepy'
copyright = f'{datetime.now().year}, spatialtissuepy developers'
author = 'spatialtissuepy developers'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'myst_parser',
    'nbsphinx',
    'sphinx_design',
]

# AutoDoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# AutoSummary settings
autosummary_generate = True
autosummary_imported_members = True

# Napoleon settings (for NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
}

# MyST parser settings (for Markdown support)
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# nbsphinx settings (for Jupyter notebooks)
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_timeout = 600

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_theme_options = {
    'logo': {
        'text': 'spatialtissuepy',
    },
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/yourusername/spatialtissuepy',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/spatialtissuepy/',
            'icon': 'fa-solid fa-box',
        },
    ],
    'use_edit_page_button': True,
    'show_toc_level': 2,
    'navigation_with_keys': True,
    'show_nav_level': 2,
    'navbar_align': 'left',
    'navbar_center': ['navbar-nav'],
    'footer_start': ['copyright'],
    'footer_end': ['sphinx-version', 'theme-version'],
    'secondary_sidebar_items': ['page-toc', 'edit-this-page', 'sourcelink'],
    'pygment_light_style': 'default',
    'pygment_dark_style': 'monokai',
}

html_context = {
    'github_user': 'yourusername',
    'github_repo': 'spatialtissuepy',
    'github_version': 'main',
    'doc_path': 'docs/source',
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Sidebar configuration
html_sidebars = {
    '**': [
        'search-field.html',
        'sidebar-nav-bs.html',
    ],
}

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
    ''',
}

latex_documents = [
    ('index', 'spatialtissuepy.tex', 'spatialtissuepy Documentation',
     'spatialtissuepy developers', 'manual'),
]

# -- Extension configuration -------------------------------------------------

# Copy button settings
copybutton_prompt_text = r'>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: '
copybutton_prompt_is_regexp = True

# Suppress warnings
suppress_warnings = ['autosummary']

# Add any paths that contain custom static files
def setup(app):
    app.add_css_file('custom.css')
