"""Sphinx configuration for mgl7760-projet-equipe3."""

from __future__ import annotations

import os
import sys

# Repository root (parent of docs/)
sys.path.insert(0, os.path.abspath("../.."))

project = "mgl7760-projet-equipe3"
copyright = "2025, Master's Project"
author = "Master's Project"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
