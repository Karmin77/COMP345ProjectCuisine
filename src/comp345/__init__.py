"""Reproducible COMP 345 cuisine-classification experiments.

The package follows the Version 2 proposal: shared ingredient normalization,
group-aware splits, TF-IDF and frozen-embedding probes, and two fusion models.
"""

__version__ = "0.1.0"

from .data import Recipe, load_recipes, normalize_ingredient, normalize_ingredients
from .labels import LabelSpace

__all__ = [
    "LabelSpace",
    "Recipe",
    "load_recipes",
    "normalize_ingredient",
    "normalize_ingredients",
]
