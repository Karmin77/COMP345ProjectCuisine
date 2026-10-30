"""Dataset loading and the shared surface normalization rules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Recipe:
    """One labelled What's Cooking recipe after validation.

    ``ingredients`` retains the source list.  ``normalized_ingredients`` is
    deterministic and is the only representation used by the modelling code.
    """

    recipe_id: int
    cuisine: str
    ingredients: tuple[str, ...]
    normalized_ingredients: tuple[str, ...]

    @property
    def group_key(self) -> str:
        """Return the exact key used to keep duplicate ingredient lists together."""

        return "\n".join(self.normalized_ingredients)

    @property
    def serialized_ingredients(self) -> str:
        """Return the frozen-embedding input serialization."""

        return self.group_key


def normalize_ingredient(value: str) -> str:
    """Lowercase, trim, and collapse whitespace in one ingredient string."""

    if not isinstance(value, str):
        raise TypeError(f"ingredient must be a string, got {type(value)!r}")
    return _WHITESPACE.sub(" ", value.strip().lower())


def normalize_ingredients(values: Sequence[str]) -> tuple[str, ...]:
    """Apply the proposal's shared normalization and lexicographic sorting."""

    if not values:
        raise ValueError("ingredient list must not be empty")
    normalized = tuple(normalize_ingredient(value) for value in values)
    if any(not value for value in normalized):
        raise ValueError("ingredient list contains an empty ingredient")
    return tuple(sorted(normalized))


def _parse_recipe(raw: object, position: int) -> Recipe:
    if not isinstance(raw, dict):
        raise ValueError(f"record {position} is not an object")
    missing = {"id", "cuisine", "ingredients"}.difference(raw)
    if missing:
        raise ValueError(f"record {position} is missing fields: {sorted(missing)}")
    recipe_id = raw["id"]
    cuisine = raw["cuisine"]
    ingredients = raw["ingredients"]
    if isinstance(recipe_id, bool) or not isinstance(recipe_id, int):
        raise ValueError(f"record {position} has a non-integer id")
    if not isinstance(cuisine, str) or not cuisine.strip():
        raise ValueError(f"record {position} has an invalid cuisine")
    if not isinstance(ingredients, list):
        raise ValueError(f"record {position} ingredients must be a list")
    normalized = normalize_ingredients(ingredients)
    return Recipe(
        recipe_id=recipe_id,
        cuisine=cuisine.strip().lower(),
        ingredients=tuple(ingredients),
        normalized_ingredients=normalized,
    )


def load_recipes(path: str | Path) -> list[Recipe]:
    """Load and validate a JSON array of labelled recipes."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"{source} must contain a JSON array")
    recipes = [_parse_recipe(item, i) for i, item in enumerate(raw)]
    ids = [recipe.recipe_id for recipe in recipes]
    if len(ids) != len(set(ids)):
        raise ValueError("recipe ids must be unique")
    if not recipes:
        raise ValueError("dataset is empty")
    return recipes


def group_summary(recipes: Iterable[Recipe]) -> dict[str, object]:
    """Summarize duplicate ingredient groups and label ambiguity."""

    groups: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    for recipe in recipes:
        groups.setdefault(recipe.group_key, set()).add(recipe.cuisine)
        counts[recipe.group_key] = counts.get(recipe.group_key, 0) + 1
    ambiguous = [key for key, labels in groups.items() if len(labels) > 1]
    return {
        "recipes": sum(counts.values()),
        "groups": len(groups),
        "ambiguous_groups": len(ambiguous),
        "ambiguous_recipes": sum(counts[key] for key in ambiguous),
    }
