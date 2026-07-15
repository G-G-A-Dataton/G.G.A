"""Backward-compatible access to the canonical attribute feature module.

The production implementation lives in :mod:`src.attributes`; keeping a
second parser caused feature names and matching semantics to diverge.
"""

from src.attributes import (  # noqa: F401
    ATTRIBUTE_FEATURE_COLS,
    add_attribute_features,
    compute_color_match,
    compute_material_match,
    compute_size_match,
    extract_query_colors,
    extract_query_materials,
    extract_query_sizes,
    get_attribute_value,
    parse_attributes,
    parse_color,
    parse_material,
    parse_size,
)


extract_color_from_query = extract_query_colors
extract_material_from_query = extract_query_materials
