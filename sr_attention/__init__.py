"""
sr_attention — Stanley–Reisner Ideals for Transformer Attention Analysis.

A novel framework bridging algebraic combinatorics and mechanistic
interpretability: the first application of Stanley–Reisner ring theory
to characterise forbidden co-attention patterns in transformer attention
heads.

Public API
----------
Extraction:
    AttentionExtractor, AttentionCache

Complex building:
    AttentionComplexBuilder

SR algebra:
    stanley_reisner_generators, primary_decomposition_from_facets,
    compute_persistent_f_vector, h_vector_and_hilbert,
    build_complex_from_facets

Persistence & Hochster:
    betti_table_via_hochster, compute_homology_over_fields,
    format_torsion_report, hochster_type_spectrum,
    hochster_pushforward_measure, format_type_spectrum,
    format_pushforward_measure, compute_facet_persistence,
    entropy_profile, dominant_type_curve, pushforward_divergence,
    classify_phase, phase_diagram, format_phase_summary,
    build_random_complex, build_complete_complex, build_empty_complex

Config:
    config module (all tuneable parameters)
"""

from sr_attention.extraction import AttentionExtractor, AttentionCache
from sr_attention.complex_builder import AttentionComplexBuilder

from sr_attention.sr_core import (
    stanley_reisner_generators,
    primary_decomposition_from_facets,
    compute_persistent_f_vector,
    h_vector_and_hilbert,
    build_complex_from_facets,
)

from sr_attention.persistence import (
    # Persistence
    compute_facet_persistence,
    summarise_persistence,
    # Hochster's formula
    betti_table_via_hochster,
    compute_homology_over_fields,
    format_torsion_report,
    format_betti_table,
    # Hochster type spectrum
    hochster_type_spectrum,
    format_type_spectrum,
    # Pushforward measure
    hochster_pushforward_measure,
    format_pushforward_measure,
    entropy_profile,
    dominant_type_curve,
    pushforward_divergence,
    # Phase transitions
    classify_phase,
    phase_diagram,
    format_phase_summary,
    # Synthetic baselines
    build_random_complex,
    build_complete_complex,
    build_empty_complex,
)

from sr_attention import config
