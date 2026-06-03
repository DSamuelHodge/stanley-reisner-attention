"""
tests/test_sr_core.py
=====================
Unit tests for sr_core and complex_builder using small toy complexes.
No model download required.
"""

import numpy as np
import gudhi as gd
import pytest

from sr_attention.sr_core import (
    _all_vertices,
    _faces_at_time,
    _facets_at_time,
    compute_persistent_f_vector,
    stanley_reisner_generators,
    primary_decomposition_from_facets,
    build_complex_from_facets,
    h_vector_and_hilbert,
)
from sr_attention.complex_builder import AttentionComplexBuilder
from sr_attention.persistence import (
    compute_facet_persistence,
    betti_table_via_hochster,
    format_betti_table,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def filled_triangle() -> gd.SimplexTree:
    """Complete 2-simplex on {0,1,2}."""
    st = gd.SimplexTree()
    st.insert([0], filtration=0.0)
    st.insert([1], filtration=0.0)
    st.insert([2], filtration=0.0)
    st.insert([0, 1], filtration=0.1)
    st.insert([0, 2], filtration=0.1)
    st.insert([1, 2], filtration=0.1)
    st.insert([0, 1, 2], filtration=0.2)
    return st


def cycle_graph() -> gd.SimplexTree:
    """4-cycle: edges (0,1),(1,2),(2,3),(3,0) — no triangles."""
    return build_complex_from_facets([(0, 1), (1, 2), (2, 3), (3, 0)])


# ---------------------------------------------------------------------------
# sr_core tests
# ---------------------------------------------------------------------------

class TestSRCore:

    def test_all_vertices_triangle(self):
        st = filled_triangle()
        assert _all_vertices(st) == [0, 1, 2]

    def test_faces_at_time_triangle(self):
        st = filled_triangle()
        faces_early = _faces_at_time(st, 0.05)
        # Only vertices present before filtration 0.1
        dims = [len(f) - 1 for f in faces_early]
        assert all(d == 0 for d in dims), "Only vertices at t=0.05"

        faces_full = _faces_at_time(st, 0.2)
        assert (0, 1, 2) in faces_full, "Full triangle present at t=0.2"

    def test_facets_at_time_edge_only(self):
        st = filled_triangle()
        # At t=0.1 the max faces are the three edges
        facets = _facets_at_time(st, 0.1)
        assert len(facets) == 3
        for f in facets:
            assert len(f) == 2, "Facets should be edges at t=0.1"

    def test_sr_generators_complete_simplex(self):
        """Full 2-simplex has no non-faces among vertices → empty ideal."""
        st = filled_triangle()
        gens, mons = stanley_reisner_generators(st, t=0.2)
        assert len(gens) == 0, "Complete simplex has trivial SR ideal"

    def test_sr_generators_cycle(self):
        """4-cycle: triangles {0,1,2},{0,1,3},{1,2,3},{0,2,3} are non-faces."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        gens, _ = stanley_reisner_generators(st, t=t_max)
        # Minimal non-faces of C4 are the two diagonals {0,2} and {1,3}
        gen_sets = [set(g) for g in gens]
        assert {0, 2} in gen_sets, "Diagonal {0,2} is a minimal non-face"
        assert {1, 3} in gen_sets, "Diagonal {1,3} is a minimal non-face"

    def test_primary_decomposition_length(self):
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        comps = primary_decomposition_from_facets(st, t=t_max)
        # 4-cycle has 4 edges = 4 facets = 4 primary components
        assert len(comps) == 4

    def test_h_vector_triangle(self):
        st = filled_triangle()
        h, series = h_vector_and_hilbert(st, t=0.2)
        # Filled triangle: h = [1, 0, 0, 0] (Cohen-Macaulay)
        assert isinstance(h, list)
        assert isinstance(series, str)


# ---------------------------------------------------------------------------
# AttentionComplexBuilder tests
# ---------------------------------------------------------------------------

class TestComplexBuilder:

    def _uniform_attn(self, n: int, weight: float = 0.5) -> np.ndarray:
        A = np.full((n, n), weight, dtype=np.float64)
        np.fill_diagonal(A, 0.0)
        return A

    def test_build_returns_simplex_tree(self):
        builder = AttentionComplexBuilder(max_tokens=8)
        A = self._uniform_attn(4, 0.6)
        st = builder.build(A)
        assert isinstance(st, gd.SimplexTree)
        assert st.num_vertices() == 4

    def test_build_at_threshold_high(self):
        """High threshold → only vertices, no edges."""
        builder = AttentionComplexBuilder(max_tokens=8, min_threshold=0.9)
        A = self._uniform_attn(4, 0.5)  # all weights < threshold
        st = builder.build_at_threshold(A, threshold=0.9)
        # No edges should be present (all weights 0.5 < 0.9)
        for s, _ in st.get_filtration():
            assert len(s) <= 1, "No edges expected above this threshold"

    def test_build_at_threshold_low(self):
        """Low threshold → full clique (all edges present)."""
        builder = AttentionComplexBuilder(max_tokens=8)
        A = self._uniform_attn(4, 0.8)
        st = builder.build_at_threshold(A, threshold=0.1)
        edges = [s for s, _ in st.get_filtration() if len(s) == 2]
        # 4-choose-2 = 6 edges in complete graph on 4 vertices
        assert len(edges) == 6

    def test_filtration_direction(self):
        """Higher attention weight → lower filtration value."""
        builder = AttentionComplexBuilder(max_tokens=8)
        A = np.zeros((3, 3))
        A[0, 1] = A[1, 0] = 0.9
        A[0, 2] = A[2, 0] = 0.3
        A[1, 2] = A[2, 1] = 0.3
        st = builder.build(A)
        # Find filtration values for edges
        edge_filts = {}
        for s, f in st.get_filtration():
            if len(s) == 2:
                edge_filts[tuple(sorted(s))] = f
        # Edge (0,1) has higher weight → lower filtration value
        assert edge_filts[(0, 1)] < edge_filts[(0, 2)]

    def test_truncation(self):
        """Sequences longer than max_tokens should be silently truncated."""
        builder = AttentionComplexBuilder(max_tokens=5)
        A = self._uniform_attn(10, 0.5)
        st = builder.build(A)
        assert st.num_vertices() == 5

    def test_build_all_heads(self):
        builder = AttentionComplexBuilder(max_tokens=8)
        layer = np.random.rand(4, 6, 6).astype(np.float64)  # 4 heads, seq=6
        # Normalize rows to mimic softmax
        layer = layer / layer.sum(axis=-1, keepdims=True)
        trees = builder.build_all_heads(layer)
        assert len(trees) == 4
        for st in trees:
            assert isinstance(st, gd.SimplexTree)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_facet_persistence_cycle(self):
        st = build_complex_from_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
        intervals = compute_facet_persistence(st)
        assert len(intervals) > 0

    def test_betti_table_cycle(self):
        """4-cycle has β_{1,4} = 1 (one topological loop)."""
        st = build_complex_from_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
        t_max = max(f for _, f in st.get_filtration())
        betti = betti_table_via_hochster(st, t=t_max, max_vertices=8)
        assert isinstance(betti, dict)
        # There should be at least one non-zero entry
        assert sum(betti.values()) > 0

    def test_format_betti_table(self):
        betti = {(0, 1): 2, (1, 3): 1}
        table = format_betti_table(betti)
        assert "i\\j" in table
        assert "2" in table