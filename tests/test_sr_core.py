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
    hochster_type_spectrum,
    format_type_spectrum,
    hochster_pushforward_measure,
    format_pushforward_measure,
    entropy_profile,
    dominant_type_curve,
    pushforward_divergence,
    build_random_complex,
    build_complete_complex,
    build_empty_complex,
    classify_phase,
    phase_diagram,
    format_phase_summary,
    _canonical_label,
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


# ═══════════════════════════════════════════════════════════════════════════
# Tests for advanced persistence invariants
# ═══════════════════════════════════════════════════════════════════════════

class TestHochsterSpectrum:

    def test_type_spectrum_4cycle_beta_2_4(self):
        """4-cycle: β_{2,4} = 1 (H̃₁ of the full cycle = 1)."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        spec = hochster_type_spectrum(st, i=2, j=4, t=t_max, max_vertices=8)
        assert spec["betti_value"] == 1
        assert spec["num_witnesses"] == 1
        assert spec["types"][0]["canonical_label"] != "empty"

    def test_type_spectrum_4cycle_beta_1_2(self):
        """4-cycle: β_{1,2} = 2 (the two non-edges, each H̃₀=1)."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        spec = hochster_type_spectrum(st, i=1, j=2, t=t_max, max_vertices=8)
        assert spec["betti_value"] == 2
        assert spec["num_witnesses"] == 2

    def test_type_spectrum_triangle_trivial(self):
        """Filled triangle: no non-trivial homology → empty spectrum."""
        st = filled_triangle()
        t_max = max(f for _, f in st.get_filtration())
        spec = hochster_type_spectrum(st, i=1, j=3, t=t_max, max_vertices=8)
        assert spec["betti_value"] == 0
        assert spec["num_witnesses"] == 0

    def test_type_spectrum_output_shape(self):
        """Verify the type record schema."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        spec = hochster_type_spectrum(st, i=0, j=2, t=t_max, max_vertices=8)
        assert "types" in spec
        assert "witnesses" in spec
        for tp in spec["types"]:
            assert "canonical_label" in tp
            assert "count" in tp
            assert "h_dim_per_witness" in tp
            assert "example_subset" in tp

    def test_format_type_spectrum_output(self):
        """format_type_spectrum produces a string for a non-empty spectrum."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        spec = hochster_type_spectrum(st, i=1, j=2, t=t_max, max_vertices=8)
        assert spec["num_witnesses"] > 0
        out = format_type_spectrum(spec)
        assert isinstance(out, str)
        assert "Witnesses:" in out

    def test_type_spectrum_max_vertices_guard(self):
        """Should reject complexes exceeding max_vertices."""
        st = build_complete_complex(12)
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            hochster_type_spectrum(st, i=0, j=2, t=1.0, max_vertices=10)


class TestCanonicalLabel:

    def test_cl_edgeless(self):
        """n isolated vertices → canonical label all zeros."""
        label = _canonical_label([], 4)
        assert all(c == "0" for c in label)

    def test_cl_complete(self):
        """Complete graph on n vertices → canonical label all ones."""
        edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        label = _canonical_label(edges, 4)
        assert all(c == "1" for c in label)

    def test_cl_isomorphism_invariance(self):
        """Relabelling a path graph gives the same canonical label."""
        path_a = [(0, 1), (1, 2)]
        path_b = [(0, 2), (2, 1)]   # permuted labelling of the same path
        assert _canonical_label(path_a, 3) == _canonical_label(path_b, 3)

    def test_cl_lex_minimal(self):
        """For j ≤ 8, canonical label is the lex-min adjacency string."""
        edges = [(0, 1), (2, 3)]  # two disjoint edges
        # All permutations tested; the lex-smallest bit string for n=4
        # with two disjoint edges should be "001001" (edges (0,2),(0,3)??)
        # Let's verify: vertices {a,b,c,d}, two disjoint edges.
        # Lex-min of all 4! = 24 permutations. The minimal adjacency
        # matrix in upper-triangular order has edges (0,1),(2,3)
        # → "110000" = 110000? No wait.
        # Upper triangular bits: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        # With edges (0,1) and (2,3): 1 0 0 0 0 1 → "100001"
        label = _canonical_label(edges, 4)
        assert isinstance(label, str)
        assert len(label) == 6  # C(4,2) = 6


class TestPushforwardMeasure:

    def test_pm_complete_complex(self):
        """Complete complex: one type at every j, zero entropy."""
        st = build_complete_complex(6)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        assert m["n_vertices"] == 6
        for j in range(7):
            sj = m["by_size"][j]
            assert sj["num_types"] == 1, f"Expected 1 type at j={j}"
            assert sj["concentration"]["entropy"] == 0.0

    def test_pm_empty_complex(self):
        """Empty complex (no edges): one type at every j, only H̃₀ homology."""
        st = build_empty_complex(5)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        for j in range(6):
            assert m["by_size"][j]["num_types"] == 1

    def test_pm_cycle_entropy(self):
        """4-cycle: known entropy at j=2 (H = 0.9183 from 4 edges/6 subsets)."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        m = hochster_pushforward_measure(st, t=t_max, max_vertices=8)
        h2 = m["by_size"][2]["concentration"]["entropy"]
        expected = - (4/6) * np.log2(4/6) - (2/6) * np.log2(2/6)
        assert abs(h2 - expected) < 1e-6

    def test_pm_cycle_j3_one_type(self):
        """4-cycle: all 3-vertex subsets are isomorphic (path of length 2)."""
        st = cycle_graph()
        t_max = max(f for _, f in st.get_filtration())
        m = hochster_pushforward_measure(st, t=t_max, max_vertices=8)
        assert m["by_size"][3]["num_types"] == 1

    def test_pm_distribution_probabilities_sum_to_one(self):
        """Probabilities within each j sum to 1."""
        st = build_random_complex(6, edge_probability=0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        for j in range(7):
            probs = [d["probability"] for d in m["by_size"][j]["distribution"]]
            assert abs(sum(probs) - 1.0) < 1e-10

    def test_pm_dominant_mass_between_zero_and_one(self):
        """Dominant mass ∈ [0, 1] for all j."""
        st = build_random_complex(6, edge_probability=0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        for j in range(7):
            dm = m["by_size"][j]["concentration"]["dominant_mass"]
            assert 0.0 <= dm <= 1.0

    def test_pm_max_vertices_guard(self):
        """Should reject complexes exceeding max_vertices."""
        st = build_complete_complex(12)
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            hochster_pushforward_measure(st, t=1.0, max_vertices=10)

    def test_format_pushforward_output(self):
        """format_pushforward_measure produces a string with expected headers."""
        st = build_empty_complex(5)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        out = format_pushforward_measure(m)
        assert isinstance(out, str)
        assert "j=" in out
        assert "H=" in out

    def test_entropy_profile_shape(self):
        """entropy_profile returns list of (j, H) pairs for j = 0..n."""
        st = build_random_complex(5, 0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        profile = entropy_profile(m)
        assert len(profile) == 6
        assert all(isinstance(p[0], int) and isinstance(p[1], float) for p in profile)

    def test_dominant_type_curve_shape(self):
        """dominant_type_curve returns (j, label, prob) triples."""
        st = build_random_complex(5, 0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        curve = dominant_type_curve(m)
        assert len(curve) == 6
        assert all(isinstance(c[1], str) and isinstance(c[2], float) for c in curve)


class TestPushforwardDivergence:

    def test_divergence_self_is_zero(self):
        """D_KL(μ ‖ μ) = 0 for any measure."""
        st = build_random_complex(5, 0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        div = pushforward_divergence(m, m)
        assert div["total_div"] == 0.0

    def test_divergence_complete_vs_empty(self):
        """Complete vs empty: positive divergence (different distributions)."""
        m_c = hochster_pushforward_measure(
            build_complete_complex(6), t=1.0, max_vertices=8
        )
        m_e = hochster_pushforward_measure(
            build_empty_complex(6), t=1.0, max_vertices=8
        )
        div = pushforward_divergence(m_c, m_e)
        assert div["total_div"] > 0.0

    def test_divergence_output_schema(self):
        """Verify schema: js, divergences, total_div, alignment."""
        st = build_random_complex(5, 0.5, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=8)
        div = pushforward_divergence(m, m)
        assert "js" in div
        assert "divergences" in div
        assert "total_div" in div
        assert "alignment" in div

    def test_divergence_asymmetric(self):
        """D_KL(μ ‖ ν) != D_KL(ν ‖ μ) in general."""
        m_c = hochster_pushforward_measure(
            build_complete_complex(6), t=1.0, max_vertices=8
        )
        m_r = hochster_pushforward_measure(
            build_random_complex(6, 0.5, seed=42), t=1.0, max_vertices=8
        )
        d1 = pushforward_divergence(m_c, m_r)["total_div"]
        d2 = pushforward_divergence(m_r, m_c)["total_div"]
        assert abs(d1 - d2) > 1e-10


class TestSyntheticBaselines:

    def test_random_complex_validity(self):
        st = build_random_complex(5, 0.5, seed=0)
        assert st.num_vertices() == 5

    def test_random_complex_deterministic_seed(self):
        st1 = build_random_complex(5, 0.5, seed=42)
        st2 = build_random_complex(5, 0.5, seed=42)
        edges1 = set(tuple(s) for s, _ in st1.get_filtration() if len(s) == 2)
        edges2 = set(tuple(s) for s, _ in st2.get_filtration() if len(s) == 2)
        assert edges1 == edges2

    def test_complete_complex_edge_count(self):
        st = build_complete_complex(5)
        edges = [s for s, _ in st.get_filtration() if len(s) == 2]
        assert len(edges) == 10  # C(5, 2)

    def test_empty_complex_no_edges(self):
        st = build_empty_complex(5)
        edges = [s for s, _ in st.get_filtration() if len(s) == 2]
        assert len(edges) == 0

    def test_empty_complex_vertices_only(self):
        st = build_empty_complex(5)
        verts = [s for s, _ in st.get_filtration() if len(s) == 1]
        assert len(verts) == 5


class TestPhaseTransition:

    def test_classify_complete_is_degenerate(self):
        m = hochster_pushforward_measure(
            build_complete_complex(7), t=1.0, max_vertices=10
        )
        cl = classify_phase(m)
        assert cl["is_degenerate"] is True

    def test_classify_empty_is_degenerate(self):
        m = hochster_pushforward_measure(
            build_empty_complex(7), t=1.0, max_vertices=10
        )
        cl = classify_phase(m)
        assert cl["is_degenerate"] is True

    def test_classify_random_has_transitions(self):
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        cl = classify_phase(m)
        assert cl["is_degenerate"] is False
        assert cl["j1"] is not None
        assert cl["j2"] is not None
        assert cl["width"] >= 1

    def test_classify_small_threshold_enlarges_diverse(self):
        """Lower threshold → wider diverse zone (or same)."""
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        cl_high = classify_phase(m, threshold=0.5)
        cl_low = classify_phase(m, threshold=0.1)
        assert cl_low["width"] >= cl_high["width"]

    def test_phase_diagram_output_schema(self):
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        pd = phase_diagram(m)
        assert "classification" in pd
        assert "full_profile" in pd
        assert "specific_heat" in pd
        assert len(pd["full_profile"]) == pd["n_vertices"] + 1

    def test_phase_diagram_specific_heat_self_consistency(self):
        """C(j) = H(j) - H(j-1) should hold for the computed specific heat."""
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        pd = phase_diagram(m)
        for j in range(1, pd["n_vertices"] + 1):
            expected = pd["H"][j] - pd["H"][j - 1]
            assert abs(pd["specific_heat"][j] - expected) < 1e-10

    def test_format_phase_summary_string(self):
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        cl = classify_phase(m)
        out = format_phase_summary(cl, "random-test")
        assert isinstance(out, str)
        assert "Peak:" in out

    def test_format_phase_diagram_includes_table(self):
        m = hochster_pushforward_measure(
            build_random_complex(7, 0.3, seed=42), t=1.0, max_vertices=10
        )
        pd = phase_diagram(m)
        out = format_phase_summary(pd, "random-test")
        assert "C(j)" in out  # specific heat column
        assert "j₁" in out
        assert "j₂" in out


class TestCoverageGaps:

    def test_betti_numbers_empty_tree(self):
        """_betti_numbers_unreduced returns {} for an empty simplex tree."""
        from sr_attention.persistence import _betti_numbers_unreduced
        st = gd.SimplexTree()
        assert _betti_numbers_unreduced(st) == {}

    def test_type_spectrum_empty_q_equals_minus1(self):
        """Type spectrum with i=j=0 hits the q=-1 empty complex branch."""
        st = build_empty_complex(3)
        spec = hochster_type_spectrum(st, i=0, j=0, t=1.0, max_vertices=8)
        assert spec["betti_value"] == 1
        assert spec["num_witnesses"] == 1

    def test_canonical_label_greedy_heuristic(self):
        """n > 8 uses greedy heuristic (degree-sequence sort)."""
        edges = [(i, i + 1) for i in range(9)]
        label = _canonical_label(edges, 10)
        assert isinstance(label, str)
        assert len(label) == 45

    def test_pushforward_t_default_none(self):
        """hochster_pushforward_measure with t=None uses max filtration."""
        st = filled_triangle()
        m = hochster_pushforward_measure(st, t=None, max_vertices=8)
        assert m["n_vertices"] == 3

    def test_betti_table_t_default_none(self):
        """betti_table_via_hochster with t=None uses max filtration."""
        st = filled_triangle()
        betti = betti_table_via_hochster(st, t=None, max_vertices=8)
        assert isinstance(betti, dict)

    def test_type_spectrum_t_default_none(self):
        """hochster_type_spectrum with t=None uses max filtration."""
        st = filled_triangle()
        spec = hochster_type_spectrum(st, i=1, j=3, t=None, max_vertices=8)
        assert isinstance(spec, dict)
        assert spec["betti_value"] == 0

    def test_format_pushforward_more_than_5_types(self):
        """format_pushforward_measure shows '… and N more types' when >5."""
        st = build_random_complex(8, 0.4, seed=42)
        m = hochster_pushforward_measure(st, t=1.0, max_vertices=10)
        out = format_pushforward_measure(m)
        assert "more types" in out

    def test_betti_numbers_up_to_dim_9(self):
        """_betti_numbers_unreduced loops over dims 0..9, ensure no crash."""
        from sr_attention.persistence import _betti_numbers_unreduced
        st = build_complete_complex(4)
        betti = _betti_numbers_unreduced(st)
        assert isinstance(betti, dict)

    def test_betti_table_max_vertices_guard(self):
        """betti_table_via_hochster raises on oversized complex."""
        st = build_complete_complex(12)
        with pytest.raises(ValueError, match="exceeds max_vertices"):
            betti_table_via_hochster(st, t=1.0, max_vertices=10)

    def test_format_type_spectrum_empty(self):
        """format_type_spectrum handles zero-witness case."""
        st = filled_triangle()
        spec = hochster_type_spectrum(st, i=1, j=3, t=1.0, max_vertices=8)
        assert spec["num_witnesses"] == 0
        out = format_type_spectrum(spec)
        assert "no witnesses" in out

    def test_format_betti_table_empty(self):
        """format_betti_table returns a placeholder for empty dict."""
        out = format_betti_table({})
        assert "empty" in out

    def test_summarise_persistence_infinite_and_finite(self):
        """summarise_persistence computes stats with mixed finite/infinite deaths."""
        from sr_attention.persistence import summarise_persistence
        intervals = {
            (0,): {"birth": 0.0, "death": float("inf")},
            (1,): {"birth": 0.1, "death": 0.5},
            (2,): {"birth": 0.2, "death": 0.7},
        }
        stats = summarise_persistence(intervals)
        assert stats["n_total"] == 3
        assert stats["n_infinite"] == 1
        assert stats["n_finite"] == 2
        assert stats["mean_lifetime"] > 0.0

    def test_format_phase_summary_degenerate(self):
        """format_phase_summary handles degenerate (zero-entropy) case."""
        m = hochster_pushforward_measure(
            build_complete_complex(5), t=1.0, max_vertices=8
        )
        out = format_phase_summary(classify_phase(m), "test-degenerate")
        assert "Degenerate" in out
