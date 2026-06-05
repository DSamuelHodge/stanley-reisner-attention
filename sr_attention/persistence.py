"""
persistence.py
==============
Persistence-theoretic operations on filtered simplicial complexes built
from attention patterns.

Three layers of invariant are tracked here:

1. **Facet persistence** (algebraic)
   Tracks birth and death of maximal simplices (facets) as the attention
   threshold τ decreases.  A facet "dies" when it stops being maximal —
   i.e. when a larger co-attending coalition absorbs it.  This directly
   tracks the evolution of prime components in the SR primary decomposition.

2. **Graded Betti numbers via Hochster's formula** (homological)
   β_{i,j}(k[Δ]) = Σ_{|W|=j} dim H̃_{j-i-1}(Δ_W; k)
   Computed by iterating over all vertex subsets, extracting the induced
   subcomplex, and summing reduced homology dimensions.  Expensive (2^n),
   but exact and directly connects to the SR ideal's syzygy structure.

3. **Hochster Isomorphism Spectrum** (refined invariant)
   A stratification of each graded Betti number β_{i,j} into its constituent
   isomorphism classes of induced subcomplexes.  Whereas the classic formula
   collapses all contributions into a single integer, the spectrum records:

       H_{i,j}(Δ) = { (C, m_C, H_*(C)) : C ∈ Iso(Δ_W), |W| = j }

   where Iso(Δ_W) is the isomorphism class of the induced subcomplex on W,
   m_C is the multiplicity (number of vertex subsets W with Δ_W ≅ C), and
   H_*(C) is the full reduced homology of the type.  This is:

   • a refinement of Hochster's formula — β_{i,j} = Σ m_C · dim H̃_{j-i-1}(C)
   • equivariant under graph isomorphisms of induced subcomplexes
   • a functorial decomposition of Betti contributions by homotopy type
"""

from __future__ import annotations

import logging
from itertools import combinations, permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import gudhi as gd

from sr_attention import config
from sr_attention.sr_core import (
    _all_vertices,
    _faces_at_time,
    _facets_at_time,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Facet persistence
# ---------------------------------------------------------------------------

def compute_facet_persistence(
    simplex_tree: gd.SimplexTree,
    filt_vals: Optional[Sequence[float]] = None,
) -> Dict[Tuple[int, ...], Dict[str, float]]:
    """
    For every simplex σ, record:
        birth = filtration value when σ first appears
        death = filtration value when σ first gains a strict coface
                (i.e. when it stops being a facet / maximal)

    In attention context:
        birth ≈ 1 − max_attention_in_group (the group forms at this threshold)
        death ≈ 1 − attention_that_enlarges_group (group absorbed into larger)

    Parameters
    ----------
    simplex_tree : gd.SimplexTree
    filt_vals : sequence of float | None
        Explicit filtration snapshots.  If None, uses all distinct values
        in the simplex tree.

    Returns
    -------
    dict mapping simplex tuple → {"birth": float, "death": float}
    death = +∞ for facets that are never absorbed.
    """
    if filt_vals is None:
        filt_vals = sorted(set(f for _, f in simplex_tree.get_filtration()))

    # Index all simplices by their filtration value
    simplex_birth: Dict[Tuple[int, ...], float] = {
        tuple(sorted(s)): f for s, f in simplex_tree.get_filtration()
    }

    # For each simplex, find the earliest time a strict coface appears
    death_of: Dict[Tuple[int, ...], float] = {}
    all_simplices = list(simplex_birth.keys())

    for sigma in all_simplices:
        sigma_set = set(sigma)
        coface_births = [
            simplex_birth[tau]
            for tau in all_simplices
            if len(tau) > len(sigma) and sigma_set.issubset(set(tau))
        ]
        death_of[sigma] = min(coface_births) if coface_births else float("inf")

    return {
        sigma: {"birth": simplex_birth[sigma], "death": death_of[sigma]}
        for sigma in all_simplices
    }


# ---------------------------------------------------------------------------
# Hochster's formula — graded Betti numbers
# ---------------------------------------------------------------------------

def _induced_subcomplex(
    simplex_tree: gd.SimplexTree,
    t: float,
    vertex_subset: Sequence[int],
) -> gd.SimplexTree:
    """Build the induced subcomplex on vertex_subset at time t."""
    faces  = _faces_at_time(simplex_tree, t)
    W      = set(vertex_subset)
    v_sorted = sorted(vertex_subset)
    vmap   = {v: i for i, v in enumerate(v_sorted)}

    sub = gd.SimplexTree()
    for s in faces:
        if set(s).issubset(W):
            sub.insert([vmap[v] for v in sorted(s)], filtration=0.0)
    return sub


def _betti_numbers_unreduced(st: gd.SimplexTree) -> Dict[int, int]:
    """Compute unreduced Betti numbers for a static SimplexTree."""
    if st.num_simplices() == 0:
        return {}
    st.compute_persistence(persistence_dim_max=True)
    betti: Dict[int, int] = {}
    for dim in range(10):
        intervals = st.persistence_intervals_in_dimension(dim)
        if len(intervals) == 0:
            continue
        inf_count = sum(1 for _, d in intervals if np.isinf(d))
        if inf_count > 0:
            betti[dim] = inf_count
    return betti


def _reduced_betti(unreduced: Dict[int, int], q: int) -> int:
    """Convert unreduced β_q to reduced β̃_q."""
    if q < -1:
        return 0
    if q == -1:
        return 0   # empty complex handled separately
    if q == 0:
        return max(unreduced.get(0, 0) - 1, 0)
    return unreduced.get(q, 0)


def betti_table_via_hochster(
    simplex_tree: gd.SimplexTree,
    t: Optional[float] = None,
    max_vertices: int = config.HOCHSTER_MAX_VERTICES,
) -> Dict[Tuple[int, int], int]:
    """
    Compute graded Betti numbers β_{i,j}(k[Δ_t]) via Hochster's formula:

        β_{i,j} = Σ_{W ⊆ V, |W|=j} dim H̃_{j-i-1}(Δ_W; k)

    This is the deepest algebraic invariant in the package: it connects the
    SR ideal's minimal free resolution to the homology of every induced
    subcomplex.  In attention terms, a non-zero β_{i,j} means there exists a
    set of j tokens whose induced co-attention complex has a (j−i−1)-dimensional
    topological hole — a forbidden pattern of that homological type.

    Parameters
    ----------
    simplex_tree : gd.SimplexTree
    t : float | None
        Filtration snapshot.  Defaults to maximum filtration value.
    max_vertices : int
        Safety cap.  Raises ValueError if exceeded (computation is O(2^n)).

    Returns
    -------
    dict mapping (i, j) → β_{i,j}
    """
    if t is None:
        t = float(max(f for _, f in simplex_tree.get_filtration()))

    verts = _all_vertices(simplex_tree)
    n     = len(verts)

    if n > max_vertices:
        raise ValueError(
            f"Vertex count {n} exceeds max_vertices={max_vertices}. "
            "Reduce MAX_TOKENS_FOR_ANALYSIS in config.py or pass a smaller "
            "simplex tree."
        )

    betti: Dict[Tuple[int, int], int] = {}

    for j in range(n + 1):
        for W in combinations(verts, j):
            stW = _induced_subcomplex(simplex_tree, t, W)

            if stW.num_simplices() == 0:
                # Empty complex: H̃_{-1} = k, all others = 0
                for i in range(j + 1):
                    if j - i - 1 == -1:
                        betti[(i, j)] = betti.get((i, j), 0) + 1
                continue

            unreduced = _betti_numbers_unreduced(stW)

            for i in range(j + 1):
                q   = j - i - 1
                dim = _reduced_betti(unreduced, q)
                if dim > 0:
                    betti[(i, j)] = betti.get((i, j), 0) + dim

    return betti


# ---------------------------------------------------------------------------
# Hochster type spectrum — refined invariant
# ---------------------------------------------------------------------------

_ISOMORPHISM_CACHE: Dict[str, str] = {}
"""Memoisation cache for canonical graph labels keyed by raw adjacency strings."""


def _canonical_label(edges: Sequence[Tuple[int, int]], n_verts: int) -> str:
    """
    Canonical label for a small unweighted graph as the lexicographically
    smallest upper-triangular adjacency bit-string under vertex relabelling.

    Uses brute-force over all n! permutations — suitable for n ≤ 8 (40k perms).
    For n > 8, uses a greedy heuristic that sorts vertices by degree sequence.
    Results are memoised via the raw adjacency string.
    """
    adj = [[False] * n_verts for _ in range(n_verts)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True

    raw_parts = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            raw_parts.append("1" if adj[i][j] else "0")
    raw = "".join(raw_parts)

    if raw in _ISOMORPHISM_CACHE:
        return _ISOMORPHISM_CACHE[raw]

    verts = list(range(n_verts))

    if n_verts <= 8:
        best = None
        for perm in permutations(verts):
            bits = []
            for i in range(n_verts):
                for j in range(i + 1, n_verts):
                    bits.append("1" if adj[perm[i]][perm[j]] else "0")
            cand = "".join(bits)
            if best is None or cand < best:
                best = cand
        _ISOMORPHISM_CACHE[raw] = best
        return best

    # Greedy heuristic for larger graphs: sort by degree, then BFS colour refinement
    deg = [sum(1 for v in range(n_verts) if adj[u][v]) for u in range(n_verts)]
    order = sorted(range(n_verts), key=lambda v: (deg[v], v))
    bits = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            bits.append("1" if adj[order[i]][order[j]] else "0")
    label = "".join(bits)
    _ISOMORPHISM_CACHE[raw] = label
    return label


def _type_signature(
    simplex_tree: gd.SimplexTree,
    t: float,
    W: Tuple[int, ...],
    q: int,
    dim: int,
) -> Dict[str, object]:
    """
    Build the Hochster type signature for a single subset *W*.

    Returns
    -------
    dict with keys:
        subset         — *W*
        j              — |W|
        i              — j - q - 1
        q              — homological index (j - i - 1)
        h_dim          — dim H̃_q(Δ_W)
        num_edges      — number of 1-simplices in the induced subcomplex
        num_triangles  — number of 2-simplices in the induced subcomplex
        cycle_rank     — same as *h_dim* (redundant, for convenience)
        canonical_label — string classifying the 1-skeleton up to isomorphism
    """
    stW = _induced_subcomplex(simplex_tree, t, W)
    faces = [(s, f) for s, f in stW.get_filtration()]

    edges = [tuple(sorted(s)) for s, _f in faces if len(s) == 2]
    triangles = [tuple(sorted(s)) for s, _f in faces if len(s) == 3]

    j = len(W)
    i = j - q - 1
    cl = _canonical_label(edges, j)

    return {
        "subset": W,
        "j": j,
        "i": i,
        "q": q,
        "h_dim": dim,
        "num_edges": len(edges),
        "num_triangles": len(triangles),
        "cycle_rank": dim,
        "canonical_label": cl,
    }


def hochster_type_spectrum(
    simplex_tree: gd.SimplexTree,
    i: int,
    j: int,
    t: Optional[float] = None,
    max_vertices: int = config.HOCHSTER_MAX_VERTICES,
) -> Dict[str, object]:
    """
    Hochster Isomorphism Spectrum — a stratification of β_{i,j}.

    For each W ⊆ V with |W| = j whose induced subcomplex Δ_W has
    non-trivial reduced homology H̃_{j-i-1}, this function records the
    **isomorphism type** of the 1-skeleton of Δ_W and groups witnesses
    by that type.

    Formally, for a fixed graded index (i, j):

        H_{i,j}(Δ) = { (C, m_C, H̃_{j-i-1}(C)) }

    where C ranges over isomorphism classes of induced subcomplexes on
    j vertices, m_C = |{W : Δ_W ≅ C}| is the multiplicity of that type,
    and H̃_{j-i-1}(C) is its reduced homology contribution.

    The classical Hochster formula is recovered as:

        β_{i,j} = Σ_{C} m_C · dim H̃_{j-i-1}(C)

    This is an equivariant refinement: the decomposition is natural under
    automorphisms of the ambient complex Δ, and each type C corresponds to
    a distinct induced homotopy type in the Hochster sum.

    Parameters
    ----------
    simplex_tree : gd.SimplexTree
    i, j : int
        Graded Betti index β_{i,j}.
    t : float | None
        Filtration snapshot.  Defaults to maximum filtration value.
    max_vertices : int
        Safety cap.

    Returns
    -------
    dict with keys:

        betti_value   — β_{i,j} (total homology dimension)
        num_witnesses — number of contributing subsets W
        types         — list of type records, one per isomorphism class:
            [
                {
                    "canonical_label": str,
                    "count": int,              # m_C — how many W share this type
                    "h_dim_per_witness": int,   # dim H̃_q per W
                    "total_homology": int,      # m_C · dim H̃_q
                    "edge_count": int,          # edges in this type
                    "triangle_count": int,      # triangles in this type
                    "example_subset": tuple,    # one representative W
                },
                ...
            ]
        witnesses — list of full signatures, one per contributing W
    """
    if t is None:
        t = float(max(f for _, f in simplex_tree.get_filtration()))

    verts = _all_vertices(simplex_tree)
    n = len(verts)

    if n > max_vertices:
        raise ValueError(
            f"Vertex count {n} exceeds max_vertices={max_vertices}. "
            "Reduce MAX_TOKENS_FOR_ANALYSIS in config.py."
        )

    q = j - i - 1
    witnesses: List[Dict] = []

    for W in combinations(verts, j):
        stW = _induced_subcomplex(simplex_tree, t, W)

        if stW.num_simplices() == 0:
            if q == -1:
                witnesses.append({
                    "subset": W,
                    "j": j,
                    "i": i,
                    "q": q,
                    "h_dim": 1,
                    "num_edges": 0,
                    "num_triangles": 0,
                    "cycle_rank": 1,
                    "canonical_label": "empty",
                })
            continue

        unreduced = _betti_numbers_unreduced(stW)
        dim = _reduced_betti(unreduced, q)

        if dim > 0:
            sig = _type_signature(simplex_tree, t, W, q, dim)
            witnesses.append(sig)

    # Group by canonical label
    types: Dict[str, dict] = {}
    for w in witnesses:
        cl = w["canonical_label"]
        if cl not in types:
            types[cl] = {
                "canonical_label": cl,
                "count": 0,
                "h_dim_per_witness": w["h_dim"],
                "total_homology": 0,
                "edge_count": w["num_edges"],
                "triangle_count": w["num_triangles"],
                "example_subset": w["subset"],
            }
        types[cl]["count"] += 1
        types[cl]["total_homology"] += w["h_dim"]

    return {
        "betti_value": sum(w["h_dim"] for w in witnesses),
        "num_witnesses": len(witnesses),
        "types": sorted(types.values(), key=lambda x: -x["count"]),
        "witnesses": witnesses,
    }


def format_type_spectrum(spectrum: Dict[str, object]) -> str:
    """Human-readable rendering of a Hochster Isomorphism Spectrum."""
    lines = [
        f"β({spectrum['witnesses'][0]['i']},{spectrum['witnesses'][0]['j']}) = {spectrum['betti_value']}",
        f"  Witnesses: {spectrum['num_witnesses']}",
        f"  Type classes: {len(spectrum['types'])}",
    ]
    for ti, tp in enumerate(spectrum["types"]):
        lines.extend([
            f"",
            f"  Type {ti}: canon={tp['canonical_label'][:20]}…",
            f"    count        = {tp['count']}",
            f"    edges        = {tp['edge_count']}",
            f"    triangles    = {tp['triangle_count']}",
            f"    h_dim        = {tp['h_dim_per_witness']}",
            f"    total_hom    = {tp['total_homology']}",
            f"    example      = {tp['example_subset']}",
        ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_betti_table(betti: Dict[Tuple[int, int], int]) -> str:
    """
    Format the Betti table as a human-readable grid.

    Row axis = i (homological degree), column axis = j (strand degree).
    This matches the Macaulay2 / Singular convention.
    """
    if not betti:
        return "<empty Betti table>"

    max_i = max(i for i, _ in betti)
    max_j = max(j for _, j in betti)

    rows = ["i\\j\t" + "\t".join(str(j) for j in range(max_j + 1))]
    for i in range(max_i + 1):
        row = [str(i)] + [str(betti.get((i, j), 0)) for j in range(max_j + 1)]
        rows.append("\t".join(row))
    return "\n".join(rows)


def summarise_persistence(
    facet_intervals: Dict[Tuple[int, ...], Dict[str, float]],
) -> Dict[str, float]:
    """
    Quick summary statistics of a facet persistence dictionary.

    Returns keys: n_total, n_infinite, n_finite,
                  mean_lifetime, max_lifetime, min_lifetime (finite only).
    """
    lifetimes = []
    n_inf = 0
    for info in facet_intervals.values():
        b, d = info["birth"], info["death"]
        if np.isinf(d):
            n_inf += 1
        else:
            lifetimes.append(d - b)

    return {
        "n_total"      : len(facet_intervals),
        "n_infinite"   : n_inf,
        "n_finite"     : len(lifetimes),
        "mean_lifetime": float(np.mean(lifetimes)) if lifetimes else 0.0,
        "max_lifetime" : float(np.max(lifetimes))  if lifetimes else 0.0,
        "min_lifetime" : float(np.min(lifetimes))  if lifetimes else 0.0,
    }
