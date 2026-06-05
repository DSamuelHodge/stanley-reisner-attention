"""
persistence.py
==============
Persistence-theoretic operations on filtered simplicial complexes built
from attention patterns.

Two independent notions of persistence are tracked here:

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
"""

from __future__ import annotations

import logging
from itertools import combinations
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
