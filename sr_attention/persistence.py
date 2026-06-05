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
import math
from itertools import combinations, permutations
from math import comb
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


# ---------------------------------------------------------------------------
# Pushforward measure — distribution of isomorphism types
# ---------------------------------------------------------------------------

def hochster_pushforward_measure(
    simplex_tree: gd.SimplexTree,
    t: Optional[float] = None,
    max_vertices: int = config.HOCHSTER_MAX_VERTICES,
) -> Dict[str, object]:
    """
    Pushforward measure from induced subcomplexes to isomorphism types.

    For each subset size *j* define:

        Ωⱼ = { W ⊆ V : |W| = j }
        φⱼ : Ωⱼ → Types,    φⱼ(W) = isomorphism_type(Δ_W)
        μⱼ = (φⱼ)₊ Uniform(Ωⱼ)

    i.e. μⱼ is the distribution over isomorphism types obtained by picking a
    uniformly random *j*-vertex subset and recording the isomorphism class of
    its induced subcomplex.

    The function reports:

    • the full distribution μⱼ for each *j*
    • concentration diagnostics (entropy, dominant mass, participation ratio)
    • phase transitions — values of *j* where the dominant type changes

    Parameters
    ----------
    simplex_tree : gd.SimplexTree
    t : float | None
        Filtration snapshot.  Defaults to maximum filtration value.
    max_vertices : int
        Safety cap.

    Returns
    -------
    dict with keys:

        n_vertices    — total number of vertices
        by_size       — dict mapping *j* → size-*j* distribution and diagnostics:
            {
                "total_subsets": int,        # C(n, j)
                "num_types": int,             # number of distinct types at this j
                "num_homologically_active": int,  # types with non-zero H̃ contribution
                "distribution": [
                    {
                        "canonical_label": str,
                        "count": int,          # how many W map to this type
                        "probability": float,  # = count / C(n, j)
                        "edges": int,
                        "triangles": int,
                        "cycle_rank": dict,    # q → dim H̃_q for this type
                        "example_subset": tuple,
                    },
                    ...
                ],
                "concentration": {
                    "entropy": float,          # Shannon entropy of μⱼ (bits)
                    "dominant_mass": float,    # max_t μⱼ(t)
                    "participation_ratio": float,  # 1 / Σ_t μⱼ(t)²
                    "types_for_90_pct": int,   # min types covering 90 % mass
                },
            }
        phase_transitions : list of transition records:
            {
                "j": int,
                "from_label": str,
                "to_label": str,
                "from_count": int,
                "to_count": int,
            }
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

    # ── per-size accumulators ──────────────────────────────────────────
    by_size: Dict[int, dict] = {}
    for j in range(n + 1):
        by_size[j] = {
            "total_subsets": comb(n, j),
            "types": {},  # canonical_label → accumulator
        }

    # ── single pass over all subsets ───────────────────────────────────
    for j in range(n + 1):
        acc = by_size[j]["types"]
        for W in combinations(verts, j):
            stW = _induced_subcomplex(simplex_tree, t, W)
            faces = [(s, _f) for s, _f in stW.get_filtration()]

            edges = [tuple(sorted(s)) for s, _f in faces if len(s) == 2]
            triangles = [tuple(sorted(s)) for s, _f in faces if len(s) == 3]

            cl = _canonical_label(edges, j) if j > 0 else "empty"

            # homology
            if stW.num_simplices() == 0:
                unreduced = {}
            else:
                unreduced = _betti_numbers_unreduced(stW)

            cycle_rank: Dict[str, int] = {}
            for q in range(-1, j):
                dim = _reduced_betti(unreduced, q)
                if dim > 0:
                    cycle_rank[str(q)] = dim

            if cl not in acc:
                acc[cl] = {
                    "canonical_label": cl,
                    "count": 0,
                    "edges": len(edges),
                    "triangles": len(triangles),
                    "cycle_rank": cycle_rank,
                    "example_subset": W,
                }
            acc[cl]["count"] += 1
            # union cycle_rank (should be uniform within a type, but take max
            # across all members to be safe)
            for k, v in cycle_rank.items():
                old = acc[cl]["cycle_rank"].get(k, 0)
                if v > old:
                    acc[cl]["cycle_rank"][k] = v

    # ── build distribution + concentration for each j ─────────────────
    result_by_size: Dict[int, dict] = {}
    dominant_labels: Dict[int, str] = {}  # for phase transition detection

    for j in range(n + 1):
        total = by_size[j]["total_subsets"]
        raw_types = list(by_size[j]["types"].values())
        if not raw_types:
            result_by_size[j] = {
                "total_subsets": total,
                "num_types": 0,
                "num_homologically_active": 0,
                "distribution": [],
                "concentration": {
                    "entropy": 0.0,
                    "dominant_mass": 0.0,
                    "participation_ratio": 0.0,
                    "types_for_90_pct": 0,
                },
            }
            continue

        # sort descending by count
        raw_types.sort(key=lambda x: -x["count"])

        distribution = []
        for rt in raw_types:
            distribution.append({
                "canonical_label": rt["canonical_label"],
                "count": rt["count"],
                "probability": rt["count"] / total,
                "edges": rt["edges"],
                "triangles": rt["triangles"],
                "cycle_rank": rt["cycle_rank"],
                "example_subset": rt["example_subset"],
            })

        probs = [d["probability"] for d in distribution]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        dominant_mass = probs[0]
        participation_ratio = 1.0 / sum(p * p for p in probs)

        # types covering 90 % of mass
        cum = 0.0
        types_90 = 0
        for p in probs:
            cum += p
            types_90 += 1
            if cum >= 0.9:
                break

        num_hom_active = sum(
            1 for d in distribution if d["cycle_rank"]
        )

        result_by_size[j] = {
            "total_subsets": total,
            "num_types": len(distribution),
            "num_homologically_active": num_hom_active,
            "distribution": distribution,
            "concentration": {
                "entropy": round(ent, 6),
                "dominant_mass": round(dominant_mass, 6),
                "participation_ratio": round(participation_ratio, 6),
                "types_for_90_pct": types_90,
            },
        }

        # record dominant label
        dominant_labels[j] = distribution[0]["canonical_label"]

    # ── phase transition detection ─────────────────────────────────────
    phase_transitions = []
    prev_label = None
    for j in range(1, n + 1):
        cur_label = dominant_labels.get(j)
        if cur_label is None:
            continue
        if prev_label is not None and cur_label != prev_label:
            phase_transitions.append({
                "j": j,
                "from_label": prev_label,
                "to_label": cur_label,
                "from_count": result_by_size[j - 1]["distribution"][0]["count"],
                "to_count": result_by_size[j]["distribution"][0]["count"],
            })
        prev_label = cur_label

    return {
        "n_vertices": n,
        "by_size": result_by_size,
        "phase_transitions": phase_transitions,
    }


def format_pushforward_measure(measure: Dict[str, object]) -> str:
    """Human-readable rendering of the pushforward measure."""
    lines = [
        f"Hochster pushforward measure — {measure['n_vertices']} vertices",
        f"{'=' * 60}",
    ]
    for j in sorted(measure["by_size"].keys()):
        sj = measure["by_size"][j]
        lines.append(f"")
        lines.append(f"j={j:2d}  ({sj['total_subsets']:4d} subsets, "
                      f"{sj['num_types']:2d} types, "
                      f"{sj['num_homologically_active']} homologically active)")
        conc = sj["concentration"]
        lines.append(f"      H={conc['entropy']:.4f}  "
                      f"dom={conc['dominant_mass']:.4f}  "
                      f"PR={conc['participation_ratio']:.4f}  "
                      f"90%={conc['types_for_90_pct']} types")
        for ti, d in enumerate(sj["distribution"][:5]):
            cl_short = d["canonical_label"][:16]
            cr_str = ",".join(f"H̃{q}={v}" for q, v in sorted(d["cycle_rank"].items()))
            lines.append(f"      [{ti}] p={d['probability']:.4f} "
                          f"E={d['edges']} T={d['triangles']} "
                          f"{cr_str}  canon={cl_short}…")
        if len(sj["distribution"]) > 5:
            lines.append(f"      … and {len(sj['distribution']) - 5} more types")

    if measure["phase_transitions"]:
        lines.append(f"")
        lines.append(f"Dominant-type phase transitions:")
        for pt in measure["phase_transitions"]:
            frm = pt["from_label"][:20] if pt["from_label"] else "(empty)"
            to_ = pt["to_label"][:20]   if pt["to_label"]   else "(empty)"
            lines.append(f"  j={pt['j']}: {frm} → {to_}  "
                          f"(n={pt['from_count']} → {pt['to_count']})")
    return "\n".join(lines)


def format_type_spectrum(spectrum: Dict[str, object]) -> str:
    """Human-readable rendering of a Hochster Isomorphism Spectrum."""
    if not spectrum["witnesses"]:
        return f"β_{{(?,?)}} = 0  (no witnesses)"

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
# Combinatorial phase transition — regime analysis
# ---------------------------------------------------------------------------

def classify_phase(
    measure: Dict[str, object],
    threshold: float = 0.15,
) -> Dict[str, object]:
    """
    Classify each subset size *j* into one of three combinatorial phases.

    Let μⱼ = (φⱼ)₊ Uniform(Ωⱼ) be the pushforward measure at subset size
    j (see `hochster_pushforward_measure`), and let:

        H(j) = H(μⱼ)              — Shannon entropy (bits)
        H_max = max_{0 ≤ k ≤ n} H(k)

    Define the **entropy threshold** T = threshold × H_max.

    The three phases are:

    1. **Rigid** (j < j₁): H(j) < T — few types dominate, the measure
       is sharply concentrated.  At the extreme, a single isomorphism
       class accounts for nearly all j-vertex subsets.  Homological
       contributions are uniform across the vertex set.

    2. **Diverse** (j₁ ≤ j ≤ j₂): H(j) ≥ T — the measure spreads over
       many types.  This is the regime where the graph generates genuine
       combinatorial variety: distinct subsets produce distinct induced
       subcomplex topologies.  Typically peaks near j ≈ n/2.

    3. **Collapse** (j > j₂): H(j) < T again — as j approaches n, the
       number of possible isomorphism types shrinks (only 1 type at
       j = n), and the measure concentrates again.

    The boundaries j₁ and j₂ are the first and last *j* where H(j)
    exceeds the threshold, starting the search from j ≥ 2 (j = 0, 1 are
    always rigid by construction).

    Parameters
    ----------
    measure : dict
        Output of ``hochster_pushforward_measure()``.
    threshold : float
        Fraction of peak entropy used to define the phase cut.
        Must be in (0, 1).  Default 0.15 (15 %).  Higher values → wider
        rigid/collapse zones; lower values → wider diverse zone.

    Returns
    -------
    dict with keys:

        n_vertices      — total vertex count
        phase_of_j      — dict mapping *j* → ``"rigid"`` | ``"diverse"`` |
                           ``"collapse"`` | ``"degenerate"``
        peak_j          — j* = argmaxⱼ H(j)
        peak_entropy    — H(j*)  (bits)
        j1              — rigid→diverse transition point
        j2              — diverse→collapse transition point
        width           — number of *j* in the diverse phase
        threshold_used  — actual entropy cutoff (bits)
        transitions     — list of transition dicts:
            [{"j": j₁, "from": "rigid", "to": "diverse", "H_j": H(j₁)},
             {"j": j₂, "from": "diverse", "to": "collapse", "H_j": H(j₂)}]
        is_degenerate   — True if H(j) ≈ 0 for all j
    """
    n = measure["n_vertices"]
    by_size = measure["by_size"]

    entropy = np.array([by_size[j]["concentration"]["entropy"] for j in range(n + 1)])
    h_max = float(np.max(entropy))
    t_used = threshold * h_max if h_max > 1e-12 else 0.0

    j_star = int(np.argmax(entropy))

    # ── degenerate case: entropy flat at zero ──────────────────────
    if h_max < 1e-12:
        phase_of_j = {j: "degenerate" for j in range(n + 1)}
        return {
            "n_vertices": n,
            "phase_of_j": phase_of_j,
            "peak_j": 0,
            "peak_entropy": 0.0,
            "j1": None,
            "j2": None,
            "width": 0,
            "threshold_used": 0.0,
            "transitions": [],
            "is_degenerate": True,
        }

    # ── find boundaries ────────────────────────────────────────────
    j1: Optional[int] = None
    for j in range(2, n + 1):
        if entropy[j] >= t_used:
            j1 = j
            break

    j2: Optional[int] = None
    for j in range(n, 1, -1):
        if entropy[j] >= t_used:
            j2 = j
            break

    # ── classify ───────────────────────────────────────────────────
    phase_of_j = {}
    for j in range(n + 1):
        if j < j1:
            phase_of_j[j] = "rigid"
        elif j1 is not None and j <= j2:
            phase_of_j[j] = "diverse"
        elif j2 is not None:
            phase_of_j[j] = "collapse"
        else:
            phase_of_j[j] = "rigid"

    transitions = []
    if j1 is not None:
        transitions.append({
            "j": j1,
            "from": "rigid",
            "to": "diverse",
            "H_j": round(entropy[j1], 6),
        })
    if j2 is not None and (j2 != j1):
        transitions.append({
            "j": j2,
            "from": "diverse",
            "to": "collapse",
            "H_j": round(entropy[j2], 6),
        })

    return {
        "n_vertices": n,
        "phase_of_j": phase_of_j,
        "peak_j": j_star,
        "peak_entropy": round(h_max, 6),
        "j1": j1,
        "j2": j2,
        "width": (j2 - j1 + 1) if (j1 is not None and j2 is not None) else 0,
        "threshold_used": round(t_used, 6),
        "transitions": transitions,
        "is_degenerate": False,
    }


def phase_diagram(
    measure: Dict[str, object],
    threshold: float = 0.15,
) -> Dict[str, object]:
    """
    Full combinatorial phase diagram for a pushforward measure.

    Extends ``classify_phase()`` with per-j diagnostics and the discrete
    "specific heat" — the first difference of the entropy profile,
    C(j) = H(j) - H(j-1), which measures how sharply diversity changes
    across a transition.

    Parameters
    ----------
    measure : dict
        Output of ``hochster_pushforward_measure()``.
    threshold : float
        Entropy fraction for phase boundaries (passed to
        ``classify_phase()``).

    Returns
    -------
    dict with keys:

        n_vertices   — total vertex count
        classification — output of ``classify_phase()``
        j_range      — list of all *j* values
        H            — list of H(j) matching *j_range*
        n_types      — list of type counts matching *j_range*
        dominant_mass — list of α(j) matching *j_range*
        participation_ratio — list of PR(j) matching *j_range*
        specific_heat — list of C(j) = H(j) - H(j-1) for j ≥ 1 (first entry 0)
        full_profile — list of per-j dicts for plotting/analysis:
            [
                {
                    "j": int,
                    "phase": str,
                    "H": float,
                    "n_types": int,
                    "dominant_mass": float,
                    "participation_ratio": float,
                    "specific_heat": float,
                },
                ...
            ]
    """
    n = measure["n_vertices"]
    by_size = measure["by_size"]

    cl = classify_phase(measure, threshold=threshold)

    j_range = list(range(n + 1))
    H_vals = []
    n_types = []
    dom_mass = []
    pr_vals = []

    for j in j_range:
        c = by_size[j]["concentration"]
        H_vals.append(c["entropy"])
        n_types.append(by_size[j]["num_types"])
        dom_mass.append(c["dominant_mass"])
        pr_vals.append(c["participation_ratio"])

    specific_heat = [0.0] + [H_vals[j] - H_vals[j - 1] for j in range(1, n + 1)]

    full_profile = []
    for j in j_range:
        full_profile.append({
            "j": j,
            "phase": cl["phase_of_j"][j],
            "H": H_vals[j],
            "n_types": n_types[j],
            "dominant_mass": dom_mass[j],
            "participation_ratio": pr_vals[j],
            "specific_heat": round(specific_heat[j], 6),
        })

    return {
        "n_vertices": n,
        "classification": cl,
        "j_range": j_range,
        "H": H_vals,
        "n_types": n_types,
        "dominant_mass": dom_mass,
        "participation_ratio": pr_vals,
        "specific_heat": specific_heat,
        "full_profile": full_profile,
    }


def format_phase_summary(
    phase_result: Dict[str, object],
    location: Optional[str] = None,
) -> str:
    """
    Human-readable phase summary.

    Parameters
    ----------
    phase_result : dict
        Output of ``classify_phase()`` or ``phase_diagram()``.
    location : str | None
        Optional label for the head/model (e.g. ``"gpt2_L0H5"``).

    Returns
    -------
    str
        Formatted summary table.
    """
    # Handle both direct classify_phase() and full phase_diagram() outputs
    if "classification" in phase_result:
        cl = phase_result["classification"]
        full = phase_result["full_profile"]
    else:
        cl = phase_result
        # Reconstruct a flat profile from classify_phase output
        n = cl["n_vertices"]
        full = None

    n = cl["n_vertices"]
    header = "Combinatorial phase diagram"
    if location:
        header += f" — {location}"
    lines = [header]
    lines.append("─" * 78)

    if cl.get("is_degenerate", False):
        lines.append("Degenerate: entropy is zero at all scales.")
        return "\n".join(lines)

    lines.append(
        f"Peak: H(μ_{{{cl['peak_j']}}}) = {cl['peak_entropy']:.4f} bits  "
        f"(argmax j* = {cl['peak_j']})"
    )
    lines.append(
        f"Threshold: ε = {cl['threshold_used']:.4f} bits "
        f"(= {cl['threshold_used'] / cl['peak_entropy'] * 100:.0f} % of peak)"
    )
    lines.append(
        f"Diverse regime width: {cl['width']} "
        f"(j₁ = {cl['j1']},  j₂ = {cl['j2']})"
    )
    lines.append("")

    if full is not None:
        # Table header
        lines.append(
            f"  j  {'phase':<12s} {'H':>8s} {'types':>5s} {'α(j)':>8s} {'PR':>8s}  {'C(j)':>8s}"
        )
        lines.append("  " + "─" * 60)
        for entry in full:
            j = entry["j"]
            phase = entry["phase"]
            H = entry["H"]
            nt = entry["n_types"]
            dm = entry["dominant_mass"]
            pr = entry["participation_ratio"]
            sh = entry["specific_heat"]

            # Mark transitions
            phase_display = phase
            for tr in cl.get("transitions", []):
                if tr["j"] == j and tr["to"] != phase:
                    phase_display = f"{phase} → {tr['to']}"

            lines.append(
                f"  {j:2d}  {phase_display:<12s} {H:>8.4f} {nt:>5d} {dm:>8.4f} "
                f"{pr:>8.4f}  {sh:>+8.4f}"
            )
        lines.append("")

    # Transitions
    for tr in cl.get("transitions", []):
        lines.append(
            f"  j{chr(0x2081) if tr['to'] == 'diverse' else chr(0x2082)} = {tr['j']}:  "
            f"{tr['from']} ──→ {tr['to']}  "
            f"(H = {tr['H_j']:.4f})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis and comparison utilities
# ---------------------------------------------------------------------------

def entropy_profile(measure: Dict[str, object]) -> List[Tuple[int, float]]:
    """
    Entropy-vs-j curve from a pushforward measure.

    Returns a list ``[(j, H(μⱼ)), ...]`` for j = 0 … n, where
    H(μⱼ) is the Shannon entropy (bits) of the type distribution
    at subset size *j*.
    """
    return [
        (j, measure["by_size"][j]["concentration"]["entropy"])
        for j in sorted(measure["by_size"].keys())
    ]


def dominant_type_curve(measure: Dict[str, object]) -> List[Tuple[int, str, float]]:
    """
    Dominant-type-vs-j: ``(j, canonical_label, probability)`` for each *j*.
    """
    out = []
    for j in sorted(measure["by_size"].keys()):
        dist = measure["by_size"][j]["distribution"]
        if dist:
            out.append((j, dist[0]["canonical_label"], dist[0]["probability"]))
        else:
            out.append((j, "", 0.0))
    return out


def pushforward_divergence(
    measure_p: Dict[str, object],
    measure_q: Dict[str, object],
) -> Dict[str, object]:
    """
    Kullback–Leibler divergence between two pushforward measures at each *j*.

    For each subset size *j*, computes:

        D_KL(μⱼ ‖ νⱼ) = Σ_t μⱼ(t) · log₂(μⱼ(t) / νⱼ(t))

    where μⱼ and νⱼ are the type distributions of *measure_p* and *measure_q*.
    Only types present in *both* measures contribute; types in μⱼ but absent
    from νⱼ receive a probability floor of 10⁻⁸ to keep divergence finite.

    Returns
    -------
    dict with keys:

        js          — list of subset sizes
        divergences — list of D_KL values (bits), same order as *js*
        total_div   — Σⱼ D_KL(μⱼ ‖ νⱼ)  (sum across all j)
        alignment   — list of dicts, one per j:
            {
                "j": int,
                "kl": float,
                "types_p": int,
                "types_q": int,
                "types_common": int,
            }
    """
    js = sorted(set(measure_p["by_size"].keys()) & set(measure_q["by_size"].keys()))
    alignments = []
    divergences = []
    EPS = 1e-8

    for j in js:
        dp = {d["canonical_label"]: d["probability"]
              for d in measure_p["by_size"][j]["distribution"]}
        dq = {d["canonical_label"]: d["probability"]
              for d in measure_q["by_size"][j]["distribution"]}

        all_types = set(dp) | set(dq)
        kl = 0.0
        for t in all_types:
            pp = dp.get(t, 0.0)
            qq = dq.get(t, EPS)          # floor to avoid log(0)
            if pp > 0 and qq > 0:
                kl += pp * math.log2(pp / qq)

        divergences.append(kl)
        alignments.append({
            "j": j,
            "kl": round(kl, 6),
            "types_p": len(dp),
            "types_q": len(dq),
            "types_common": len(set(dp) & set(dq)),
        })

    return {
        "js": js,
        "divergences": divergences,
        "total_div": round(sum(divergences), 6),
        "alignment": alignments,
    }


# ---------------------------------------------------------------------------
# Synthetic baseline complexes
# ---------------------------------------------------------------------------

def build_random_complex(
    n_vertices: int,
    edge_probability: float = 0.5,
    seed: Optional[int] = None,
) -> gd.SimplexTree:
    """
    Erdős–Rényi random simplicial complex: each edge appears independently
    with probability *edge_probability*, then extended to a clique complex
    up to dimension ``MAX_SIMPLEX_DIM`` (from config).

    This provides a null baseline for the pushforward measure — a random
    attention graph with no structural constraints should produce a distinct
    entropy profile.
    """
    rng = np.random.default_rng(seed)
    st = gd.SimplexTree()
    for v in range(n_vertices):
        st.insert([v], filtration=0.0)

    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < edge_probability:
                st.insert([i, j], filtration=0.0)

    max_dim = config.MAX_SIMPLEX_DIM
    if max_dim >= 2:
        st.expansion(max_dim)

    return st


def build_complete_complex(n_vertices: int) -> gd.SimplexTree:
    """
    Complete simplicial complex on *n_vertices* — the fully saturated
    baseline where every edge is present.

    The pushforward measure for a complete complex is deterministic:
    every *j*-vertex subset induces the complete simplex on *j* vertices,
    giving a single isomorphism type at every *j*.
    """
    st = gd.SimplexTree()
    for v in range(n_vertices):
        st.insert([v], filtration=0.0)
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            st.insert([i, j], filtration=0.0)

    max_dim = config.MAX_SIMPLEX_DIM
    if max_dim >= 2:
        st.expansion(max_dim)

    return st


def build_empty_complex(n_vertices: int) -> gd.SimplexTree:
    """
    Empty baseline: vertices only, no edges.

    The pushforward measure for an empty complex has a single type at every
    *j*: the type of *j* isolated vertices.  Homologically quiet everywhere.
    """
    st = gd.SimplexTree()
    for v in range(n_vertices):
        st.insert([v], filtration=0.0)
    return st


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
