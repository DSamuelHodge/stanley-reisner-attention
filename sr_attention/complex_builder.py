"""
complex_builder.py
==================
The bridge between TransformerLens attention scores and Gudhi simplicial
complexes.  This is the novel layer that does not exist anywhere in the
literature: it defines precisely how a [seq, seq] attention matrix becomes
a filtered simplicial complex whose Stanley–Reisner ideal encodes the
*forbidden co-attention patterns* of that head.

Mathematical definition
-----------------------
Given attention matrix A ∈ [0,1]^{n×n} and threshold τ ∈ [0,1]:

  A_sym[i,j] = symmetrize(A)[i,j]     (see config.SYMMETRIZE_MODE)

  Graph G(τ): edge {i,j} present  ⟺  A_sym[i,j] ≥ τ

  Complex Δ(τ): the *clique complex* (flag complex) of G(τ) —
    a subset σ ⊆ [n] is a face of Δ(τ) iff every pair in σ shares an edge.

Filtration direction
--------------------
Faces are assigned filtration value:
    f(σ) = 1 − min_{i,j ∈ σ, i≠j} A_sym[i,j]

so that faces appear as τ *decreases* (i.e. as filtration value *increases*
from 0 → 1). This makes the empty complex the starting point (τ=1) and the
full clique the end (τ=0), matching the standard Vietoris–Rips convention
where filtration value = "scale at which the face becomes active."

In plain English: a token pair with high attention score gets a *low*
filtration value (they co-attend early / strongly), while a low-attention
pair gets a *high* filtration value (they only co-attend weakly / late).

The Stanley–Reisner ideal I(Δ(τ)) then describes the minimal sets of tokens
that *cannot* all mutually attend above threshold τ.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import gudhi as gd

from sr_attention import config

logger = logging.getLogger(__name__)


class AttentionComplexBuilder:
    """
    Convert a single [seq, seq] attention matrix into a filtered SimplexTree.

    Parameters
    ----------
    max_simplex_dim : int
        Maximum simplex dimension to insert (2 = triangles, 3 = tetrahedra …).
        Higher dimensions give richer topology but are exponentially more
        expensive for Betti computation.
    symmetrize_mode : str
        How to turn the asymmetric attention matrix into a symmetric edge
        weight.  One of "mean", "max", "min", "source".
    n_steps : int
        Number of threshold steps in [min_threshold, max_threshold].
    min_threshold : float
        Lowest attention score at which to include edges.
    max_threshold : float
        Upper bound (should be 1.0 for post-softmax probabilities).
    max_tokens : int
        Truncate sequences longer than this before building the complex.
    """

    def __init__(
        self,
        max_simplex_dim : int   = config.MAX_SIMPLEX_DIM,
        symmetrize_mode : str   = config.SYMMETRIZE_MODE,
        n_steps         : int   = config.FILTRATION_N_STEPS,
        min_threshold   : float = config.FILTRATION_MIN_THRESHOLD,
        max_threshold   : float = config.FILTRATION_MAX_THRESHOLD,
        max_tokens      : int   = config.MAX_TOKENS_FOR_ANALYSIS,
    ) -> None:
        self.max_simplex_dim = max_simplex_dim
        self.symmetrize_mode = symmetrize_mode
        self.n_steps         = n_steps
        self.min_threshold   = min_threshold
        self.max_threshold   = max_threshold
        self.max_tokens      = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        attn_matrix: np.ndarray,
        token_labels: Optional[list[str]] = None,
    ) -> gd.SimplexTree:
        """
        Build a filtered SimplexTree from a [seq, seq] attention matrix.

        Parameters
        ----------
        attn_matrix : np.ndarray
            Shape [seq, seq].  Post-softmax attention probabilities.
        token_labels : list[str] | None
            Optional token strings for logging/debugging.

        Returns
        -------
        gd.SimplexTree
            Filtered simplicial complex ready for SR ideal extraction or
            persistence computation.  Filtration value f(σ) = 1 − min_edge.
        """
        A = self._prepare_matrix(attn_matrix)
        n = A.shape[0]

        if token_labels is not None and len(token_labels) != n:
            logger.warning("token_labels length %d ≠ matrix size %d — ignoring labels.",
                           len(token_labels), n)
            token_labels = None

        logger.debug("Building complex: n_tokens=%d, max_dim=%d, mode=%s",
                     n, self.max_simplex_dim, self.symmetrize_mode)

        st = gd.SimplexTree()
        self._insert_vertices(st, n)
        self._insert_edges_and_cliques(st, A, n)

        # Gudhi requires pruning to enforce the max dimension
        st.prune_above_dimension(self.max_simplex_dim)

        return st

    def build_at_threshold(
        self,
        attn_matrix: np.ndarray,
        threshold: float,
    ) -> gd.SimplexTree:
        """
        Build a *static* (non-filtered) clique complex at a fixed threshold.

        Useful for quick SR ideal extraction at one particular τ.
        """
        A   = self._prepare_matrix(attn_matrix)
        n   = A.shape[0]
        st  = gd.SimplexTree()
        self._insert_vertices(st, n, filtration=0.0)
        self._insert_edges_and_cliques_at_threshold(st, A, n, threshold)
        st.prune_above_dimension(self.max_simplex_dim)
        return st

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_matrix(self, A: np.ndarray) -> np.ndarray:
        """Validate, truncate, and symmetrize the attention matrix."""
        A = np.asarray(A, dtype=np.float64)

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"attn_matrix must be square 2D; got shape {A.shape}")

        n = A.shape[0]
        if n > self.max_tokens:
            logger.info("Truncating sequence from %d to %d tokens.", n, self.max_tokens)
            A = A[:self.max_tokens, :self.max_tokens]

        # Zero out diagonal (self-attention does not form edges)
        np.fill_diagonal(A, 0.0)

        # Symmetrize
        A = self._symmetrize(A)
        return A

    def _symmetrize(self, A: np.ndarray) -> np.ndarray:
        mode = self.symmetrize_mode
        if mode == "mean":
            return (A + A.T) / 2.0
        elif mode == "max":
            return np.maximum(A, A.T)
        elif mode == "min":
            return np.minimum(A, A.T)
        elif mode == "source":
            return A   # asymmetric; pairs are (query→key) weight
        else:
            raise ValueError(
                f"Unknown symmetrize_mode={mode!r}. "
                "Choose from: 'mean', 'max', 'min', 'source'."
            )

    def _insert_vertices(
        self, st: gd.SimplexTree, n: int, filtration: float = 0.0
    ) -> None:
        """Insert all n vertices with the given filtration value."""
        for i in range(n):
            st.insert([i], filtration=filtration)

    def _edge_filtration(self, weight: float) -> float:
        """
        Convert an attention weight ∈ [0,1] to a filtration value.

        f = 1 − weight:  high attention → low filtration → appears early.
        Clipped to [0, 1].
        """
        return float(np.clip(1.0 - weight, 0.0, 1.0))

    def _insert_edges_and_cliques(
        self, st: gd.SimplexTree, A: np.ndarray, n: int
    ) -> None:
        """
        Insert all edges {i,j} with A[i,j] ≥ min_threshold, then extend to
        the clique complex up to max_simplex_dim via Gudhi's
        ``expansion`` method.

        Each edge {i,j} gets filtration value = 1 − A[i,j].
        Higher-order simplices inherit the maximum filtration value of their
        constituent edges (Gudhi's default clique-complex convention).
        """
        for i in range(n):
            for j in range(i + 1, n):
                w = float(A[i, j])
                if w >= self.min_threshold:
                    f_val = self._edge_filtration(w)
                    st.insert([i, j], filtration=f_val)

        # Extend edges to full clique complex up to max_simplex_dim
        if self.max_simplex_dim >= 2:
            st.expansion(self.max_simplex_dim)

    def _insert_edges_and_cliques_at_threshold(
        self, st: gd.SimplexTree, A: np.ndarray, n: int, threshold: float
    ) -> None:
        """Static variant: insert only edges with A[i,j] ≥ threshold."""
        for i in range(n):
            for j in range(i + 1, n):
                w = float(A[i, j])
                if w >= threshold:
                    st.insert([i, j], filtration=0.0)

        if self.max_simplex_dim >= 2:
            st.expansion(self.max_simplex_dim)

    # ------------------------------------------------------------------
    # Convenience: batch build across all heads of a layer
    # ------------------------------------------------------------------

    def build_all_heads(
        self,
        layer_patterns: np.ndarray,
        token_labels: Optional[list[str]] = None,
    ) -> list[gd.SimplexTree]:
        """
        Build one SimplexTree per head for a full layer.

        Parameters
        ----------
        layer_patterns : np.ndarray
            Shape [n_heads, seq, seq].

        Returns
        -------
        list of gd.SimplexTree, length n_heads.
        """
        if layer_patterns.ndim != 3:
            raise ValueError(f"Expected [n_heads, seq, seq]; got {layer_patterns.shape}")
        return [
            self.build(layer_patterns[h], token_labels=token_labels)
            for h in range(layer_patterns.shape[0])
        ]
