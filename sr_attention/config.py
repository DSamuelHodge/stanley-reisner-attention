"""
config.py
=========
Single source of truth for every constant and tuneable parameter in
sr_attention.  Import this module; never hard-code values elsewhere.

Layout
------
MODEL_*         : default model identifiers and loading flags
CACHE_*         : TransformerLens cache / hook naming
FILTRATION_*    : how attention scores become a simplicial filtration
COMPLEX_*       : limits on simplex dimension (controls combinatorial blow-up)
HOCHSTER_*      : Hochster formula computation limits
PERSISTENCE_*   : facet-persistence thresholds
OUTPUT_*        : paths and display preferences
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MODEL defaults
# ---------------------------------------------------------------------------

# Default model.  Any HuggingFace model supported by TransformerLens works.
# Qwen2.5-0.5B is the primary development target (GQA, small, fast).
DEFAULT_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B"

# dtype passed to HookedTransformer.from_pretrained / TransformerBridge
# Options: "float32", "float16", "bfloat16"
DEFAULT_DTYPE: str = "float32"

# Move model to GPU if available
DEFAULT_MOVE_TO_DEVICE: bool = True

# Fold layer-norm weights into adjacent weight matrices for cleaner activations
# (standard TransformerLens practice; set False to inspect raw LN outputs)
FOLD_LN: bool = True

# ---------------------------------------------------------------------------
# CACHE / HOOK naming  (TransformerLens ≥ 3 canonical names)
# ---------------------------------------------------------------------------

# hook_pattern  shape: [batch, n_heads, seq_len, seq_len]
# This is the *post-softmax* attention probability matrix — exactly what we
# want.  TransformerLens expands GQA K/V heads before computing the pattern,
# so the returned shape always has the full n_heads (query heads) dimension.
HOOK_PATTERN_KEY: str = "pattern"          # short form for cache[key, layer]
HOOK_PATTERN_FULL: str = "blocks.{layer}.attn.hook_pattern"   # full form

# hook_attn_scores  shape: [batch, n_heads, seq_len, seq_len]  (pre-softmax)
# Useful for studying the *raw* score distribution before normalisation.
HOOK_ATTN_SCORES_KEY: str = "blocks.{layer}.attn.hook_attn_scores"

# ---------------------------------------------------------------------------
# FILTRATION parameters
# ---------------------------------------------------------------------------

# Number of threshold steps from 1.0 → 0.0.
# More steps = finer filtration = more accurate barcodes, but slower.
# 20 is a good default for prompts up to ~64 tokens.
FILTRATION_N_STEPS: int = 20

# Minimum threshold to include — faces with attention score below this
# are never added (avoids trivially including everything at τ = 0).
FILTRATION_MIN_THRESHOLD: float = 0.01

# Maximum threshold (attention probability ≤ 1.0 by definition).
FILTRATION_MAX_THRESHOLD: float = 1.0

# How to make a single [seq, seq] matrix from an asymmetric attention matrix.
# Options:
#   "source"   – use A[query, key] as-is (directed, standard)
#   "mean"     – symmetrize: (A + A.T) / 2
#   "max"      – symmetrize: max(A, A.T)
#   "min"      – symmetrize: min(A, A.T)  (strictest; fewer faces)
SYMMETRIZE_MODE: str = "mean"

# ---------------------------------------------------------------------------
# COMPLEX construction limits
# ---------------------------------------------------------------------------

# Maximum simplex dimension to insert when building the clique complex.
# dim=1 → graph only (edges); dim=2 → triangles; dim=3 → tetrahedra …
# Hochster's formula is exponential in the number of vertices, so keep this
# small (2 or 3) unless working on very short prompts.
MAX_SIMPLEX_DIM: int = 2

# Maximum number of tokens to use for full SR + Betti analysis.
# Longer prompts are truncated to this length before building the complex.
# (The extraction itself uses the full sequence; truncation is analysis-only.)
MAX_TOKENS_FOR_ANALYSIS: int = 16

# ---------------------------------------------------------------------------
# HOCHSTER formula limits
# ---------------------------------------------------------------------------

# Hard cap on vertex count passed to betti_table_via_hochster.
# The computation is O(2^n * n^2) — beyond ~12 vertices it becomes slow.
HOCHSTER_MAX_VERTICES: int = 12

# ---------------------------------------------------------------------------
# PERSISTENCE parameters
# ---------------------------------------------------------------------------

# Minimum facet lifetime (in threshold units) to include in barcode plots.
PERSISTENCE_MIN_LIFETIME: float = 0.0

# Cap for visualising infinite-death bars (in threshold units above max birth).
PERSISTENCE_INF_DISPLAY_DELTA: float = 0.15

# ---------------------------------------------------------------------------
# OUTPUT / display
# ---------------------------------------------------------------------------

# How many SR generators to print before truncating
SR_PRINT_LIMIT: int = 15

# Figures
FIGURE_DPI: int = 120
FIGURE_SIZE_BARCODE: tuple[int, int] = (10, 6)
FIGURE_SIZE_MATRIX: tuple[int, int] = (8, 6)

# Results directory (relative to repo root)
RESULTS_DIR: str = "results"