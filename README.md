**Stanley–Reisner Ideals for Transformer Attention Analysis**

> *A novel framework bridging algebraic combinatorics and mechanistic interpretability: the first application of Stanley–Reisner ring theory to characterise forbidden co-attention patterns in transformer attention heads.*

---

## What and Why

Deep learning is rich in heuristics and engineering tricks, but poor in first-principles algebraic understanding of *why* its core mechanisms work. Transformer attention is no exception: the literature analyses softmax through statistical physics, mean-field theory, and convex geometry — but never through algebraic combinatorics.

This library introduces a different lens.

**The central observation:** the set of token subsets that *can* mutually co-attend above a threshold τ forms a simplicial complex Δ(τ). The complement — the minimal sets of tokens that *cannot* all simultaneously co-attend — generates the **Stanley–Reisner ideal** I(Δ(τ)).

This gives us a new algebraic language for attention:

| Classical language | This framework |
|---|---|
| "These tokens attend to each other" | "This token subset is a face of Δ(τ)" |
| "These tokens cannot co-attend" | "This subset is a generator of I(Δ(τ))" |
| "Attention head is sparse" | "I(Δ(τ)) has many low-degree generators" |
| "Attention head is diffuse" | "Δ(τ) is close to the full simplex; I(Δ(τ)) ≈ 0" |
| "Layer-wise complexity" | "Evolution of Betti numbers β_{i,j} across layers" |
| "Induction heads" | "Specific prime components of I(Δ(τ)) persist across layers" |

The SR ideal lives in the polynomial ring k[x₁, …, xₙ] where each variable xᵢ represents a token. Its **primary decomposition** partitions tokens into maximal co-attending groups. Its **graded Betti numbers** (via Hochster's formula) encode topological holes in the forbidden-pattern structure.

---

## Mathematical Background

### Simplicial Complex from Attention

Given attention matrix **A** ∈ ℝⁿˣⁿ (post-softmax) and threshold τ ∈ (0, 1]:

1. **Symmetrize**: A_sym[i,j] = (A[i,j] + A[j,i]) / 2
2. **Threshold graph** G(τ): edge {i,j} ∈ E ⟺ A_sym[i,j] ≥ τ
3. **Flag complex** Δ(τ): a subset σ ⊆ [n] is a face iff every pair in σ shares an edge in G(τ)

### Filtration Direction

Each edge {i,j} gets filtration value **f({i,j}) = 1 − A_sym[i,j]**, so:
- High attention ↔ low filtration value ↔ face appears early
- Low attention ↔ high filtration value ↔ face appears late

This makes the filtration **Δ(0) ⊇ Δ(τ₁) ⊇ Δ(τ₂) ⊇ … ⊇ Δ(1) = ∅** a proper nested family as τ increases.

### Stanley–Reisner Ideal

For simplicial complex Δ on n vertices:

```
I(Δ) = ⟨x_{i₁}·…·x_{iₖ} : {i₁,…,iₖ} ∉ Δ is a minimal non-face⟩
```

Each generator is a squarefree monomial corresponding to a **minimal forbidden co-attention pattern**.

### Primary Decomposition

```
I(Δ) = ⋂_{σ ∈ Facets(Δ)} P_σ,   where P_σ = ⟨x_j : j ∉ σ⟩
```

Each prime component P_σ describes the tokens *excluded from* the maximal co-attending group σ.

### Hochster's Formula

The graded Betti numbers of the SR ring k[Δ] = k[x₁,…,xₙ]/I(Δ):

```
β_{i,j}(k[Δ]) = Σ_{W ⊆ [n], |W|=j} dim H̃_{j-i-1}(Δ_W; k)
```

A non-zero β_{i,j} means there exists a set of j tokens whose induced co-attention complex has a (j−i−1)-dimensional topological hole — a forbidden pattern with that homological signature.

---

## Installation

```bash
git clone https://git.hodgederrick.com/hodge360/stanley-reisner-attention
cd sr-attention
pip install -e ".[dev]"
```

**Dependencies:** `gudhi ≥ 3.8`, `transformer-lens ≥ 3.0`, `torch ≥ 2.1`, `numpy`, `matplotlib`

---

## Quick Start

```python
from sr_attention import AttentionExtractor, AttentionComplexBuilder
from sr_attention import stanley_reisner_generators, betti_table_via_hochster
from sr_attention import config

# 1. Load model and extract attention patterns
extractor = AttentionExtractor("Qwen/Qwen2.5-0.5B")
cache     = extractor.run("The mathematician proved the theorem using algebra.")

# cache.patterns has shape [n_layers, n_heads, seq_len, seq_len]
print(f"Layers: {cache.n_layers}, Heads: {cache.n_heads}, Tokens: {cache.seq_len}")
print(f"Tokens: {cache.tokens}")

# 2. Build the filtered simplicial complex for layer 0, head 0
attn    = cache.head(layer=0, head=0)   # [seq, seq] attention matrix
builder = AttentionComplexBuilder()
st      = builder.build(attn, token_labels=cache.tokens)

# 3. Extract SR ideal generators (forbidden co-attention patterns)
gens, monomials = stanley_reisner_generators(st, token_labels=cache.tokens, verbose=True)
# e.g. "x_The·x_proved" means tokens "The" and "proved" cannot co-attend above τ

# 4. Primary decomposition (maximal co-attending groups)
from sr_attention import primary_decomposition_from_facets
components = primary_decomposition_from_facets(st, token_labels=cache.tokens, verbose=True)

# 5. Betti table (topological structure of forbidden patterns)
betti = betti_table_via_hochster(st)
from sr_attention import format_betti_table
print(format_betti_table(betti))
```

### Command-line demo

```bash
python scripts/run_analysis.py \
    --model  Qwen/Qwen2.5-0.5B \
    --prompt "The cat sat on the mat." \
    --layer  0 \
    --head   0 \
    --all-heads

# With torsion detection (multi-field homology comparison)
python scripts/run_analysis.py \
    --model  Qwen/Qwen2.5-0.5B \
    --prompt "The cat sat on the mat." \
    --layer  0 \
    --head   0 \
    --torsion-check
```

---

## Torsion Detection

Betti numbers computed via Hochster's formula depend on the coefficient field F_p.
By the Universal Coefficient Theorem:

```
dim_{F_p} H_q(Δ_W; F_p) = rank H_q(Δ_W; Z) + dim_{F_p} Tor(H_{q-1}(Δ_W; Z), F_p)
```

A Betti number that *differs* between F_2 and F_3 signals 2- or 3-torsion in the
integer homology of some induced subcomplex.  Two functions automate this check:

```python
from sr_attention import compute_homology_over_fields, format_torsion_report

# Compare Betti tables across F_2, F_3, F_5, F_7, F_11, F_997
results = compute_homology_over_fields(st, max_vertices=12)

print(format_torsion_report(results))
# Output:
#   TORSION DETECTED — Betti numbers vary across coefficient fields
#   Fields tested: F_2, F_3, F_5, F_7, F_11, F_997
#   Affected graded indices: 1
#     i   j   F_2   F_3   F_5   F_7   F_11  F_997  Δ
#     ─────────────────────────────────────────────
#     1   3   1     0     0     0     0     0     1
#
#   Interpretation:
#     β_{i,j}^{F_p} - β_{i,j}^{F_q} > 0  ⇒  p- or q-torsion in integer homology
```

The default homology field for all computations is set in `config.HOCHSTER_HOMOLOGY_FIELD`
(default: `2`, i.e. F_2).  **Important:** Gudhi's `SimplexTree.compute_persistence()`
defaults to 11 (F_11), not 2 — the library passes the config value explicitly for
reproducibility.

Default field list for `compute_homology_over_fields()`: `[2, 3, 5, 7, 11, 997]`.
The entry 997 serves as a proxy for characteristic-zero (no small torsion), since
Gudhi does not support ℚ directly.  Any prime up to 46337 is supported.

```bash
# CLI shortcut
python scripts/run_analysis.py --model Qwen/Qwen2.5-0.5B --prompt "..." --torsion-check
```

---

## Module Structure

```
sr_attention/
│
├── __init__.py          # Public API surface (lists every exported symbol)
├── config.py            # All constants and tuneable parameters ← edit here
│
├── extraction.py        # TransformerLens ≥3 model loading + cache extraction
│                        # AttentionExtractor: model → AttentionCache
│                        # Supports HookedTransformer + TransformerBridge (TL3)
│                        # GQA-safe: hook_pattern always returns [batch, n_heads, seq, seq]
│
├── complex_builder.py   # AttentionComplexBuilder: [seq,seq] → gd.SimplexTree
│                        # THE NOVEL BRIDGE LAYER (no prior art)
│                        # Defines filtration: f(σ) = 1 − min_edge_weight(σ)
│                        # Builds flag/clique complex via Gudhi expansion()
│
├── sr_core.py           # Stanley–Reisner algebra
│                        # stanley_reisner_generators()
│                        # primary_decomposition_from_facets()
│                        # compute_persistent_f_vector()
│                        # h_vector_and_hilbert()
│                        # build_complex_from_facets()
│
├── persistence.py       # Persistence-theoretic operations
│                        # betti_table_via_hochster()      ← Hochster's formula
│                        # compute_homology_over_fields()  ← multi-field torsion detection
│                        # format_torsion_report()
│                        # compute_facet_persistence()
│                        # summarise_persistence()
│                        # format_betti_table()
│
└── utils.py             # Token labels, display, numpy helpers

tests/
├── test_sr_core.py      # Unit tests (no model download needed)
└── ...

scripts/
└── run_analysis.py      # End-to-end CLI demo

notebooks/               # Exploratory Jupyter notebooks (add yours here)
```

---

## Configuration

All tuneable parameters live in `sr_attention/config.py`. Never hard-code values elsewhere.

Key parameters and their effects:

| Parameter | Default | Effect |
|---|---|---|
| `DEFAULT_MODEL_NAME` | `Qwen/Qwen2.5-0.5B` | Model to analyse |
| `MAX_TOKENS_FOR_ANALYSIS` | `16` | Tokens used for SR/Betti (Hochster is 2^n) |
| `MAX_SIMPLEX_DIM` | `2` | Max triangle dimension in complex |
| `FILTRATION_N_STEPS` | `20` | Threshold resolution for persistence |
| `SYMMETRIZE_MODE` | `"mean"` | How to symmetrize attention matrix |
| `HOCHSTER_MAX_VERTICES` | `12` | Hard cap for Betti computation |
| `HOCHSTER_HOMOLOGY_FIELD` | `2` | Coefficient field F_p (prime ≤ 46337). Gudhi default is 11 — set explicitly here |

---

## Supported Models

Any model supported by TransformerLens ≥ 3 works out of the box. Confirmed:

| Model | Architecture | GQA |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | Qwen2 | ✓ (2 KV heads, 8 Q heads) |
| `gpt2` | GPT-2 | — |
| `EleutherAI/pythia-70m` | GPT-NeoX | — |
| `meta-llama/Llama-3.2-1B` | Llama 3 | ✓ |
| `google/gemma-2-2b` | Gemma 2 | ✓ |

For gated models (Llama, Gemma) set `HUGGING_FACE_HUB_TOKEN` in your environment.

---

## Research Context

### The Gap This Fills

The existing literature on transformer attention uses:
- **Statistical physics**: mean-field / particle-system dynamics (Rigollet 2024)
- **Convex geometry**: softmax output constrained to probability simplex
- **Graph theory**: sparse attention patterns as adjacency matrices
- **Simplicial complexes on data domains**: 2-simplicial transformers (Clift et al. 2019)

What does **not** exist (as of June 2026):
- Stanley–Reisner ideals applied to attention *constraint structure*
- Algebraic characterisation of *which* co-attention patterns are forbidden
- Hochster's formula connecting attention topology to syzygy structure
- Primary decomposition as a formal language for "maximal co-attending groups"

### Key Open Questions

1. **Expressiveness via Betti numbers**: Do the graded Betti numbers of I(Δ) at each layer predict downstream task performance?
2. **Attention sink characterisation**: Does attention sink formation correspond to a specific prime component of I(Δ) becoming principal?
3. **Layer-wise algebraic flow**: How does I(Δ) evolve from layer 0 to layer L? Does it factor through a sequence of ideal inclusions?
4. **Cross-model comparison**: Do models fine-tuned for reasoning show structurally different SR ideals than base models?
5. **Unifying sparse attention variants**: Can local, strided, and dilated attention be classified by the minimal generating sets of their SR ideals?

---

## Running Tests

```bash
pytest tests/ -v
```

Tests in `tests/test_sr_core.py` require only `gudhi` and `numpy` — no model download.

---

## Citation

If you use this framework in research, please cite:

```bibtex
@software{sr_attention_2026,
  title  = {Stanley–Reisner Ideals for Transformer Attention Analysis},
  year   = {2026},
  url    = {https://git.hodgederrick.com/hodge360/stanley-reisner-attention},
  note   = {Novel framework bridging algebraic combinatorics and mechanistic interpretability}
}
```

---

## License

MIT