"""
scripts/run_analysis.py
=======================
End-to-end demonstration script.

Usage
-----
    python scripts/run_analysis.py \
        --model  Qwen/Qwen2.5-0.5B \
        --prompt "The mathematician proved the theorem using algebra." \
        --layer  0 \
        --head   0

This will:
1. Load the model via TransformerLens.
2. Run the prompt and extract all attention patterns.
3. Build the filtered simplicial complex for the chosen (layer, head).
4. Compute and print:
   - SR ideal generators (forbidden co-attention patterns)
   - Primary decomposition (prime components of the ideal)
   - Betti table via Hochster's formula
   - Facet persistence summary
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sr_attention import config
from sr_attention.extraction import AttentionExtractor
from sr_attention.complex_builder import AttentionComplexBuilder
from sr_attention.sr_core import (
    stanley_reisner_generators,
    primary_decomposition_from_facets,
    compute_persistent_f_vector,
    h_vector_and_hilbert,
)
from sr_attention.persistence import (
    compute_facet_persistence,
    betti_table_via_hochster,
    format_betti_table,
    summarise_persistence,
)
from sr_attention.utils import (
    clean_token_labels,
    print_attention_head_summary,
    print_sr_summary,
    head_attention_stats,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_analysis")


def parse_args():
    p = argparse.ArgumentParser(description="SR Attention Analysis")
    p.add_argument("--model",  default=config.DEFAULT_MODEL_NAME)
    p.add_argument("--prompt", default="The cat sat on the mat.")
    p.add_argument("--layer",  type=int, default=0)
    p.add_argument("--head",   type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Attention threshold τ for static SR ideal extraction.")
    p.add_argument("--max-tokens", type=int, default=config.MAX_TOKENS_FOR_ANALYSIS,
                   help="Max tokens to use for algebraic analysis (Hochster is O(2^n)).")
    p.add_argument("--all-heads", action="store_true",
                   help="Print SR generator counts for all heads in the chosen layer.")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Extract attention patterns
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print(f"Model : {args.model}")
    print(f"Prompt: {args.prompt!r}")
    print("="*60)

    extractor = AttentionExtractor(model_name=args.model)
    cache     = extractor.run(args.prompt)

    tokens = clean_token_labels(cache.tokens)
    print(f"\nTokens ({cache.seq_len}): {tokens}")
    print(f"Layers: {cache.n_layers},  Heads per layer: {cache.n_heads}")
    print(f"GQA KV heads: {cache.metadata.get('n_key_value_heads', 'N/A')}")

    # ------------------------------------------------------------------ #
    # 2. Build complex for chosen (layer, head)
    # ------------------------------------------------------------------ #
    attn = cache.head(args.layer, args.head)   # [seq, seq]
    stats = head_attention_stats(attn)
    print(f"\n[Layer {args.layer}, Head {args.head}] attention stats: {stats}")

    print_attention_head_summary(attn, cache.tokens)

    builder = AttentionComplexBuilder(
        max_tokens=args.max_tokens,
        min_threshold=args.threshold,
    )

    # Full filtered complex (for persistence)
    st_filtered = builder.build(attn, token_labels=tokens)
    print(f"\nFiltered complex: {st_filtered.num_simplices()} simplices, "
          f"{st_filtered.num_vertices()} vertices")

    # Static complex at the chosen threshold τ (for SR ideal)
    st_static = builder.build_at_threshold(attn, threshold=args.threshold)

    # ------------------------------------------------------------------ #
    # 3. f-vector and h-vector at threshold τ
    # ------------------------------------------------------------------ #
    t_val = 1.0 - args.threshold   # filtration value corresponding to τ
    fvec  = compute_persistent_f_vector(st_filtered, t=t_val)
    h, hilbert = h_vector_and_hilbert(st_filtered, t=t_val)
    print(f"\nf-vector at τ={args.threshold}: {fvec}")
    print(f"h-vector at τ={args.threshold}: {h}")
    print(f"Hilbert series: {hilbert}")

    # ------------------------------------------------------------------ #
    # 4. Stanley–Reisner ideal
    # ------------------------------------------------------------------ #
    gens, mons = stanley_reisner_generators(
        st_static,
        token_labels=tokens[:args.max_tokens],
        verbose=True,
    )
    print_sr_summary(gens, mons, cache.tokens, threshold=args.threshold)

    # ------------------------------------------------------------------ #
    # 5. Primary decomposition
    # ------------------------------------------------------------------ #
    comps = primary_decomposition_from_facets(
        st_static,
        token_labels=tokens[:args.max_tokens],
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # 6. Facet persistence (across all thresholds)
    # ------------------------------------------------------------------ #
    intervals = compute_facet_persistence(st_filtered)
    summary   = summarise_persistence(intervals)
    print(f"\nFacet persistence summary: {summary}")

    # ------------------------------------------------------------------ #
    # 7. Betti table via Hochster's formula
    # ------------------------------------------------------------------ #
    n_for_hochster = min(st_static.num_vertices(), args.max_tokens, config.HOCHSTER_MAX_VERTICES)
    print(f"\nBetti table (Hochster, n={n_for_hochster}):")
    try:
        betti = betti_table_via_hochster(st_static, max_vertices=n_for_hochster)
        print(format_betti_table(betti))
    except ValueError as e:
        print(f"  Skipped: {e}")

    # ------------------------------------------------------------------ #
    # 8. Optional: all-heads summary
    # ------------------------------------------------------------------ #
    if args.all_heads:
        print(f"\nSR generator counts — Layer {args.layer}, all heads:")
        layer_patterns = cache.layer(args.layer)
        trees = builder.build_all_heads(layer_patterns, token_labels=tokens)
        for h_idx, st_h in enumerate(trees):
            g, _ = stanley_reisner_generators(st_h)
            print(f"  Head {h_idx:2d}: {len(g):3d} generators")


if __name__ == "__main__":
    main()
