"""
scripts/compare_pushforward.py
==============================
Cross-model comparison of the Hochster pushforward measure.

Compares the entropy profile μⱼ across:
  - multiple attention heads within a model
  - multiple models (GPT-2, Qwen, etc.)
  - synthetic baselines (random, complete, empty)

Usage
-----
    python scripts/compare_pushforward.py --prompt "The cat sat on the mat."
    python scripts/compare_pushforward.py --models gpt2,Qwen/Qwen2.5-0.5B --heads 0,2,5,7
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sr_attention.extraction import AttentionExtractor
from sr_attention.complex_builder import AttentionComplexBuilder
from sr_attention.sr_core import _all_vertices
from sr_attention.persistence import (
    betti_table_via_hochster,
    format_betti_table,
    hochster_pushforward_measure,
    format_pushforward_measure,
    entropy_profile,
    pushforward_divergence,
    build_random_complex,
    build_complete_complex,
    build_empty_complex,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("compare_pushforward")


def measure_model(model_name: str, prompt: str, layer: int, head: int):
    """Convenience: load model, build complex, compute pushforward measure."""
    extractor = AttentionExtractor(model_name)
    cache = extractor.run(prompt)
    attn = cache.head(layer=layer, head=head)
    builder = AttentionComplexBuilder()
    st = builder.build(attn, token_labels=cache.tokens)
    return hochster_pushforward_measure(st)


def print_comparison_table(results: dict):
    """Print a compact entropy-profile comparison table."""
    # Collect all j values
    all_js = set()
    for name, data in results.items():
        profile = dict(entropy_profile(data["measure"]))
        data["profile"] = profile
        all_js.update(profile.keys())

    js = sorted(all_js)
    names = list(results.keys())

    # Header
    header = "j\t" + "\t".join(f"{n[:12]}" for n in names)
    sep = "─" * len(header.expandtabs())
    print("\nEntropy profile H(μⱼ) (bits)")
    print(sep)
    print(header)
    print(sep)

    for j in js:
        row = [str(j)]
        for name in names:
            entropy = results[name]["profile"].get(j, float("nan"))
            row.append(f"{entropy:.4f}" if not np.isnan(entropy) else "  —  ")
        print("\t".join(row))
    print(sep)

    # Summary row: total entropy
    row = ["Σ"]
    for name in names:
        total = sum(results[name]["profile"].get(j, 0.0) for j in js)
        row.append(f"{total:.2f}")
    print("\t".join(row))


def main():
    parser = argparse.ArgumentParser(description="Cross-model pushforward comparison")
    parser.add_argument("--prompt", default="The mathematician proved the theorem using algebra.")
    parser.add_argument("--models", default="gpt2,Qwen/Qwen2.5-0.5B",
                        help="Comma-separated model names")
    parser.add_argument("--heads", default="0,5",
                        help="Comma-separated head indices (applied to all models)")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index (applied to all models)")
    parser.add_argument("--baselines", action="store_true", default=True,
                        help="Include synthetic baseline complexes")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--divergence", action="store_true",
                        help="Compute KL divergence matrix between all measures")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    heads = [int(h.strip()) for h in args.heads.split(",")]

    results = {}

    # ── Model heads ──────────────────────────────────────────────────
    for model_name in models:
        for head in heads:
            key = f"{model_name.split('/')[-1]}_L{args.layer}H{head}"
            logger.info(f"Measuring {key} …")
            try:
                measure = measure_model(model_name, args.prompt, args.layer, head)
                results[key] = {"measure": measure, "source": "model"}
            except Exception as e:
                logger.error(f"  Failed: {e}")

    # ── Baselines ────────────────────────────────────────────────────
    if args.baselines and results:
        # Use vertex count from the first successful model
        n = list(results.values())[0]["measure"]["n_vertices"]

        st_rand = build_random_complex(n, edge_probability=0.3, seed=args.random_seed)
        results["random(p=0.3)"] = {
            "measure": hochster_pushforward_measure(st_rand),
            "source": "baseline",
        }

        st_complete = build_complete_complex(n)
        results["complete"] = {
            "measure": hochster_pushforward_measure(st_complete),
            "source": "baseline",
        }

        st_empty = build_empty_complex(n)
        results["empty"] = {
            "measure": hochster_pushforward_measure(st_empty),
            "source": "baseline",
        }

    # ── Print results ────────────────────────────────────────────────
    print_comparison_table(results)

    # ── KL divergence matrix ─────────────────────────────────────────
    if args.divergence and len(results) > 1:
        names = list(results.keys())
        print(f"\n\nKL divergence matrix D_KL(μⱼ ‖ νⱼ), summed across j (bits)")
        print("─" * 70)

        # Header
        header = " " * 22 + "".join(f"{n[:18]:>18}" for n in names)
        print(header)

        for i, name_p in enumerate(names):
            row = [f"{name_p[:20]:>20}"]
            for name_q in names:
                if name_p == name_q:
                    row.append(f"{'—':>18}")
                else:
                    div = pushforward_divergence(
                        results[name_p]["measure"],
                        results[name_q]["measure"],
                    )
                    row.append(f"{div['total_div']:>18.4f}")
            print("".join(row))

    # ── Detailed per-model output ────────────────────────────────────
    print("\n\nDetailed per-model pushforward measures:")
    print("=" * 70)
    for name, data in results.items():
        print(f"\n── {name} ──")
        print(format_pushforward_measure(data["measure"]))


if __name__ == "__main__":
    main()
