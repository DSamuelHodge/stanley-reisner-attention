"""
extraction.py
=============
TransformerLens ≥ 3 interface: load any supported model, run a prompt,
and return the full attention-pattern tensor from every layer and head.

Key design decisions
--------------------
* Uses ``run_with_cache`` — no manual hook registration needed.
* Returns a structured ``AttentionCache`` dataclass so downstream code
  never has to know the internal TL cache key format.
* GQA models (e.g. Qwen2.5-0.5B): TransformerLens automatically expands
  K/V heads with ``torch.repeat_interleave`` before computing hook_pattern,
  so the returned shape is always [batch, n_heads, seq, seq] regardless of
  the number of KV groups.
* Supports both the classic ``HookedTransformer.from_pretrained`` path and
  the newer ``TransformerBridge`` path introduced in TL3 (auto-detected).

Tensor shape convention (throughout the whole package)
------------------------------------------------------
patterns : [n_layers, n_heads, seq_len, seq_len]  (batch dim stripped)
           axis 0 = layer index
           axis 1 = head index (query heads, GQA-expanded)
           axis 2 = query token position
           axis 3 = key   token position
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from sr_attention import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class AttentionCache:
    """
    Structured output of a single forward pass.

    Attributes
    ----------
    patterns : np.ndarray
        Shape [n_layers, n_heads, seq_len, seq_len].
        Post-softmax attention probabilities.
    tokens : list[str]
        String representation of each token (from model tokeniser).
    token_ids : np.ndarray
        Shape [seq_len].  Integer token ids.
    model_name : str
        HuggingFace model id that produced this cache.
    n_layers : int
    n_heads : int
    seq_len : int
    """
    patterns   : np.ndarray          # [n_layers, n_heads, seq, seq]
    tokens     : list[str]
    token_ids  : np.ndarray          # [seq]
    model_name : str
    n_layers   : int
    n_heads    : int
    seq_len    : int
    metadata   : dict = field(default_factory=dict)

    def head(self, layer: int, head: int) -> np.ndarray:
        """Return [seq, seq] attention matrix for one (layer, head) pair."""
        return self.patterns[layer, head]

    def layer(self, layer: int) -> np.ndarray:
        """Return [n_heads, seq, seq] for a full layer."""
        return self.patterns[layer]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    Load a TransformerLens-supported model and extract attention patterns.

    Parameters
    ----------
    model_name : str
        HuggingFace model id, e.g. ``"Qwen/Qwen2.5-0.5B"``.
    dtype : str
        Torch dtype string.  See ``config.DEFAULT_DTYPE``.
    device : str | None
        "cuda", "cpu", or None (auto-detect).
    fold_ln : bool
        Fold LayerNorm weights into adjacent matrices (standard practice).
    """

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL_NAME,
        dtype: str = config.DEFAULT_DTYPE,
        device: Optional[str] = None,
        fold_ln: bool = config.FOLD_LN,
    ) -> None:
        self.model_name = model_name
        self.dtype      = dtype
        self.fold_ln    = fold_ln
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model     = None   # lazy load

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load(self) -> "AttentionExtractor":
        """
        Explicitly load the model.  Called automatically on first ``run``.

        TransformerLens ≥ 3 introduced ``TransformerBridge`` for arbitrary HF
        architectures.  We try the classic ``HookedTransformer.from_pretrained``
        first (works for well-known models like GPT-2, Pythia, Qwen) and fall
        back gracefully if needed.
        """
        if self._model is not None:
            return self

        # Resolve torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float32)

        logger.info("Loading model %s (dtype=%s, device=%s)", self.model_name, self.dtype, self.device)

        try:
            from transformer_lens import HookedTransformer
            self._model = HookedTransformer.from_pretrained(
                self.model_name,
                dtype=torch_dtype,
                fold_ln=self.fold_ln,
                move_to_device=True,
            )
        except Exception as e:
            # TL3 TransformerBridge path for architectures not in classic TL
            logger.warning(
                "HookedTransformer.from_pretrained failed (%s); trying TransformerBridge.", e
            )
            try:
                from transformer_lens.model_bridge import TransformerBridge
                self._model = TransformerBridge.from_pretrained(
                    self.model_name,
                    dtype=torch_dtype,
                    move_to_device=True,
                )
                # Enable HookedTransformer-equivalent numerics (TL3 requirement)
                self._model.enable_compatibility_mode()
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load {self.model_name} via HookedTransformer "
                    f"or TransformerBridge.\nOriginal: {e}\nBridge: {e2}"
                ) from e2

        self._model.eval()
        logger.info("Model loaded. Layers=%d, Heads=%d",
                    self._model.cfg.n_layers, self._model.cfg.n_heads)
        return self

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pattern_key(layer: int) -> str:
        """Full cache key for attention pattern at a given layer."""
        return config.HOOK_PATTERN_FULL.format(layer=layer)

    # ------------------------------------------------------------------
    # Forward pass + cache extraction
    # ------------------------------------------------------------------

    def run(
        self,
        prompt: str,
        prepend_bos: bool = True,
    ) -> AttentionCache:
        """
        Run ``prompt`` through the model and extract all attention patterns.

        Parameters
        ----------
        prompt : str
            Raw text prompt.
        prepend_bos : bool
            Prepend the model's BOS token (standard TransformerLens practice).

        Returns
        -------
        AttentionCache
            Structured cache with patterns of shape
            [n_layers, n_heads, seq_len, seq_len].
        """
        if self._model is None:
            self.load()

        model = self._model
        n_layers = model.cfg.n_layers
        n_heads  = model.cfg.n_heads

        # --- Tokenise -------------------------------------------------
        tokens    = model.to_tokens(prompt, prepend_bos=prepend_bos)    # [1, seq]
        str_tokens = model.to_str_tokens(prompt, prepend_bos=prepend_bos)
        seq_len   = tokens.shape[1]

        # --- Forward pass with full cache ----------------------------
        # We only need hook_pattern keys to save memory.
        # names_filter accepts a callable: keep only pattern hooks.
        pattern_keys = {self._pattern_key(l) for l in range(n_layers)}

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name in pattern_keys,
                prepend_bos=False,   # already prepended above
            )

        # --- Assemble [n_layers, n_heads, seq, seq] numpy array ------
        pattern_list = []
        for layer in range(n_layers):
            key = self._pattern_key(layer)
            p = cache[key]   # [batch=1, n_heads, seq, seq]
            pattern_list.append(p[0].cpu().float().numpy())   # [n_heads, seq, seq]

        patterns = np.stack(pattern_list, axis=0)   # [n_layers, n_heads, seq, seq]

        return AttentionCache(
            patterns   = patterns,
            tokens     = list(str_tokens),
            token_ids  = tokens[0].cpu().numpy(),
            model_name = self.model_name,
            n_layers   = n_layers,
            n_heads    = n_heads,
            seq_len    = seq_len,
            metadata   = {
                "n_key_value_heads": getattr(model.cfg, "n_key_value_heads", n_heads),
                "d_model"          : model.cfg.d_model,
                "prepend_bos"      : prepend_bos,
            },
        )

    # ------------------------------------------------------------------
    # Convenience: model info
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        """Access the underlying HookedTransformerConfig (after loading)."""
        if self._model is None:
            self.load()
        return self._model.cfg

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"AttentionExtractor(model={self.model_name!r}, {loaded})"
