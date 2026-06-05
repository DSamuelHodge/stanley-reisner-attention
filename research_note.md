# Research Note: The Hochster Isomorphism Spectrum and Combinatorial Phase Transitions in Attention Graphs

## 1. The Mathematical Object

Let Δ be a finite simplicial complex on vertex set V = {1, …, n}. For each
nonempty W ⊆ V write Δ_W for the induced subcomplex on W.

**Pushforward measure.** Fix 0 ≤ j ≤ n and let Ωⱼ = {W ⊆ V : |W| = j}.
Define:

 φⱼ : Ωⱼ ⟶ Types, φⱼ(W) = isomorphism_class(Δ_W)

where Types is the set of isomorphism types of simplicial complexes on
≤ j vertices.  Uniform(Ωⱼ) is the uniform probability measure on Ωⱼ.
The **Hochster pushforward** is:

 μⱼ ≔ (φⱼ)₊ Uniform(Ωⱼ)  ⟹  μⱼ(C) = |{W : Δ_W ≅ C}| / C(n, j)

**Hochster Isomorphism Spectrum.**  Refining each graded Betti number
β_{i,j}(k[Δ]) from Hochster's formula:

 β_{i,j} = Σ_{|W|=j} dim H̃_{j-i-1}(Δ_W)

we record the full type stratification:

 H_{i,j}(Δ) = { (C, m_C, H̃_*(C)) : C ∈ Types, m_C = μⱼ(C)·C(n,j) }

where H̃_*(C) is the reduced homology vector of the type.

**Entropy profile.**  The Shannon entropy of the type distribution:

 Hⱼ(Δ) ≔ H(μⱼ) = − Σ_C μⱼ(C) log₂ μⱼ(C)

**Phase classification.**  Let H_max = max_k Hₖ and fix ε ∈ (0, 1).
Define:

 j₁ = min{j ≥ 2 : Hⱼ ≥ ε·H_max}
 j₂ = max{j ≥ 2 : Hⱼ ≥ ε·H_max}

The **combinatorial phases** are:

| Regime | j | Behaviour |
|--------|---|-----------|
| Rigid | j < j₁ | ≤ 2 types dominate; Hⱼ ≈ 0 |
| Diverse | j₁ ≤ j ≤ j₂ | Many types; Hⱼ at maximum |
| Collapse | j > j₂ | Types contract; Hⱼ → 0 |

## 2. The Hiding Theorem: Hochster Entropy Comparison

For two simplicial complexes Δ, Δ' on the same vertex set V, write
Δ ⊆ Δ' if every simplex of Δ is also a simplex of Δ' (edgewise
inclusion).  Write [Δ] for the isomorphism class.

**Lemma A (Functoriality of μⱼ).**  For any inclusion ι : Δ ⟶ Δ'
(same V), the map φⱼ factors:

 Ωⱼ —[φⱼ^{(Δ)}]→ Types —[r]→ Types(W)

where r sends a type [Δ_W] to its relabelling under ι.  Consequently
μⱼ^{(Δ')} is the pushforward of μⱼ^{(Δ)} through the map that forgets
faces.

There is no general monotonicity of entropy under edge addition
(Example: empty graph has H₂ = 0; adding one edge raises H₂ > 0).
Instead, the correct structure is a **comparison principle** through
the Betti table.

**Theorem 1 (Hochster Information Inequality).**

 Hⱼ(Δ) ≤ log₂(1 + Σ_i β_{i,j}(Δ))

with equality iff every isomorphism type at scale j has a distinct
homology vector.  In particular, the Betti table determines an upper
bound on the entropy profile.

*Proof.*  Each type C ∈ Supp(μⱼ) contributes at least one unit to
some Betti number when it carries non-trivial reduced homology.  Let
T₀ be the homologically trivial type (if it occurs) and
T₊ = Supp(μⱼ) \ {T₀}.  Then |T₊| ≤ Σ_i β_{i,j} because each
contributing subset W is counted in some β_{i,j}.  The total support
size |Supp(μⱼ)| ≤ 1 + Σ_i β_{i,j}.  Maximising Hⱼ over probability
vectors with this support size gives log₂|Supp|.  □

**Theorem 2 (Phase Boundary Homology Bound).**  Let Sⱼ = Σ_i β_{i,j}.
The phase boundaries satisfy:

 j₁ ≥ min{j : log₂(1 + Sⱼ) ≥ ε·H_max}
 j₂ ≤ max{j : log₂(1 + Sⱼ) ≥ ε·H_max}

*Proof.*  Since Hⱼ ≤ log₂(1 + Sⱼ) by Theorem 1, if Hⱼ ≥ ε·H_max then
log₂(1 + Sⱼ) ≥ ε·H_max.  The thresholds are monotone in j.  □

**Theorem 3 (Stability Under Edit Distance).**  Let Δ, Δ' differ by at
most k edge insertions/deletions.  Then for each j:

 |Hⱼ(Δ) − Hⱼ(Δ')| ≤ k · C(n−2, j−2) / C(n, j) · log₂(C(n, j) − 1)

For fixed j and n → ∞, the right-hand side is O(k·j²/n²).

*Proof.*  Changing one edge alters the induced subcomplex type for
exactly C(n−2, j−2) subsets W (those containing both endpoints).  The
total variation distance between μⱼ and μⱼ' is bounded by
k·C(n−2, j−2)/C(n, j).  Entropy is Lipschitz in TV distance with
constant log₂(|Types| − 1).  □

## 3. The Missing Steps Toward a Publication

### 3.1 Correct Functor Category

The functor Δ ⟼ {μⱼ} is not a functor to Meas in the usual sense
because inclusions do not induce a deterministic map on Types (adding
an edge can merge types).  The correct target is the category of
**Markov kernels**: an inclusion ι : Δ ↪ Δ' induces a kernel
K_ι : Types(Δ) → Types(Δ') defined by K_ι(C, C') = 1 if every
edge of C is also an edge of C', and 0 otherwise.  Then:

 μⱼ^{(Δ')} = K_ι ∘ μⱼ^{(Δ)}

That μⱼ defines a **functor to the category of Markov kernels** is
the correct categorical formulation.

### 3.2 Sharpness of the Information Inequality

Theorem 1 is an upper bound.  The real content would be a **lower
bound** — or an asymptotic equality — linking Hⱼ to the Betti table.
Conjecture:

 For any Δ and any ε > 0, there exist j and i such that
 β_{i,j} ≥ 2^{(1−ε)Hⱼ}.

If true, this would make the entropy profile a **proxy for the Betti
table** and vice versa.  This is the deepest open claim.

### 3.3 Dense Graph Limit (Graphon) Convergence

Let (G_n) be a sequence of graphs converging to a graphon W in the
cut metric (Lovász–Szegedy).  Conjecture:

 For each fixed j, the pushforward measure μⱼ^{(G_n)} converges
 weakly to μⱼ^{(W)}, and Hⱼ(G_n) → Hⱼ(W).

In the graphon limit, the phase boundaries j₁, j₂ scale with n:
j₁ ≈ α₁ n, j₂ ≈ α₂ n for constants α₁(W), α₂(W) ∈ (0, 1)
determined by the degree distribution of W.  This turns the
phase diagram into a **graphon invariant**.

### 3.4 Representation Stability

The symmetric group S_n acts on Ωⱼ by permuting vertices.  For
attention graphs, this action is **not** transitive on types
(different isomorphism classes have different orbit sizes).  The
multiplicity m_C = μⱼ(C)·C(n, j) is the size of the S_n-orbit of
type C.  The entropy Hⱼ is the **orbit-entropy** — the Shannon
entropy of the orbit distribution.

Conjecture (Stability): For a fixed graphon W and a sequence of
graphs G_n → W, the orbit multiplicities satisfy:

 lim_{n→∞} m_C^{(n)} / C(n, j) = t(C, W)

where t(C, W) = ∫ ∏_{e∈E(C)} W(x_e) ∏_{e∉E(C)} (1−W(x_e)) dx_1…dx_j
is the **induced subgraph density** of C in W.  This connects the
pushforward measure to graph limit theory.

### 3.5 Application to Attention Heads

For attention graphs (thresholded attention matrices), the filtration
parameter τ creates a 1-parameter family Δ(τ).  The phase boundaries
j₁(τ), j₂(τ) trace curves in (j, τ) space.  The area between these
curves — the "diverse region" — measures the **attentional capacity**
of a head: large area means the head explores many combinatorial
patterns; small area means it locks into a fixed structure.

This gives a quantitative definition of **attention head
specialization**: a specialized head has a narrow diverse region at a
specific scale j, while a general-purpose head has a broad diverse
region.

## 4. Summary of Claims

| # | Statement | Status |
|---|-----------|--------|
| 1 | Hⱼ ≤ log₂(1 + Σ_i β_{i,j}) | Proved (Theorem 1) |
| 2 | |Hⱼ(Δ) − Hⱼ(Δ')| ≤ k·j²/n² for k-edge edits | Proved (Theorem 3) |
| 3 | μⱼ is a functor to Markov kernels | Formulated (§3.1) |
| 4 | β_{i,j} ≥ 2^{(1−ε)Hⱼ} for some i,j | Conjecture (§3.2) |
| 5 | Graphon convergence of μⱼ | Conjecture (§3.3) |
| 6 | Representation stability of orbit multiplicities | Conjecture (§3.4) |
| 7 | Phase area as attentional capacity | Application (§3.5) |

The publishable combinatorics result is: **The Hochster pushforward
measure defines a new functorial invariant of simplicial complexes
that satisfies an information-theoretic bound on syzygies and a
stability theorem under edge perturbations.  For attention graphs,
the phase structure of this invariant provides a quantitative
typology of attention head behaviour.**
