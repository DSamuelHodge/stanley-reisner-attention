# Research Note: The Hochster Isomorphism Spectrum and Combinatorial Phase Transitions in Attention Graphs

## 1. The Mathematical Object

Let Δ be a finite simplicial complex on vertex set V = {1, …, n}. For
each W ⊆ V write Δ_W for the induced subcomplex on W.

**Pushforward measure.** Fix 0 ≤ j ≤ n and let Ωⱼ = {W ⊆ V : |W| = j}.
Let SimpCompⱼ be the set of simplicial complexes on at most j labelled
vertices.  Define the quotient:

 Typesⱼ ≔ SimpCompⱼ / ≅

and the map:

 φⱼ : Ωⱼ → Typesⱼ, φⱼ(W) = [Δ_W] (the isomorphism class)

Uniform(Ωⱼ) is the uniform probability measure on Ωⱼ.  The **Hochster
pushforward** is:

 μⱼ ≔ (φⱼ)₊ Uniform(Ωⱼ) ⟹ μⱼ(C) = |{W : Δ_W ≅ C}| / C(n, j)

This is the **empirical distribution of induced subcomplex isomorphism
types at size j** — a probability measure on the finite set Typesⱼ.

**Hochster Isomorphism Spectrum (corrected).**  Refining Hochster's formula

 β_{i,j} = Σ_{|W|=j} dim H̃_{j-i-1}(Δ_W)

define a **decorated probability measure**:

 ℋ_{i,j}(Δ) ≔ (μⱼ, C ↦ dim H̃_{j-i-1}(C))

i.e. the measure μⱼ on Typesⱼ together with a homology-decoration
functor from Typesⱼ to ℕ.  The classical Betti number is recovered as:

 β_{i,j} = C(n, j) · 𝔼_{C∼μⱼ}[dim H̃_{j-i-1}(C)]

This formulation keeps the invariant categorical (a measure + a
functor) rather than an ad-hoc set.

**Entropy profile.**  The Shannon entropy of the type distribution:

 Hⱼ(Δ) ≔ H(μⱼ) = − Σ_{C ∈ Typesⱼ} μⱼ(C) log₂ μⱼ(C)

Hⱼ depends **only** on the pushforward measure μⱼ, not on the
Betti/Hochster structure.  In general, Hⱼ is strictly finer than the
Betti table: two complexes with identical Betti numbers can have
different entropy profiles (because different type distributions can
produce the same aggregate homology sum).

**Phase classification (construction).**  Let H_max = max_k Hₖ and fix
ε ∈ (0, 1).  Define:

 j₁ = min{j ≥ 2 : Hⱼ ≥ ε·H_max}
 j₂ = max{j ≥ 2 : Hⱼ ≥ ε·H_max}

The **combinatorial phases** are a functional construction, not a
theorem:

| Regime | j | Behaviour |
|--------|---|-----------|
| Rigid | j < j₁ | ≤ 2 types dominate; Hⱼ ≈ 0 |
| Diverse | j₁ ≤ j ≤ j₂ | Many types; Hⱼ at maximum |
| Collapse | j > j₂ | Types contract; Hⱼ → 0 |

Empirically this three-regime structure appears across all attention
graphs tested, but it is a **descriptive classification**, not a
proved property of general complexes.

---

## 2. Results That Survive Scrutiny

### 2.1 Elementary Support-Size Bound

Hⱼ(Δ) ≤ log₂ |Typesⱼ|

This is immediate from the definition: maximum entropy of a
distribution on a finite set of size N is log₂ N.  No Betti numbers
appear.

A tighter bound exists if one knows the number of homologically active
types:

 Hⱼ(Δ) ≤ log₂( |Supp(μⱼ)| ) where |Supp(μⱼ)| ≤ C(n, j)

The claimed inequality Hⱼ ≤ log₂(1 + Σ_i β_{i,j}) from the earlier
draft is **false in general**: two distinct isomorphism types can have
identical homology, collapsing multiple types into the same Betti
contribution.  There is no injective map from types to Betti summands,
so the Betti sum does not bound the support size.

**Correction.**  The correct relationship between entropy and Betti
numbers is the expectation identity:

 β_{i,j} = C(n, j) · 𝔼_{C∼μⱼ}[dim H̃_{j-i-1}(C)]

but this does **not** imply an inequality for Hⱼ.

### 2.2 Stability Under Edge Perturbations

Let Δ, Δ' be two simplicial complexes on the same vertex set V that
differ by at most k edge insertions/deletions.

**Theorem (Stability).**  For each j ≥ 2:

 |Hⱼ(Δ) − Hⱼ(Δ')| ≤ k · C(n−2, j−2) / C(n, j) · log₂(|Typesⱼ|)

For fixed j and n → ∞, the right-hand side scales as O(k · j²/n²).

*Proof.*  One edge affects the induced subcomplex type for exactly
C(n−2, j−2) subsets W (those containing both endpoints).  Hence the
total variation distance between μⱼ and μⱼ' satisfies:

 d_TV(μⱼ, μⱼ') ≤ k · C(n−2, j−2) / C(n, j)

Entropy is Lipschitz in total variation with constant
log₂(|Typesⱼ| − 1) ≤ log₂(|Typesⱼ|).  The stated bound follows.  The
asymptotic scaling for fixed j uses C(n−2, j−2)/C(n, j) ∼ j²/n².  □

The constant log₂(|Typesⱼ|) is finite (Typesⱼ is a finite set) and
grows roughly like the number of unlabelled graphs on j vertices,
i.e. ∼ 2^{C(j,2)} / j! for the 1-skeleton case.

This is the **only inequality in the note that is both non-trivial and
rigorously correct**.  It makes the entropy profile a Lipschitz
function of the graph in edit distance, which is essential for any
statistical use.

### 2.3 Phase Decomposition Is a Valid Construction

The assignment Δ ↦ (j₁, j₂, {Hⱼ}) is a well-defined functional
construction.  It is:

* **invariant** under relabelling of vertices (since μⱼ is)
* **computable** for n ≤ HOCHSTER_MAX_VERTICES (currently 12)
* **empirically stable** — runs on GPT-2 and Qwen heads produce
  consistent three-regime profiles

It is **not** a theorem about the existence of phase transitions in
general complexes.  It is an algorithm for classifying scales.
Whether the three-regime structure holds for broad classes of
complexes is an open empirical question.

---

## 3. Structural Corrections to Earlier Claims

### 3.1 Categorical Status (corrected)

The map Δ ⟼ μⱼ is **not** a functor to the category of Markov
kernels.  The issue is that the quotient by isomorphism (Typesⱼ) is
not natural under inclusions: an inclusion ι : Δ → Δ' induces a map
on labelled complexes but does not descend to a well-defined
deterministic map on isomorphism classes, because two non-isomorphic
induced subcomplexes of Δ can become isomorphic when embedded in
Δ' (if the extra faces of Δ' erase the distinction).

The correct target is:

 FinSimp —→ Prob(𝒢ⱼ)

where 𝒢ⱼ is the **groupoid** of simplicial complexes on at most j
vertices with isomorphisms as morphisms.  A functor to Prob(𝒢ⱼ) sends
each Δ to a probability measure on the isomorphism classes
(Obj(𝒢ⱼ)/≅), which is precisely μⱼ.  This is conceptually clean but
requires groupoid-valued measure theory to make precise — a
non-trivial categorical overhead that is beyond the scope of this
note.

### 3.2 Graphon Convergence (corrected)

The claim "μⱼ^{(G_n)} converges weakly to μⱼ^{(W)}" is **not
currently supported**.  Graphon convergence gives convergence of
**induced subgraph densities**:

 t(C, G_n) → t(C, W) for every fixed graph C

but this is not the same as convergence of the full measure μⱼ (which
lives on Typesⱼ, a space that grows in size with j).  For fixed j,
the space Typesⱼ is fixed, and each type C has density t(C, G_n).
Graphon convergence implies:

 lim_{n→∞} μⱼ^{(G_n)}(C) = t(C, W) / Σ_{C' ∈ Typesⱼ} t(C', W)

i.e. the **normalised type frequencies** converge.  This is a
consequence of the Graphon Sampling Lemma (Lovász, 2012, §10) and is
**already a theorem**, not a conjecture.  However, the entropy
profile Hⱼ(G_n) then converges to Hⱼ(W) because entropy is continuous
on the finite space Typesⱼ.  So this direction is actually solid once
stated correctly — the earlier error was overcomplicating a standard
fact.

**Corrected claim.**  For a graph sequence G_n → W in cut distance,
for each fixed j:

 Hⱼ(G_n) → Hⱼ(W)

where Hⱼ(W) is the entropy of the j-vertex induced subgraph
distribution of W.  This follows directly from the Graphon Sampling
Lemma and continuity of entropy on a finite domain.

### 3.3 Representation Stability (corrected)

The claim that m_C = μⱼ(C)·C(n, j) is an "S_n-orbit size" is
**incorrect**.  The symmetric group S_n acts on Ωⱼ, and the S_n-orbit
of a subset W is the set of all j-subsets of V.  This orbit is
transitive — there is only one orbit.  The multiplicity m_C counts
how many j-subsets induce a given type, which is a **sampling
statistic**, not a group-orbit size.

The correct statement involves the **type frequency** as n grows:

 lim_{n→∞} μⱼ^{(G_n)}(C) = t(C, W) / Σ t(C', W)

which is a law-of-large-numbers result, not a representation-stability
result.  The connection to representation theory (if one exists) would
involve the **FI-module** structure of type multiplicities as n
varies, but this requires a consistent embedding of V_n into V_{n+1}
and is future work.

---

## 4. Application to Attention Heads (Heuristic)

For attention graphs (thresholded attention matrices), the filtration
parameter τ creates a 1-parameter family Δ(τ).  The phase boundaries
j₁(τ), j₂(τ) trace curves in (j, τ) space.  The area between these
curves — the "diverse region" — is a candidate quantitative measure
of **attentional capacity**.

This is an **interpretive heuristic**, not a theorem.  It is included
to motivate the construction but does not form part of the core
combinatorial claims.  A rigorous connection would require a model of
attention in which the pushforward measure is shown to distinguish
meaningful functional classes of heads with statistical significance.

---

## 5. Summary of Actual Status

| # | Statement | Status |
|---|-----------|--------|
| 1 | μⱼ is a well-defined pushforward measure on Typesⱼ | ✓ correct |
| 2 | ℋ_{i,j} as decorated measure refines Hochster's formula | ✓ correct |
| 3 | Hⱼ ≤ log₂ |Typesⱼ| | ✓ trivial bound |
| 4 | Hⱼ is strictly finer than the Betti table | ✓ correct |
| 5 | Stability: |Hⱼ(Δ) − Hⱼ(Δ')| ≤ k·C(n−2,j−2)/C(n,j)·log₂|Typesⱼ| | ✓ proved |
| 6 | Phase classification (j₁, j₂) | ✓ valid construction |
| 7 | Functor to Prob(𝒢ⱼ) | ⚠ correct target but heavy formalism |
| 8 | Graphon convergence of Hⱼ | ⚠ true via standard sampling lemma |
| 9 | Representation stability of type multiplicities | ✗ misidentified; replace with sampling statistics |
| 10 | Attention head typology | heuristic |
| 11 | Hⱼ ≤ log₂(1 + Σ β_{i,j}) | ✗ FALSE — withdrawn |
| 12 | Phase bounds via Betti numbers | ✗ FALSE — withdrawn (depended on #11) |

## 6. What a Real Paper Would Contain

A publishable combinatorics/TDA paper built on this work would have:

1. **Definition** of μⱼ and ℋ_{i,j} as a decorated measure (Sections 1–2 above).
2. **Stability theorem** (Section 2.2) — the one non-trivial inequality.
3. **Graphon limit** (Section 3.2 corrected) — convergence of Hⱼ for
   dense graph sequences, as a corollary of the Sampling Lemma.
4. **Empirical demonstration** on attention graphs (GPT-2, Qwen) showing
   the three-regime structure and its stability across prompts.
5. **Phase area** as a heuristic classifier for attention heads
   (explicitly marked as empirical, not derived).

Items that would be removed or deferred:

* The entropy–Betti inequality (false).
* The categorical claim in full generality (deferred to a separate
  categorical TDA paper).
* Representation stability (misidentified; replaced with sampling LLN).

The paper would be titled something like:

> *"The Hochster Pushforward Measure: A Combinatorial Invariant of
> Attention Graphs with a Stability Theorem"*
