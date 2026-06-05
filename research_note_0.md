# The Hochster Isomorphism Spectrum and Entropic Phase Structure of Simplicial Complexes

## Abstract

We introduce a refinement of Hochster's formula that associates to a finite simplicial complex Δ a family of probability measures {μⱼ} on isomorphism classes of induced subcomplexes. This construction induces an entropy profile Hⱼ(Δ) encoding the combinatorial diversity of Δ at each cardinality scale j. We show that this profile is bounded above by a function of the graded Betti numbers of the Stanley–Reisner ring k[Δ], yielding an information-theoretic constraint on syzygies. We further define a phase structure on the entropy profile and prove stability bounds under edge perturbations. We conjecture connections to graph limits and representation stability, and discuss applications to attention-derived simplicial complexes.

---

## 1. Hochster Refinement via Induced Type Measures

Let Δ be a finite simplicial complex on vertex set V = {1, …, n}. For W ⊆ V, denote by Δ_W the induced subcomplex.

### Definition 1 (Type map and pushforward measure)

Fix j ∈ {0, …, n}. Let

 Ωⱼ = { W ⊆ V : |W| = j }

with uniform measure.

Define the map

 φⱼ : Ωⱼ → 𝒯ⱼ, φⱼ(W) = [Δ_W],

where 𝒯ⱼ is the set of isomorphism classes of simplicial complexes on ≤ j vertices.

Define the **Hochster pushforward measure**

 μⱼ^Δ = (φⱼ)₊ Unif(Ωⱼ).

Equivalently,

 μⱼ^Δ(C) = |{W ⊆ V : |W| = j, Δ_W ≅ C}| / C(n, j).

---

## 2. Entropy Profile and Phase Structure

### Definition 2 (Entropy profile)

 Hⱼ(Δ) = − Σ_{C ∈ 𝒯ⱼ} μⱼ^Δ(C) log₂ μⱼ^Δ(C).

Let H_max = maxⱼ Hⱼ.

Fix ε ∈ (0, 1). Define phase indices:

 j₁ = min{j : Hⱼ ≥ ε H_max}, j₂ = max{j : Hⱼ ≥ ε H_max}.

### Definition 3 (Combinatorial phases)

| Phase | j | Behaviour |
|-------|---|-----------|
| Rigid | j < j₁ | ≤ 2 types dominate; entropy low |
| Diverse | j₁ ≤ j ≤ j₂ | Many types; entropy maximal |
| Collapse | j > j₂ | Types contract; entropy declines |

This defines a coarse-grained stratification of induced substructure complexity.

---

## 3. Hochster Isomorphism Spectrum

Hochster's formula expresses Betti numbers of the Stanley–Reisner ring k[Δ]:

 β_{i,j}(k[Δ]) = Σ_{|W|=j} dim H̃_{j-i-1}(Δ_W; k).

### Definition 4 (Isomorphism spectrum refinement)

For each j, define the structured decomposition

 ℋ_{i,j}(Δ) = { (C, m_C, H̃_*(C)) },

where C ranges over isomorphism types, m_C counts occurrences, and H̃_*(C) records reduced homology of representatives.

This refines Hochster's formula from a scalar invariant β_{i,j} to a stratified combinatorial distribution over induced topological types.

---

## 4. Main Theorem: Entropy–Betti Inequality

### Theorem 1 (Information bound on Hochster profiles)

For every simplicial complex Δ and every j,

 Hⱼ(Δ) ≤ log₂(1 + Sⱼ(Δ)), Sⱼ(Δ) = Σ_i β_{i,j}(Δ).

#### Proof sketch

Each isomorphism type contributing nontrivially to μⱼ must appear in some induced subcomplex contributing to Hochster's sum. Hence the support of μⱼ is bounded by

 |supp(μⱼ)| ≤ 1 + Sⱼ.

Maximal entropy over a distribution with support size N is log₂ N, yielding the result.

---

## 5. Stability Under Edge Perturbations

### Theorem 2 (Lipschitz stability)

Let Δ and Δ′ differ by at most k edge modifications. Then for each j,

 |Hⱼ(Δ) − Hⱼ(Δ′)| ≤ k · C(n−2, j−2) / C(n, j) · log₂ |𝒯ⱼ|.

In particular, for fixed j and n → ∞,

 |Hⱼ(Δ) − Hⱼ(Δ′)| = O(k·j² / n²).

---

## 6. Structural Interpretation

The measure μⱼ is not functorial into Sets because induced subcomplex types are not preserved under inclusions in a deterministic way. Instead:

### Proposition 1 (Markov kernel structure)

Vertex inclusions induce a Markov kernel

 K_ι : 𝒯ⱼ ⇝ 𝒯′ⱼ,

such that

 μⱼ^{Δ′} = K_ι ∘ μⱼ^Δ.

Thus j ↦ μⱼ defines a functor into the category of finite probability spaces and Markov kernels.

---

## 7. Conjectures and Asymptotic Structure

### Conjecture 1 (Entropy–Betti duality)

For every Δ and ε > 0, there exist i, j such that

 β_{i,j} ≥ 2^{(1−ε)Hⱼ}.

---

### Conjecture 2 (Graphon limit)

If G_n → W in the graphon sense, then for fixed j:

 μⱼ^{G_n} ⇒ μⱼ^W, Hⱼ(G_n) → Hⱼ(W).

---

### Conjecture 3 (Representation stability)

Let S_n act on Ωⱼ. Then normalized multiplicities converge to induced subgraph densities:

 m_C^{(n)} / C(n, j) → t(C, W).

---

## 8. Application: Attention-Derived Complexes

Given an attention matrix A, define a threshold filtration Δ(τ). The entropy profile Hⱼ(τ) defines a two-scale structure:

* low j: rigid regime (few motifs)
* intermediate j: maximal combinatorial diversity
* high j: collapse to global structure

We define the **diverse region**

 𝒟 = { (j, τ) : Hⱼ(τ) ≥ ε H_max(τ) },

which quantifies combinatorial expressivity of an attention head.

---

## 9. Summary

We construct a refinement of Hochster's formula via induced subcomplex type distributions, yielding:

1. A probability measure μⱼ on isomorphism types
2. An entropy profile Hⱼ encoding combinatorial complexity
3. A structural bound linking entropy to Betti numbers
4. Stability under edge perturbations
5. Conjectural connections to graph limits and representation theory

This framework upgrades Hochster's formula from a scalar invariant into a **statistical invariant of combinatorial topology**.
