# Numerical checks and experiments for the CLeaR 2027 paper

This folder has every script used to check the theory and to produce the numerical appendix (Appendix F) of
*Diagonally dominant confounding synthesis caps confounding strength in causal discovery benchmarks*.
Each entry gives the question, the setup, the command, the seed, the runtime and the result, so that you can find
and rerun any number in the paper.

Run everything from the repository root. The FCI experiments need `causal-learn` (in the project environment);
the others only need `numpy`, `scipy` and `matplotlib`:

```bash
uv run python examples/verify_clear2027/<script>.py [options]
```

## Notation

| Symbol | Meaning |
|---|---|
| `W` | weighted DAG among the observables, `W[u, v] != 0` means `v -> u`; data `X = (I - W)^{-1} eps` |
| `Omega` | noise covariance, `eps ~ N(0, Omega)`; off-diagonal nonzeros are the bidirected edges of the ADMG |
| `R~` | partial correlation matrix; `rho(R~)` its largest eigenvalue |
| `P` | normalized precision matrix; small `lambda_min(P)` means strong partial correlations |
| DD | diagonally dominant construction of the paper, `Omega_ii = 1 + sum_j |Omega_ij|` |
| root explicit (SDD) | explicit parameterization with root latents (lower blocks of the explicit weight matrix zero), one latent per bidirected edge with loadings `a, b`: `Omega = I + sum (a e_i + s b e_j)(a e_i + s b e_j)^T`. `a = b` gives DD; unequal loadings give scaled diagonally dominant matrices |
| PD | every positive definite `Omega` with the same pattern and `lambda_min(Omega) >= 1`; reachable by the explicit parameterization in general (converse of Prop. 3) |

DD ⊂ root explicit ⊂ PD.

**Fairness rule used everywhere since 2026-11-12:** within a comparison, hold the ground-truth ADMG fixed (the
DAG `W` among the observables and the bidirected pattern) and change only `Omega`. With `W` free, every spectrum is
reachable (Cholesky), so a comparison over random `W` says nothing about diagonal dominance. Fixing only the
top-left block `W_OO` of the explicit weight matrix is the same thing only when the lower blocks are zero; otherwise
the induced implicit `W` is `W_OO - Lambda S` (Prop. 3) and the ground truth changes.

## Theory checks (no figure in the paper)

| Script | What it checks | Result |
|---|---|---|
| `check.py` | the original lemmas behind Theorem 2 on small counterexamples (seeds 0, 1) | found the errors that were fixed in the paper (Appendix proofs) |
| `check2.py` | Theorem 2 with `W` free: a diagonal `Omega` and a Cholesky `W` reach `rho(R~) = 1 + (n-1)c` | the original statement fails without bounding `W`; motivates the fixed-`W` framing |
| `check3.py` | stress test of the bound `rho(R~) <= kappa(Omega) min(1 + d_in_max, ||I-W||^2)` (30000 random instances plus hill climbing, seed 7) | no violations |

## Appendix F.1: diagonal dominance caps confounding strength

* Script: `confounding_strength.py --trials 300 --n 9 --seed 0` (seconds). Output: `figs/confounding_strength.pdf`.
* Setup: one confounder with `|loadings|` in `[1, 2]` on `k` of 9 observables, unit noise. Both arms share `W`, the
  confounded set and the off-diagonal `Omega`; explicit `Omega = Lambda Lambda^T + I`, DD replaces the diagonal by `1 + row sum`.
* Result: `rho(R_eps) <= 2` under DD (Cor. 2), and the strongest confounder-induced correlation is capped; panel (c)
  (random `W`) shows little difference in `rho(R~)` for typical draws.

`fair_spectrum.py` (`--trials 500 --n 9 --seed 0`, output `fair_spectrum_output.txt`) is the same-`W` version of
the original F.1 table: medians differ by 2-10% only, and the explicit arm never exceeds the DD maximum.

## Appendix F.2: original spectrum comparison (kept for discussion)

The restored old figure. It is not fair: 4 vs about 9 variables, one fixed graph vs a new graph each time, different
edge weights, different confounding, and the implicit arm is unseeded. See the `claude:` note in the paper.

## Appendix F.3: FCI root vs intermediate hidden confounder

* Script: `examples/compare_root_vs_intermediate_grouped.py --macro-node-sizes 3 4 --replicates 100 --samples 500 --min-num-local-nodes 3 --max-num-local-nodes 5`
  (about 25 min, sequential). Output: `results.csv`, `grouped_paired_boxplot.pdf`.
* **Bug found 2026-11-12, fixed on branch `fix-shd-node-order`:** `causalspyne.ancestral_shd.structural_hamming_distance`
  compared `dag2pag(...).graph` with FCI's output directly, but `dag2pag` returns the nodes in a scrambled order. A
  perfect prediction scored a normalized SHD of 0.28-0.69. The old F.3 numbers (0.37-0.50) were computed with the bug.
  See `fci_root_vs_intermediate_rerun.txt` for the old and corrected numbers.

## Appendix F.4: partial correlation spectrum, sparse vs dense DD

* Script: `rtilde_sparse_vs_dense.py --trials 100 --s 2 --c 0.7 --c-dense 1.0 --seed 0` (about 1 min). Output: `figs/rtilde_sparse_vs_dense.pdf`.
* Result: sparse bounded `Omega`: `rho(R~)` about 1.9-2.4 for `|V|` from 10 to 200, below the bound
  `2(1+sc)(1+d_in_max)`. Dense `Omega`: grows as `(1+c|V|)/(1+c)` (93.8 at `|V| = 200`), so DD alone does not bound `rho(R~)`.

## Appendix F.5: fixed `W`, extremes of the spectrum in three classes

* Script: `fixedW_three_classes.py --graphs 24 --budget 1500` (seed 0, about 1 min). Output: `fixedW_three_classes_output.txt`.
* Setup: 24 random DAGs, `|V| = 10`, edge probability 0.3, `|W_ij|` in `[0.1, 0.5]` or `[0.5, 1.5]`; bidirected pairs
  with each variable in at most two; `|Omega_ij| <= c` with `c` in {0.7, 2}; `lambda_min(Omega) >= 1`. For each class,
  random starts plus coordinate hill climbing (1500 steps) for max `rho(R~)`, min `lambda_min(P)`, max mean squared
  partial correlation. Root-explicit loading ratio `a/b` in `[e^-2, e^2]`. PD reported as max(PD search, root search).
* Result (medians): DD has a 16-29% smaller max `rho(R~)`, an 11-25x larger min `lambda_min(P)` and a 31-56% smaller mean
  squared partial correlation than root explicit; almost the whole gap to PD is already there for root latents.
* `fixedW_extremes.py` is the earlier two-class version (DD vs PD with diagonal box `[0, 6c]`, 12 DAGs); it understated the gap.

## Appendix F.6: does the restriction change FCI's measured performance?

Common setup: `|V| = 10` observed, edge probability 0.3, bidirected edges only between non-adjacent variables, each
variable in at most two. Within a pair: same `W`, bidirected pattern, true PAG (one latent parent per bidirected edge,
via `dag2pag`, reordered by node name) and the same standard normal draws; only `Omega` differs. FCI from causal-learn,
Fisher-z, `alpha = 0.05`, standardized data. Metrics: normalized SHD to the true PAG, adjacency and arrowhead
precision/recall; paired Wilcoxon test. 8 settings: `|W|` range x `c` in {0.7, 2} x `N` in {500, 5000}. 16 processes, about 35 s each.

| Design | Script | Pairs/setting | Seeds | Arms |
|---|---|---|---|---|
| A matched | `fci_dd_vs_pd.py --reps 200` | 200 | `10000*s + r` | same off-diagonals (`|theta|` in `[c/4, c]`); DD diagonal vs smallest common shift with `lambda_min = 1` |
| B extremes | `fci_extremes.py 150` | 150 | `20000*s + r` | per class, the `Omega` minimizing `lambda_min(P)` (search of F.5, PD box `[0, 6c]`) |
| C rejection | `fci_rejection.py 150` | 150 | `30000*s + r` | random DD vs random PD accepted only if `rho(R~)` > 95th percentile of 2000 random DD draws |

Outputs: `fci_dd_vs_pd_output.txt`, `fci_extremes_output.txt`, `fci_rejection_output.txt`.

Result: in B and C the PD spectrum is clearly larger (`p < 1e-24`), but the normalized SHD changes by at most 0.016.
Significant shifts are small and mostly favour PD (B, `|W|` in `[0.1, 0.5]`, `c = 2`, `N = 5000`: arrowhead recall
0.449 -> 0.524, adjacency recall 0.607 -> 0.647). Design A is not informative (with the noise floor the two spectra
almost coincide). Random PD draws never exceeded the DD *maximum* of `rho(R~)` (0 of 20000 per DAG), so the
rejection threshold is the DD 95th percentile.

Note: FCI orientation in causal-learn is not fully deterministic, so reruns can differ in the third decimal.
