# Entangle.jl

[![CI](https://github.com/awkward-crow/Entangle.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/awkward-crow/Entangle.jl/actions/workflows/ci.yml)

**Cyber risk quantification based on the FAIR framework.**

Models the annual loss distribution for a single organisation as a compound Poisson process with a spliced severity distribution combining expert-elicited body distributions and a Generalised Pareto tail. A Student-t factor copula aggregates correlated losses across a portfolio of organisations. Outputs are exceedance probability curves and PML estimates in standard catastrophe modelling format.

---

## Motivation

The cyber insurance market has grown rapidly, but its quantitative foundations remain underdeveloped. Existing implementations of the FAIR framework are predominantly Excel-based tools running single-organisation Monte Carlo with no mechanism for capturing the correlated losses that arise when a common threat actor — malware propagating through a shared software dependency, a cloud provider outage, a coordinated campaign — affects many policyholders simultaneously.

This is the **cyber accumulation problem**: the defining challenge of cyber portfolio underwriting. Events like NotPetya (2017), WannaCry (2017), and the CrowdStrike outage (2024) demonstrated that a single incident can produce correlated losses across thousands of unrelated organisations. Without a portfolio-level model capturing these dependencies, an insurer cannot know its true accumulated exposure and cannot price or manage it.

Entangle.jl addresses this with four layers: (1) a single-organisation FAIR model extended with proper heavy-tailed severity distributions and named, factor-tagged loss nodes; (2) a factor model mapping policyholders to common cyber risk exposures; (3) a Student-t factor copula aggregating correlated tail losses across the portfolio; and (4) a scenario catalogue of named systemic events operating at node level.

The factor copula in layer (3) applies the framework of Oh & Patton (2017) — developed
in an econometric/systemic-risk context — to insurance accumulation risk. Cross-firm
correlation in cyber-insurance has been modelled with a t-copula before (Böhme &
Kataria, 2006), but with a single implied correlation rather than an explicit factor
structure; Entangle.jl extends this to a scalable, named-factor construction, so that
correlation is driven by shared exposures (a common cloud provider, a shared software
dependency) rather than a fitted correlation parameter.

The design uses two simulation modes. Mode 1 runs the factor copula at aggregate level — an acknowledged approximation (Böhme & Kataria, 2006) that is acceptable because in the tail a single node typically dominates the aggregate. Mode 2 directly activates the factor-tagged nodes corresponding to a named scenario, giving node-level precision for calibration and validation.

---

## What's implemented

**Distributions** (`src/distributions/`)
- `Metalog` — fitted to arbitrary quantile specifications (Keelin 2016); preferred for all expert-elicited inputs
- `PERT` — parameterised by (min, mode, max); included for compatibility with standard FAIR tooling
- `GPD` — Generalised Pareto Distribution for heavy-tailed severity modelling
- `SplicedSeverity` — body distribution below a threshold joined to a GPD tail above it

**Single-organisation FAIR model** (`src/fair/`)
- `FrequencyModel` — Poisson thinning: loss event count is Poisson(λV) where λ (TEF) and V (vulnerability) are both uncertain distributions
- `MagnitudeModel` — primary and secondary loss components; secondary can be specified as an independent distribution or as a fraction of primary
- `FAIRNode`, `FAIRModel` — compose frequency and magnitude into a compound Poisson annual loss model
- `simulate` — Monte Carlo engine returning a vector of annual loss draws

**Metrics** (`src/Metrics.jl`)
- `ael` — Annual Expected Loss
- `pml` — Probable Maximum Loss at a given return period
- `exceedance_probability` — EP curve at standard return periods (5, 10, 20, 50, 100, 200, 500, 1000 years)

**Portfolio simulation** (`src/portfolio/`)
- `FactorLoadings` — per-organisation exposure vector across named risk factors (e.g. `:aws`, `:crowdstrike`, `:ransomware`); validates the norm constraint required for copula simulation
- `PortfolioLoadings` — collection of `FactorLoadings` indexed by organisation name
- `loading_matrix` — constructs the `(n × K)` factor loading matrix from a vector of `FactorLoadings`
- `Portfolio` — mutable collection of `FAIRModel`s with pre-computed sorted loss samples
- `StudentTFactorCopula` — Student-t factor copula parameterised by degrees of freedom `ν`; lower `ν` produces stronger tail dependence
- `rand_portfolio_loss` — draws aggregate portfolio losses; with `loadings` and `copula` correlates losses via the factor copula, without draws each organisation independently from its marginal; returns a scalar with no `n_samples` argument, a vector otherwise

**Scenario catalogue** (`src/scenarios/`)
- `FAIRNode` — individual risk node within a `FAIRModel`; carries a `name` and an optional `factor` symbol that links it to a systemic scenario (e.g. `factor = :aws`)
- `PortfolioExposures` — per-organisation Bernoulli hit probabilities for named systemic scenarios; `insert!` adds new entries (throws on duplicate), `update!` overwrites existing ones (returns old values)
- `rand_scenario_losses` — Mode 2 engine: for each sample, nodes tagged with the active scenario fire with the organisation's hit probability; all other nodes fire at baseline; operates at node level without pre-computed marginal samples

---

## Examples

```julia
using Entangle

org = FAIRModel(
    tef = Metalog(quantiles = [0.10 => 1.0, 0.50 => 3.0, 0.90 => 6.0], lower = 0),
    vulnerability = Beta(2.0, 8.0),
    primary_severity = SplicedSeverity(
        body      = Metalog(quantiles = [0.10 => 50_000, 0.50 => 200_000, 0.90 => 800_000], lower = 0),
        threshold = 1_000_000,
        gpd       = GPD(400_000, 0.5)
    ),
    secondary_severity    = Metalog(quantiles = [0.10 => 0.05, 0.50 => 0.30, 0.90 => 0.80], lower = 0),
    secondary_as_fraction = true,
)

losses = simulate(org, n_scenarios = 100_000, seed = 42)

println("AEL:      £$(round(ael(losses) / 1e3))k")
println("1-in-100: £$(round(pml(losses, 100) / 1e6, digits=1))m")
println("1-in-200: £$(round(pml(losses, 200) / 1e6, digits=1))m")

ep = exceedance_probability(losses)
```

### Portfolio with correlated losses and tail dependence

```julia
using Entangle, Distributions, Random

portfolio = Portfolio()
loadings  = PortfolioLoadings()

for (name, aws, rs) in [(:acme, 0.4, 0.3), (:globex, 0.6, 0.0), (:initech, 0.3, 0.5)]
    model = FAIRModel(
        name               = name,
        tef                = Metalog(quantiles = [0.10 => 0.5, 0.50 => 1.5, 0.90 => 4.0], lower = 0),
        vulnerability      = Beta(2.0, 8.0),
        primary_severity   = SplicedSeverity(
            body      = Metalog(quantiles = [0.10 => 50_000, 0.50 => 200_000, 0.90 => 800_000], lower = 0),
            threshold = 1_000_000,
            gpd       = GPD(400_000, 0.5)
        ),
        secondary_severity    = Metalog(quantiles = [0.10 => 0.05, 0.50 => 0.30, 0.90 => 0.80], lower = 0),
        secondary_as_fraction = true,
    )
    insert!(portfolio, model)
    insert!(loadings, name, FactorLoadings(aws = aws, ransomware = rs))
end

calculate_marginal_losses!(portfolio; n_scenarios = 100_000, seed = 42)


seed = 103357224
n = 100_000

independent    = rand_portfolio_loss(seed, portfolio, n)
correlated     = rand_portfolio_loss(seed, loadings, GaussianFactorCopula(),       portfolio, n)
tail_dependent = rand_portfolio_loss(seed, loadings, StudentTFactorCopula(ν = 4),  portfolio, n)

println("1-in-200 PML — independent:    £$(round(pml(independent,    200) / 1e6, digits = 1))m")
println("1-in-200 PML — correlated:     £$(round(pml(correlated,     200) / 1e6, digits = 1))m")
println("1-in-200 PML — tail_dependent: £$(round(pml(tail_dependent, 200) / 1e6, digits = 1))m")
# 1-in-200 PML — independent:    £7.4m
# 1-in-200 PML — correlated:     £7.9m
# 1-in-200 PML — tail_dependent: £9.3m

# All three share the same marginals. The Gaussian and Student-t copulas use the
# same loadings, giving the same linear correlation. The remaining gap — correlated
# to tail_dependent — is the tail dependence contribution.
```

### Scenario losses (Mode 2)

```julia
using Entangle, Distributions

# Build a node tagged to the :aws factor — the scenario engine fires it
# with each org's hit probability when simulating an AWS outage scenario
aws_node(name) = FAIRNode(
    FrequencyModel(
        Metalog(quantiles = [0.10 => 0.2, 0.50 => 0.8, 0.90 => 2.0], lower = 0),
        Beta(2.0, 8.0),
    ),
    MagnitudeModel(
        SplicedSeverity(
            body      = Metalog(quantiles = [0.10 => 50_000, 0.50 => 200_000, 0.90 => 800_000], lower = 0),
            threshold = 1_000_000,
            gpd       = GPD(400_000, 0.5),
        ),
        Metalog(quantiles = [0.10 => 0.05, 0.50 => 0.30, 0.90 => 0.80], lower = 0);
        secondary_as_fraction = true,
    );
    name   = name,
    factor = :aws,
)

portfolio = Portfolio()
for name in (:acme, :globex, :initech)
    insert!(portfolio, FAIRModel(name, aws_node(name)))
end

exposures = PortfolioExposures()
insert!(exposures, :acme,    aws = 0.7)
insert!(exposures, :globex,  aws = 0.3)
insert!(exposures, :initech, aws = 0.5)

losses = rand_scenario_losses(42, :aws, exposures, portfolio; n_samples = 100_000)

println("Scenario AEL:      £$(round(ael(losses) / 1e3))k")
println("Scenario 1-in-100: £$(round(pml(losses, 100) / 1e6, digits = 1))m")
```

---

## Dependencies

Julia 1.10 or later. Key packages: `Distributions.jl`, `Statistics` (stdlib).

---

## Status

- [x] Phase 1 — Distributions
- [x] Phase 2 — Single-organisation FAIR model and metrics
- [x] Phase 3 — Portfolio simulation and copula
- [x] Phase 4 — Scenario catalogue
- [ ] Phase 5 — Bayesian calibration

---

## References

**FAIR framework**
- [FAIR Standard v3.0](https://www.fairinstitute.org/hubfs/Standards%20Artifacts/Factor%20Analysis%20of%20Information%20Risk%20(FAIR)%20Standard%20v3.0%20(January%202025).pdf) (January 2025), FAIR Institute

**Cyber accumulation**
- Böhme, R. and Kataria, G. (2006). Models and Measures for Correlation in Cyber-Insurance. Workshop on the Economics of Information Security (WEIS). https://weis2006.econinfosec.org/docs/16.pdf
- Oh, D. H. and Patton, A. J. (2017). Modeling Dependence in High Dimensions with Factor Copulas. Journal of Business & Economic Statistics, 35(1), 139–154. https://doi.org/10.1080/07350015.2015.1062384
- Embrechts, P., McNeil, A. and Straumann, D. (2002). Correlation and Dependence in Risk Management: Properties and Pitfalls. In Risk Management: Value at Risk and Beyond, ed. M.A.H. Dempster, Cambridge University Press, pp. 176–223. https://doi.org/10.1017/CBO9780511615337.008
- [CyRiM Bashe Attack: Global Infection by Contagious Malware](https://assets.lloyds.com/assets/pdf-bashe-attack-cyrimbasheattack-finalbashe-attack/1/pdf-bashe-attack-CyRiMBasheAttack_FINALbashe-attack.pdf) (2019)
- [Practical Management of Cyber Exposures and Aggregations](https://lmalloyds.com/wp-content/uploads/2025/07/LMA-EMWG-Cyber-Risk-Paper.pdf) (2025), LMA
- Zeller, G. & Scherer, M. (2022). A comprehensive model for cyber risk based on marked point processes and its application to insurance. *European Actuarial Journal*, 12(1), 33–85. https://doi.org/10.1007/s13385-021-00290-1
- Zeller, G. & Scherer, M. (2024). Is accumulation risk in cyber methodically underestimated? European Actuarial Journal, 14(3), 711–748. https://doi.org/10.1007/s13385-024-00381-9
- Carannante, M. & Mazzoccoli, A. (2025). "An Analytical Review of Cyber Risk Management by Insurance Companies: A Mathematical Perspective." Risks, 13(8), 144. https://doi.org/10.3390/risks13080144

**Extreme value theory and copulas**
- Coles, *An Introduction to Statistical Modeling of Extreme Values*, Springer, 2001
- McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton University Press

**Metalog distribution**
- Keelin (2016). The Metalog Distributions. *Decision Analysis*, 13(4), 243–277.

---

## Licence

MIT
