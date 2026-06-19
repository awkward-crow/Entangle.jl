# Entangle.jl

[![CI](https://github.com/awkward-crow/Entangle.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/awkward-crow/Entangle.jl/actions/workflows/ci.yml)

**Cyber risk quantification based on the FAIR framework.**

Models the annual loss distribution for a single organisation as a compound Poisson process with a spliced severity distribution combining expert-elicited body distributions and a Generalised Pareto tail. A Student-t factor copula aggregates correlated losses across a portfolio of organisations. Outputs are exceedance probability curves and PML estimates in standard catastrophe modelling format.

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
- `rand_portfolio_losses` — draws aggregate portfolio losses via the factor copula; organisations absent from loadings are treated as purely idiosyncratic

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

### Portfolio with correlated losses

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
    add!(portfolio, model)
    add!(loadings, name, FactorLoadings(aws = aws, ransomware = rs))
end

calculate_losses!(portfolio; n_scenarios = 100_000, seed = 42)

copula = StudentTFactorCopula(ν = 4)
n      = 100_000

# Correlated via Student-t factor copula
correlated = rand_portfolio_losses(Xoshiro(42), portfolio, loadings, copula; n_scenarios = n)

# Under independence: independent draws per organisation, summed
independent = sum(
    simulate(portfolio.models[name]; n_scenarios = n, seed = i)
    for (i, name) in enumerate(sort!(collect(names(portfolio))))
)

println("1-in-200 PML — correlated:  £$(round(pml(correlated,  200) / 1e6, digits = 1))m")
println("1-in-200 PML — independent: £$(round(pml(independent, 200) / 1e6, digits = 1))m")
# The gap between the two is the accumulation risk a naive independence
# assumption would miss — the defining exposure in cyber portfolio underwriting.
```

---

## Dependencies

Julia 1.10 or later. Key packages: `Distributions.jl`, `Statistics` (stdlib).

---

## Status

- [x] Phase 1 — Distributions
- [x] Phase 2 — Single-organisation FAIR model and metrics
- [x] Phase 3 — Portfolio simulation and copula
- [ ] Phase 4 — Scenario catalogue
- [ ] Phase 5 — Bayesian calibration

---

## Licence

MIT
