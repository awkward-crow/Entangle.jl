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
- `rand_portfolio_loss` — draws aggregate portfolio losses; with `loadings` and `copula` correlates losses via the factor copula, without draws each organisation independently from its marginal; returns a scalar with no `n_samples` argument, a vector otherwise

**Scenario catalogue** (`src/scenarios/`)
- `FAIRNode` — individual risk node within a `FAIRModel`; carries a `name` and an optional `factor` symbol that links it to a systemic scenario (e.g. `factor = :aws`)
- `ScenarioExposures` — per-organisation Bernoulli hit probabilities for named systemic scenarios; `insert!` adds new entries (throws on duplicate), `update!` overwrites existing ones (returns old values)
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
    insert!(portfolio, model)
    insert!(loadings, name, FactorLoadings(aws = aws, ransomware = rs))
end

calculate_marginal_losses!(portfolio; n_scenarios = 100_000, seed = 42)


seed = 103357224
n = 100_000

independent = rand_portfolio_loss(seed, portfolio, n)

copula = StudentTFactorCopula(ν = 4)
correlated  = rand_portfolio_loss(seed, loadings, copula, portfolio, n)

println("1-in-200 PML — correlated:  £$(round(pml(correlated,  200) / 1e6, digits = 1))m")
println("1-in-200 PML — independent: £$(round(pml(independent, 200) / 1e6, digits = 1))m")
# The gap between the two is the accumulation risk a naive independence
# assumption would miss — the defining exposure in cyber portfolio underwriting.
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

exposures = ScenarioExposures()
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

## Licence

MIT
