# Entangle.jl

**Cyber risk quantification in Julia, built on the FAIR framework.**

Models the annual loss distribution for a single organisation as a compound Poisson process, with a spliced severity distribution combining expert-elicited body distributions and a Generalised Pareto tail. Outputs exceedance probability curves and PML estimates in standard catastrophe modelling format.

Portfolio-level simulation with copula-based correlation is planned but not yet implemented.

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

**Metrics** (`src/metrics/`)
- `ael` — Annual Expected Loss
- `pml` — Probable Maximum Loss at a given return period
- `exceedance_probability` — EP curve at standard return periods (5, 10, 20, 50, 100, 200, 250, 500, 1000 years)

---

## Example

```julia
using Entangle

org = FAIRModel(
    tef = Metalog(quantiles = [0.10 => 1.0, 0.50 => 3.0, 0.90 => 6.0], lower = 0),
    vulnerability = Beta(2.0, 8.0),
    primary_severity = SplicedSeverity(
        body      = Metalog(quantiles = [0.10 => 50_000, 0.50 => 200_000, 0.90 => 800_000]),
        threshold = 1_000_000,
        gpd       = GPD(400_000, 0.5)
    ),
    secondary_severity    = Metalog(quantiles = [0.10 => 0.05, 0.50 => 0.30, 0.90 => 0.80]),
    secondary_as_fraction = true,
)

losses = simulate(org, n_scenarios = 100_000, seed = 42)

println("AEL:      £$(round(ael(losses) / 1e3))k")
println("1-in-100: £$(round(pml(losses, 100) / 1e6, digits=1))m")
println("1-in-200: £$(round(pml(losses, 200) / 1e6, digits=1))m")

ep = exceedance_probability(losses)
```

---

## Dependencies

Julia 1.10 or later. Key packages: `Distributions.jl`, `Statistics` (stdlib).

---

## Status

- [x] Phase 1 — Distributions
- [x] Phase 2 — Single-organisation FAIR model and metrics
- [ ] Phase 3 — Portfolio simulation and copula
- [ ] Phase 4 — Scenario catalogue
- [ ] Phase 5 — Bayesian calibration

---

## Licence

MIT
