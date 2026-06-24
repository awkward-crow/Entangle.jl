using Entangle
using Distributions: Beta
using Statistics:    median
using Printf

n_orgs      = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
n_scenarios = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
n_reps      = 7

portfolio = Portfolio()
loadings  = PortfolioLoadings()

for i in 1:n_orgs
    name = Symbol("org_$i")
    add!(portfolio, FAIRModel(
        name          = name,
        tef           = Metalog(quantiles = [0.10 => 0.5, 0.50 => 1.5, 0.90 => 4.0], lower = 0),
        vulnerability = Beta(2.0, 8.0),
        primary_severity = SplicedSeverity(
            body      = Metalog(quantiles = [0.10 => 50_000, 0.50 => 200_000, 0.90 => 800_000], lower = 0),
            threshold = 1_000_000,
            gpd       = GPD(400_000, 0.5),
        ),
        secondary_severity    = Metalog(quantiles = [0.10 => 0.05, 0.50 => 0.30, 0.90 => 0.80], lower = 0),
        secondary_as_fraction = true,
    ))
    add!(loadings, name, FactorLoadings(aws = 0.4, ransomware = 0.3))
end

calculate_losses!(portfolio; n_scenarios = 50_000, seed = 42)

copula = StudentTFactorCopula(ν = 4)
seed   = 103357224

println("n_orgs=$n_orgs  n_scenarios=$n_scenarios  threads=$(Threads.nthreads())")

rand_portfolio_losses(seed, portfolio, loadings, copula; n_scenarios = n_scenarios)  # warmup

times = [@elapsed rand_portfolio_losses(seed + s, portfolio, loadings, copula; n_scenarios = n_scenarios) for s in 1:n_reps]
sort!(times)
@printf "  min=%.1fms  median=%.1fms\n" minimum(times) * 1000 median(times) * 1000
