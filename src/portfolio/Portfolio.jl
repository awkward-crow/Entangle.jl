"""
    Portfolio()

Mutable collection of single-organisation `FAIRModel`s and their pre-computed
sorted loss samples. Samples must be computed before portfolio simulation via
`calculate_losses!`.

    add!(portfolio, model)
    calculate_losses!(portfolio; n_scenarios, seed)
"""
mutable struct Portfolio
    models              :: Dict{Symbol, FAIRModel}
    sorted_loss_samples :: Dict{Symbol, Vector{Float64}}
end

Portfolio() = Portfolio(Dict{Symbol, FAIRModel}(), Dict{Symbol, Vector{Float64}}())

"""
    add!(portfolio, model::FAIRModel) -> portfolio

Register `model` in `portfolio`. Any previously cached loss samples for the
same name are invalidated.
"""
function add!(portfolio::Portfolio, model::FAIRModel)
    portfolio.models[model.name] = model
    delete!(portfolio.sorted_loss_samples, model.name)
    return portfolio
end

Base.length(portfolio::Portfolio)               = length(portfolio.models)
Base.haskey(portfolio::Portfolio, name::Symbol) = haskey(portfolio.models, name)
Base.names(portfolio::Portfolio)                = keys(portfolio.models)

has_loss_samples(portfolio::Portfolio, name::Symbol) =
    haskey(portfolio.sorted_loss_samples, name)

"""
    empirical_quantile(sorted_samples, u) -> Float64

Return the `u`-quantile of a pre-sorted sample vector by linear interpolation.
"""
function empirical_quantile(sorted_samples::Vector{Float64}, u::Float64)
    n  = length(sorted_samples)
    ix = clamp(u, 0.0, 1.0) * (n - 1) + 1.0
    lo = clamp(floor(Int, ix), 1, n)
    hi = min(lo + 1, n)
    t  = ix - lo
    return sorted_samples[lo] + t * (sorted_samples[hi] - sorted_samples[lo])
end

# ---- Loss sample calculation -------------------------------------------------

"""
    calculate_loss!(portfolio, name; n_scenarios, seed) -> portfolio

Simulate `n_scenarios` annual losses for organisation `name` and store the
sorted result. The effective RNG seed is `seed ⊻ hash(name)`, so the result
is stable regardless of portfolio composition.
"""
function calculate_loss!(
    portfolio   :: Portfolio,
    name        :: Symbol;
    n_scenarios :: Int,
    seed        :: Integer,
)
    haskey(portfolio.models, name) ||
        throw(ArgumentError("no model named :$name in portfolio"))
    org_seed = xor(UInt64(seed), hash(name))
    rng      = Xoshiro(org_seed)
    samples  = sort!([rand_annual_loss(rng, portfolio.models[name]) for _ in 1:n_scenarios])
    portfolio.sorted_loss_samples[name] = samples
    return portfolio
end

"""
    calculate_losses!(portfolio; n_scenarios, seed) -> portfolio

Compute sorted loss samples for every organisation in `portfolio`. Each
organisation's effective seed is `seed ⊻ hash(name)`.
"""
function calculate_losses!(portfolio::Portfolio; n_scenarios::Int, seed::Integer)
    org_names = collect(names(portfolio))
    results   = Vector{Vector{Float64}}(undef, length(org_names))
    Threads.@threads for i in eachindex(org_names)
        name       = org_names[i]
        org_seed   = xor(UInt64(seed), hash(name))
        rng        = Xoshiro(org_seed)
        results[i] = sort!([rand_annual_loss(rng, portfolio.models[name]) for _ in 1:n_scenarios])
    end
    for (i, name) in enumerate(org_names)
        portfolio.sorted_loss_samples[name] = results[i]
    end
    return portfolio
end

# ---- Portfolio simulation ----------------------------------------------------

"""
    rand_portfolio_losses(seed, portfolio, loadings, copula; n_scenarios) -> Vector{Float64}

Draw `n_scenarios` aggregate portfolio losses via the factor copula.

Organisation names are sorted before simulation so the RNG sequence is stable
regardless of insertion order. Organisations absent from `loadings` are treated
as purely idiosyncratic (zero factor exposures).

Each scenario uses an independent `Xoshiro` RNG seeded with `seed ⊻ s`, so
scenarios are computed in parallel and results are reproducible regardless of
thread count.
"""
function rand_portfolio_losses(
    seed      :: Integer,
    portfolio :: Portfolio,
    loadings  :: PortfolioLoadings,
    copula    :: FactorCopula;
    n_scenarios :: Int,
)
    org_names = sort!(collect(names(portfolio)))

    for name in org_names
        has_loss_samples(portfolio, name) ||
            throw(ArgumentError("no loss samples for :$name — call calculate_losses! first"))
    end

    factor_loadings = [haskey(loadings, n) ? loadings[n] : FactorLoadings() for n in org_names]
    β               = loading_matrix(factor_loadings)
    n_orgs          = length(org_names)
    losses          = Vector{Float64}(undef, n_scenarios)

    Threads.@threads for s in 1:n_scenarios
        rng       = Xoshiro(xor(UInt64(seed), UInt64(s)))
        U         = rand_uniforms(rng, copula, β)
        losses[s] = sum(empirical_quantile(portfolio.sorted_loss_samples[org_names[i]], U[i]) for i in 1:n_orgs)
    end

    return losses
end
