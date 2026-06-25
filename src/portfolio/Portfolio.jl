"""
    Portfolio()

Mutable collection of single-organisation `FAIRModel`s and their pre-computed
sorted loss samples. Samples must be computed before portfolio simulation via
`calculate_marginal_losses!`.

    insert!(portfolio, model)
    calculate_marginal_losses!(portfolio; n_scenarios, seed)
"""
mutable struct Portfolio
    models              :: Dict{Symbol, FAIRModel}
    sorted_loss_samples :: Dict{Symbol, Vector{Float64}}
end

Portfolio() = Portfolio(Dict{Symbol, FAIRModel}(), Dict{Symbol, Vector{Float64}}())

"""
    insert!(portfolio, model::FAIRModel) -> portfolio

Register `model` in `portfolio`. Throws if a model with the same name is
already present; use `update!` to replace an existing model.
"""
function Base.insert!(portfolio::Portfolio, model::FAIRModel)
    haskey(portfolio.models, model.name) && throw(ArgumentError(
        "a model named :$(model.name) is already in the portfolio — use update! to replace it"
    ))
    portfolio.models[model.name] = model
    return portfolio
end

"""
    update!(portfolio, model::FAIRModel) -> FAIRModel

Replace an existing model in `portfolio`. Throws if no model with that name is
present; use `insert!` to add a new model. Clears any cached loss samples for
the replaced model. Returns the old model.
"""
function update!(portfolio::Portfolio, model::FAIRModel)
    haskey(portfolio.models, model.name) || throw(ArgumentError(
        "no model named :$(model.name) in portfolio — use insert! to add a new model"
    ))
    old = portfolio.models[model.name]
    portfolio.models[model.name] = model
    delete!(portfolio.sorted_loss_samples, model.name)
    return old
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
    calculate_marginal_loss!(portfolio, name; n_scenarios, seed) -> portfolio

Simulate `n_scenarios` annual losses for organisation `name` and store the
sorted result. The effective RNG seed is `seed ⊻ hash(name)`, so the result
is stable regardless of portfolio composition.
"""
function calculate_marginal_loss!(
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
    calculate_marginal_losses!(portfolio; n_scenarios, seed) -> portfolio

Compute sorted loss samples for every organisation in `portfolio`. Each
organisation's effective seed is `seed ⊻ hash(name)`.
"""
function calculate_marginal_losses!(portfolio::Portfolio; n_scenarios::Int, seed::Integer)
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
    rand_portfolio_loss(seed, loadings, copula, portfolio, n_samples) -> Vector{Float64}
    rand_portfolio_loss(seed, loadings, copula, portfolio)            -> Float64
    rand_portfolio_loss(loadings, copula, portfolio, n_samples)       -> Vector{Float64}
    rand_portfolio_loss(loadings, copula, portfolio)                  -> Float64
    rand_portfolio_loss(seed, portfolio, n_samples)                   -> Vector{Float64}
    rand_portfolio_loss(seed, portfolio)                              -> Float64
    rand_portfolio_loss(portfolio, n_samples)                         -> Vector{Float64}
    rand_portfolio_loss(portfolio)                                    -> Float64

Draw aggregate portfolio losses. With `n_samples` returns a vector; without
returns a single draw. The no-seed forms generate a seed from
`Random.default_rng()`.

Forms with `loadings` and `copula` correlate losses via the factor copula.
Forms without draw each organisation's loss independently from its marginal
distribution. Both forms use the pre-computed sorted loss samples, so the
marginal distributions are identical and the only difference is dependence
structure — the direct comparison for accumulation risk.

Organisation names are sorted for RNG stability regardless of insertion order.
Each sample uses an independent `Xoshiro` RNG seeded with `seed ⊻ s`, so
samples are computed in parallel and results are reproducible regardless of
thread count.
"""
function rand_portfolio_loss(
    seed      :: Integer,
    loadings  :: PortfolioLoadings,
    copula    :: FactorCopula,
    portfolio :: Portfolio,
    n_samples :: Int,
) :: Vector{Float64}
    org_names = sort!(collect(names(portfolio)))

    for name in org_names
        has_loss_samples(portfolio, name) ||
            throw(ArgumentError("no loss samples for :$name — call calculate_marginal_losses! first"))
    end

    factor_loadings = [haskey(loadings, n) ? loadings[n] : FactorLoadings() for n in org_names]
    β               = loading_matrix(factor_loadings)
    n_orgs          = length(org_names)
    out             = Vector{Float64}(undef, n_samples)

    Threads.@threads for s in 1:n_samples
        rng    = Xoshiro(xor(UInt64(seed), UInt64(s)))
        U      = rand_uniforms(rng, copula, β)
        out[s] = sum(empirical_quantile(portfolio.sorted_loss_samples[org_names[i]], U[i]) for i in 1:n_orgs)
    end

    return out
end

rand_portfolio_loss(
    seed      :: Integer,
    loadings  :: PortfolioLoadings,
    copula    :: FactorCopula,
    portfolio :: Portfolio,
) :: Float64 = rand_portfolio_loss(seed, loadings, copula, portfolio, 1)[1]

rand_portfolio_loss(loadings::PortfolioLoadings, copula::FactorCopula, portfolio::Portfolio, n_samples::Int) =
    rand_portfolio_loss(rand(Random.default_rng(), UInt64), loadings, copula, portfolio, n_samples)

rand_portfolio_loss(loadings::PortfolioLoadings, copula::FactorCopula, portfolio::Portfolio) =
    rand_portfolio_loss(rand(Random.default_rng(), UInt64), loadings, copula, portfolio)

function rand_portfolio_loss(
    seed      :: Integer,
    portfolio :: Portfolio,
    n_samples :: Int,
) :: Vector{Float64}
    org_names = sort!(collect(names(portfolio)))

    for name in org_names
        has_loss_samples(portfolio, name) ||
            throw(ArgumentError("no loss samples for :$name — call calculate_marginal_losses! first"))
    end

    n_orgs = length(org_names)
    out    = Vector{Float64}(undef, n_samples)

    Threads.@threads for s in 1:n_samples
        rng    = Xoshiro(xor(UInt64(seed), UInt64(s)))
        out[s] = sum(empirical_quantile(portfolio.sorted_loss_samples[org_names[i]], rand(rng)) for i in 1:n_orgs)
    end

    return out
end

rand_portfolio_loss(seed::Integer, portfolio::Portfolio) :: Float64 =
    rand_portfolio_loss(seed, portfolio, 1)[1]

rand_portfolio_loss(portfolio::Portfolio, n_samples::Int) =
    rand_portfolio_loss(rand(Random.default_rng(), UInt64), portfolio, n_samples)

rand_portfolio_loss(portfolio::Portfolio) =
    rand_portfolio_loss(rand(Random.default_rng(), UInt64), portfolio)
