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
Base.keys(portfolio::Portfolio)                 = keys(portfolio.models)

has_loss_samples(portfolio::Portfolio, name::Symbol) =
    haskey(portfolio.sorted_loss_samples, name)

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
    for name in keys(portfolio.models)
        calculate_loss!(portfolio, name; n_scenarios = n_scenarios, seed = seed)
    end
    return portfolio
end
