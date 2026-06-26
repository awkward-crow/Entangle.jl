"""
    Exposures(; scenario = value, ...)

Bernoulli hit probabilities for a single organisation across named scenarios.
All values must be in [0, 1]; validated at construction.

    Exposures(aws_outage = 0.7, ransomware = 0.4)
"""
struct Exposures
    values :: Dict{Symbol, Float64}
end

function Exposures(; kwargs...)
    d = Dict{Symbol, Float64}(k => Float64(v) for (k, v) in kwargs)
    for (k, v) in d
        0.0 <= v <= 1.0 || throw(ArgumentError(
            "exposure :$k must be in [0, 1]; got $v"
        ))
    end
    return Exposures(d)
end

Base.getindex(e::Exposures, k::Symbol) = get(e.values, k, 0.0)
Base.isempty(e::Exposures)             = isempty(e.values)
Base.names(e::Exposures)               = keys(e.values)

# ---- PortfolioExposures -------------------------------------------------------

"""
    PortfolioExposures()

Per-organisation Bernoulli hit probabilities for named systemic scenarios,
indexed by organisation name and scenario symbol.

    se = PortfolioExposures()
    insert!(se, :acme,   aws = 0.7, ransomware = 0.4)
    insert!(se, :globex, aws = 0.3, crowdstrike = 0.6)

    losses = rand_scenario_losses(seed, :aws, se, portfolio; n_samples = 50_000)

Kept separate from `PortfolioLoadings`: factor loadings are copula parameters
with a joint norm constraint; scenario exposures are independent Bernoulli hit
probabilities with no coupling across scenarios.
"""
struct PortfolioExposures
    exposures :: Dict{Symbol, Dict{Symbol, Float64}}
end

PortfolioExposures() = PortfolioExposures(Dict{Symbol, Dict{Symbol, Float64}}())

function Base.getindex(se::PortfolioExposures, org::Symbol, scenario::Symbol)
    d = get(se.exposures, org, nothing)
    d === nothing ? 0.0 : get(d, scenario, 0.0)
end

# ---- Mutation ----------------------------------------------------------------

"""
    insert!(exposures, org, e::Exposures) -> exposures
    insert!(exposures, org; scenario = value, ...) -> exposures

Add new scenario exposures for `org`. Throws if any specified `org`/scenario
combination is already present; use `update!` to overwrite existing values.
Changes are all-or-nothing: no partial writes on error.
"""
function Base.insert!(se::PortfolioExposures, org::Symbol, e::Exposures)
    org_dict = get(se.exposures, org, nothing)
    for scenario in names(e)
        org_dict !== nothing && haskey(org_dict, scenario) && throw(ArgumentError(
            "exposure for :$org/:$scenario already set — use update! to overwrite"
        ))
    end
    d = get!(se.exposures, org, Dict{Symbol, Float64}())
    for scenario in names(e)
        d[scenario] = e[scenario]
    end
    return se
end

Base.insert!(se::PortfolioExposures, org::Symbol; kwargs...) =
    insert!(se, org, Exposures(; kwargs...))

"""
    update!(exposures, org, e::Exposures) -> Tuple{Symbol, Exposures}
    update!(exposures, org; scenario = value, ...) -> Tuple{Symbol, Exposures}

Overwrite existing scenario exposures for `org`. Throws if any specified
`org`/scenario combination has not been set; use `insert!` to add new entries.
Returns `(org, old_exposures)` so that the result can be splatted back:
`update!(se, update!(se, org, e)...)`. Changes are all-or-nothing: no partial
writes on error.
"""
function update!(se::PortfolioExposures, org::Symbol, e::Exposures)
    org_dict = get(se.exposures, org, nothing)
    for scenario in names(e)
        (org_dict === nothing || !haskey(org_dict, scenario)) && throw(ArgumentError(
            "no exposure for :$org/:$scenario — use insert! to add a new entry"
        ))
    end
    old = Dict{Symbol, Float64}()
    for scenario in names(e)
        old[scenario]      = org_dict[scenario]
        org_dict[scenario] = e[scenario]
    end
    return (org, Exposures(old))
end

update!(se::PortfolioExposures, org::Symbol; kwargs...) =
    update!(se, org, Exposures(; kwargs...))

# ---- Mode 2 simulation engine ------------------------------------------------

"""
    rand_scenario_losses(seed, scenario, exposures, portfolio; n_samples) -> Vector{Float64}

Draw `n_samples` aggregate portfolio losses under `scenario` (Mode 2).

For each sample, every organisation's nodes are partitioned:
- Nodes whose `factor` matches `scenario` fire with the Bernoulli hit
  probability `exposures[org, scenario]` (0 if absent).
- All other nodes (including idiosyncratic nodes with `factor = nothing`) fire
  at their baseline rates unconditionally.

Portfolio loss per sample is the sum across all organisations and nodes. Unlike
Mode 1 (`rand_portfolio_losses`), this operates at node level and does not
require pre-computed marginal samples. Organisation names are sorted for RNG
stability regardless of insertion order.
"""
function rand_scenario_losses(
    seed      :: Integer,
    scenario  :: Symbol,
    exposures :: PortfolioExposures,
    portfolio :: Portfolio;
    n_samples :: Int,
) :: Vector{Float64}
    org_names = sort!(collect(names(portfolio)))
    losses    = Vector{Float64}(undef, n_samples)

    Threads.@threads for s in 1:n_samples
        rng  = Xoshiro(xor(UInt64(seed), UInt64(s)))
        loss = 0.0

        for name in org_names
            model = portfolio.models[name]
            p_hit = exposures[name, scenario]

            for node in model.nodes
                if node.factor === scenario
                    rand(rng) < p_hit && (loss += rand_annual_loss(rng, node))
                else
                    loss += rand_annual_loss(rng, node)
                end
            end
        end

        losses[s] = loss
    end

    return losses
end
