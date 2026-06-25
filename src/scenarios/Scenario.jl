"""
    ScenarioExposures()

Per-organisation Bernoulli hit probabilities for named systemic scenarios,
indexed by organisation name and scenario symbol.

    se = ScenarioExposures()
    insert!(se, :acme,   aws = 0.7, ransomware = 0.4)
    insert!(se, :globex, aws = 0.3, crowdstrike = 0.6)

    losses = rand_scenario_losses(seed, :aws, se, portfolio; n_samples = 50_000)

Kept separate from `PortfolioLoadings`: factor loadings are copula parameters
with a joint norm constraint; scenario exposures are independent Bernoulli hit
probabilities with no coupling across scenarios.
"""
struct ScenarioExposures
    exposures :: Dict{Symbol, Dict{Symbol, Float64}}
end

ScenarioExposures() = ScenarioExposures(Dict{Symbol, Dict{Symbol, Float64}}())

function Base.getindex(se::ScenarioExposures, org::Symbol, scenario::Symbol)
    d = get(se.exposures, org, nothing)
    d === nothing ? 0.0 : get(d, scenario, 0.0)
end

# ---- Mutation ----------------------------------------------------------------

"""
    insert!(exposures, org; scenario = value, ...) -> exposures

Add new scenario exposures for `org`. Throws if any specified `org`/scenario
combination is already present; use `update!` to overwrite existing values.
All values must be in [0, 1]. Changes are all-or-nothing: no partial writes
on error.
"""
function Base.insert!(se::ScenarioExposures, org::Symbol; kwargs...)
    org_dict = get(se.exposures, org, nothing)
    for (scenario, value) in kwargs
        0.0 <= Float64(value) <= 1.0 || throw(ArgumentError(
            "exposure for :$org/:$scenario must be in [0, 1]; got $value"
        ))
        org_dict !== nothing && haskey(org_dict, scenario) && throw(ArgumentError(
            "exposure for :$org/:$scenario already set — use update! to overwrite"
        ))
    end
    d = get!(se.exposures, org, Dict{Symbol, Float64}())
    for (scenario, value) in kwargs
        d[scenario] = Float64(value)
    end
    return se
end

"""
    update!(exposures, org; scenario = value, ...) -> Dict{Symbol, Float64}

Overwrite existing scenario exposures for `org`. Throws if any specified
`org`/scenario combination has not been set; use `insert!` to add new entries.
Returns the previous values. All new values must be in [0, 1]. Changes are
all-or-nothing: no partial writes on error.
"""
function update!(se::ScenarioExposures, org::Symbol; kwargs...)
    org_dict = get(se.exposures, org, nothing)
    for (scenario, value) in kwargs
        0.0 <= Float64(value) <= 1.0 || throw(ArgumentError(
            "exposure for :$org/:$scenario must be in [0, 1]; got $value"
        ))
        (org_dict === nothing || !haskey(org_dict, scenario)) && throw(ArgumentError(
            "no exposure for :$org/:$scenario — use insert! to add a new entry"
        ))
    end
    old = Dict{Symbol, Float64}()
    for (scenario, value) in kwargs
        old[scenario]      = org_dict[scenario]
        org_dict[scenario] = Float64(value)
    end
    return old
end

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
    exposures :: ScenarioExposures,
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
