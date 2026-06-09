"""
    FAIRModel(name, nodes)
    FAIRModel(name, node)

Single-organisation FAIR model: a named collection of `FAIRNode`s whose
annual aggregate losses are summed to give a total annual aggregate loss:

    S_org = ∑_k S_k,   S_k ~ compound Cox process (node k)
"""

struct FAIRModel
    name::String
    nodes::Vector{FAIRNode}
end

FAIRModel(name::String, node::FAIRNode) = FAIRModel(name, [node])

# Convenience: single-node model from raw parameters
function FAIRModel(;
    name::String = "org",
    tef,
    vulnerability,
    primary_severity,
    secondary_severity,
    secondary_as_fraction::Bool = false,
)
    freq = FrequencyModel(tef, vulnerability)
    mag  = MagnitudeModel(primary_severity, secondary_severity;
                          secondary_as_fraction = secondary_as_fraction)
    return FAIRModel(name, FAIRNode(freq, mag))
end

# ---- Sampling ----------------------------------------------------------------

function rand_annual_loss(rng::AbstractRNG, model::FAIRModel)
    return sum(node -> rand_annual_loss(rng, node), model.nodes; init=0.0)
end

rand_annual_loss(model::FAIRModel) = rand_annual_loss(Random.default_rng(), model)

# ---- Moments -----------------------------------------------------------------

mean_annual_loss(model::FAIRModel) =
    sum(mean_annual_loss, model.nodes; init=0.0)

# ---- simulate ----------------------------------------------------------------

"""
    simulate(model::FAIRModel; n_scenarios=10_000, seed=nothing) -> Vector{Float64}

Draw `n_scenarios` independent annual aggregate losses from `model`.
Pass an integer `seed` for reproducibility.
"""
function simulate(model::FAIRModel; n_scenarios::Int = 10_000, seed = nothing)
    rng = seed === nothing ? Random.default_rng() : Xoshiro(seed)
    losses = Vector{Float64}(undef, n_scenarios)
    for i in eachindex(losses)
        losses[i] = rand_annual_loss(rng, model)
    end
    return losses
end
