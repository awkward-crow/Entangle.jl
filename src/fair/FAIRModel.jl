"""
    FAIRModel(name, nodes)
    FAIRModel(name, node)

Single-organisation FAIR model: a named collection of `FAIRNode`s whose
annual aggregate losses are summed to give a total annual aggregate loss:

    S_org = ∑_k S_k,   S_k ~ compound Cox process (node k)

It is assumed that the unit of time is a year.
"""

struct FAIRModel
    name::Symbol
    nodes::Vector{FAIRNode}
end

FAIRModel(name::Symbol, node::FAIRNode) = FAIRModel(name, [node])

# Convenience: single-node model from raw parameters
function FAIRModel(;
    name::Symbol = :org,
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

Distributions.partype(model::FAIRModel) =
    isempty(model.nodes) ? Float64 : partype(first(model.nodes))

# ---- Sampling ----------------------------------------------------------------

function rand_annual_loss(rng::AbstractRNG, model::FAIRModel)
    return sum(node -> rand_annual_loss(rng, node), model.nodes; init=zero(partype(model)))
end

rand_annual_loss(model::FAIRModel) = rand_annual_loss(Random.default_rng(), model)

# ---- Moments -----------------------------------------------------------------

mean_annual_loss(model::FAIRModel) =
    sum(mean_annual_loss, model.nodes; init=zero(partype(model)))

