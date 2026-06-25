"""
    FAIRNode(frequency, magnitude; name = :unnamed, factor = nothing)

One risk scenario in the FAIR hierarchy. Combines a `FrequencyModel` and a
`MagnitudeModel` to produce realisations of annual aggregate loss:

    S = ∑_{i=1}^{N} L_i,   N ~ Poisson(λV)

where N is drawn from the frequency model and each L_i is drawn independently
from the magnitude model.

`name` labels the node within its model; `factor` tags it to a systemic risk
factor (e.g. `:aws`, `:ransomware`). The scenario engine activates only nodes
whose `factor` matches the scenario; nodes with `factor = nothing` are
idiosyncratic and always run at baseline.
"""

struct FAIRNode{F<:FrequencyModel, M<:MagnitudeModel}
    name      :: Symbol
    factor    :: Union{Symbol, Nothing}
    frequency :: F
    magnitude :: M
end

function FAIRNode(
    frequency :: FrequencyModel,
    magnitude :: MagnitudeModel;
    name   :: Symbol                 = :unnamed,
    factor :: Union{Symbol, Nothing} = nothing,
)
    return FAIRNode(name, factor, frequency, magnitude)
end

Distributions.partype(node::FAIRNode) = partype(node.magnitude)

# ---- Sampling ----------------------------------------------------------------

function rand_annual_loss(rng::AbstractRNG, node::FAIRNode)
    n = rand_count(rng, node.frequency)
    return sum(_ -> rand_loss(rng, node.magnitude), 1:n; init=zero(partype(node)))
end

rand_annual_loss(node::FAIRNode) = rand_annual_loss(Random.default_rng(), node)

# ---- Moments -----------------------------------------------------------------

# E[S] = E[N] × E[L]  (Wald's identity; N and L_i independent)
mean_annual_loss(node::FAIRNode) = mean_rate(node.frequency) * mean_loss(node.magnitude)
