"""
    FAIRNode(frequency, magnitude)

One risk scenario in the FAIR hierarchy. Combines a `FrequencyModel` and a
`MagnitudeModel` to produce an annual loss via a compound Poisson process:

    S = ∑_{i=1}^{N} L_i,   N ~ Poisson(λV)

where N is drawn from the frequency model and each L_i is drawn independently
from the magnitude model.
"""
struct FAIRNode{F<:FrequencyModel, M<:MagnitudeModel}
    frequency::F
    magnitude::M
end

# ---- Sampling ----------------------------------------------------------------

function rand_annual_loss(rng::AbstractRNG, node::FAIRNode)
    n = rand_count(rng, node.frequency)
    return sum(_ -> rand_loss(rng, node.magnitude), 1:n; init=0.0)
end

rand_annual_loss(node::FAIRNode) = rand_annual_loss(Random.default_rng(), node)

# ---- Moments -----------------------------------------------------------------

# E[S] = E[N] × E[L]  (Wald's identity; N and L_i independent)
mean_annual_loss(node::FAIRNode) = mean_rate(node.frequency) * mean_loss(node.magnitude)
