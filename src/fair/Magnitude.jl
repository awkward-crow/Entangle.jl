"""
    MagnitudeModel(primary, secondary)

FAIR loss magnitude model: total loss is the sum of independent primary and
secondary loss components.

    L = L_primary + L_secondary

`primary` and `secondary` are distributions over non-negative losses (e.g.
`SplicedSeverity`). Each simulated event draws independent realisations from
both components.
"""
struct MagnitudeModel{P<:ContinuousUnivariateDistribution,
                      S<:ContinuousUnivariateDistribution}
    primary::P
    secondary::S
end

# ---- Sampling ----------------------------------------------------------------

function rand_components(rng::AbstractRNG, m::MagnitudeModel)
    p = rand(rng, m.primary)
    s = rand(rng, m.secondary)
    return (p, s)
end

rand_components(m::MagnitudeModel) = rand_components(Random.default_rng(), m)

function rand_loss(rng::AbstractRNG, m::MagnitudeModel)
    p, s = rand_components(rng, m)
    return p + s
end

rand_loss(m::MagnitudeModel) = rand_loss(Random.default_rng(), m)

# ---- Moments -----------------------------------------------------------------

# E[L] = E[L_primary] + E[L_secondary]  (linearity of expectation)
mean_loss(m::MagnitudeModel) = mean(m.primary) + mean(m.secondary)
