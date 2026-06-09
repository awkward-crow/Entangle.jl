"""
    MagnitudeModel(primary, secondary; secondary_as_fraction=false)

FAIR loss magnitude model.

When `secondary_as_fraction = false` (default): primary and secondary losses
are sampled independently and summed.

    L = L_primary + L_secondary

When `secondary_as_fraction = true`: `secondary` is a distribution over
fractions ∈ [0, 1] and secondary loss scales with the primary realisation.

    L = L_primary · (1 + F),   F ~ secondary

The fractional form is appropriate when secondary costs (fines, litigation,
reputational damage) are elicited as a proportion of primary severity rather
than as an independent absolute amount.
"""

struct MagnitudeModel{P<:ContinuousUnivariateDistribution,
                      S<:ContinuousUnivariateDistribution}
    primary::P
    secondary::S
    secondary_as_fraction::Bool
end

MagnitudeModel(primary, secondary; secondary_as_fraction::Bool = false) =
    MagnitudeModel(primary, secondary, secondary_as_fraction)

# ---- Sampling ----------------------------------------------------------------

function rand_components(rng::AbstractRNG, m::MagnitudeModel)
    p = rand(rng, m.primary)
    s = m.secondary_as_fraction ? p * rand(rng, m.secondary) : rand(rng, m.secondary)
    return (p, s)
end

rand_components(m::MagnitudeModel) = rand_components(Random.default_rng(), m)

function rand_loss(rng::AbstractRNG, m::MagnitudeModel)
    p, s = rand_components(rng, m)
    return p + s
end

rand_loss(m::MagnitudeModel) = rand_loss(Random.default_rng(), m)

# ---- Moments -----------------------------------------------------------------

Distributions.partype(m::MagnitudeModel) = partype(m.primary)

function mean_loss(m::MagnitudeModel)
    # E[L] = E[primary] · (1 + E[fraction])  when secondary_as_fraction
    m.secondary_as_fraction && return mean(m.primary) * (1 + mean(m.secondary))
    return mean(m.primary) + mean(m.secondary)
end
