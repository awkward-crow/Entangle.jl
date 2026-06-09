"""
    FrequencyModel(tef, vulnerability)

Poisson thinning model for annual loss event frequency.

`tef` is a distribution over the threat event frequency λ ≥ 0 (Poisson rate of
attack attempts). `vulnerability` is a distribution over V ∈ [0, 1] (probability
that any attempt succeeds). By the Poisson thinning theorem the resulting loss
event count is Poisson(λV).

Both parameters are uncertain; each simulated scenario draws fresh λ and V
realisations, so the marginal count distribution is overdispersed relative to a
fixed-rate Poisson.
"""

struct FrequencyModel{T<:UnivariateDistribution, U<:UnivariateDistribution}
    tef::T
    vulnerability::U
end

# ---- Sampling ----------------------------------------------------------------

rand_rate(rng::AbstractRNG, m::FrequencyModel) =
    rand(rng, m.tef) * rand(rng, m.vulnerability)

rand_rate(m::FrequencyModel) = rand_rate(Random.default_rng(), m)

function rand_count(rng::AbstractRNG, m::FrequencyModel)
    rand(rng, Poisson(rand_rate(rng, m)))
end

rand_count(m::FrequencyModel) = rand_count(Random.default_rng(), m)

# ---- Moments -----------------------------------------------------------------

# E[N] = E[λ] · E[V]  (independence of tef and vulnerability)
mean_rate(m::FrequencyModel) = mean(m.tef) * mean(m.vulnerability)
