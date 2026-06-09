"""
    GPD(σ, ξ)

Generalised Pareto Distribution with scale σ > 0 and shape ξ ∈ ℝ.

By the Pickands–Balkema–de Haan theorem, GPD is the limiting distribution of
scaled threshold exceedances for any distribution in the maximum domain of
attraction. The shape parameter controls tail behaviour:

- ξ > 0: heavy power-law tail; support [0, ∞)
- ξ = 0: exponential tail; equivalent to Exponential(1/σ); support [0, ∞)
- ξ < 0: bounded upper tail; support [0, -σ/ξ]
"""

struct GPD{T<:Real} <: ContinuousUnivariateDistribution
    σ::T  # scale
    ξ::T  # shape

    function GPD{T}(σ::T, ξ::T) where {T<:Real}
        σ > zero(T) || throw(ArgumentError("scale σ must be positive, got $σ"))
        new{T}(σ, ξ)
    end
end

GPD(σ::T, ξ::T) where {T<:Real} = GPD{T}(σ, ξ)
GPD(σ::Real, ξ::Real) = GPD(promote(σ, ξ)...)
GPD(σ::Integer, ξ::Integer) = GPD(float(σ), float(ξ))

# ---- Accessors ---------------------------------------------------------------

Distributions.params(d::GPD) = (d.σ, d.ξ)
Distributions.partype(d::GPD{T}) where {T} = T

# ---- Support -----------------------------------------------------------------

Base.minimum(d::GPD{T}) where {T} = zero(T)
Base.maximum(d::GPD{T}) where {T} = d.ξ >= zero(T) ? T(Inf) : -d.σ / d.ξ

Distributions.insupport(d::GPD, x::Real) =
    zero(x) <= x && (d.ξ >= 0 || x <= -d.σ / d.ξ)

# ---- Log-density -------------------------------------------------------------

function Distributions.logpdf(d::GPD, x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    σ, ξ = d.σ, d.ξ
    abs(ξ) < 1e-10 && return -log(σ) - x / σ
    return -log(σ) - (1/ξ + 1) * log1p(ξ * x / σ)
end

# ---- CDF and survival function -----------------------------------------------

function Distributions.cdf(d::GPD, x::Real)
    x <= minimum(d) && return zero(float(x))
    x >= maximum(d) && return one(float(x))
    σ, ξ = d.σ, d.ξ
    abs(ξ) < 1e-10 && return -expm1(-x / σ)
    return -expm1((-1/ξ) * log1p(ξ * x / σ))
end

function Distributions.ccdf(d::GPD, x::Real)
    x <= minimum(d) && return one(float(x))
    x >= maximum(d) && return zero(float(x))
    σ, ξ = d.σ, d.ξ
    abs(ξ) < 1e-10 && return exp(-x / σ)
    return exp((-1/ξ) * log1p(ξ * x / σ))
end

function Distributions.logccdf(d::GPD, x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    σ, ξ = d.σ, d.ξ
    abs(ξ) < 1e-10 && return -x / σ
    return (-1/ξ) * log1p(ξ * x / σ)
end

# ---- Quantile ----------------------------------------------------------------

function Distributions.quantile(d::GPD, p::Real)
    zero(p) <= p <= one(p) || throw(DomainError(p, "p must be in [0, 1]"))
    σ, ξ = d.σ, d.ξ
    abs(ξ) < 1e-10 && return -σ * log1p(-p)
    return σ / ξ * expm1(-ξ * log1p(-p))
end

# ---- Moments -----------------------------------------------------------------

function Distributions.mean(d::GPD{T}) where {T}
    d.ξ < one(T) || return T(Inf)
    return d.σ / (1 - d.ξ)
end

function Distributions.var(d::GPD{T}) where {T}
    d.ξ < T(0.5) || return T(Inf)
    σ, ξ = d.σ, d.ξ
    return σ^2 / ((1 - ξ)^2 * (1 - 2ξ))
end

Distributions.std(d::GPD) = sqrt(var(d))

Distributions.mode(d::GPD{T}) where {T} = zero(T)

Distributions.median(d::GPD) = quantile(d, oftype(float(d.σ), 0.5))

# ---- Sampling ----------------------------------------------------------------

Base.rand(rng::Random.AbstractRNG, d::GPD) = quantile(d, rand(rng))
