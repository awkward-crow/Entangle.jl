"""
    PERT(a, b, c[, λ=4])

PERT-Beta distribution with minimum `a`, mode `b`, maximum `c`, and shape
parameter `λ > 0` (default 4).

Parameterises a scaled Beta distribution on [a, c]. The equivalent Beta
parameters are:
    α₁ = 1 + λ(b − a)/(c − a),   α₂ = 1 + λ(c − b)/(c − a)

The mean is the classic PERT formula (a + λb + c)/(λ + 2), placing λ times
as much weight on the mode as on the endpoints. Standard FAIR tooling uses
λ = 4.
"""

struct PERT{T<:Real} <: ContinuousUnivariateDistribution
    a::T   # minimum
    b::T   # mode
    c::T   # maximum
    λ::T   # shape (default 4)
    α₁::T  # first Beta shape parameter
    α₂::T  # second Beta shape parameter

    function PERT{T}(a::T, b::T, c::T, λ::T) where {T<:Real}
        a < b || throw(ArgumentError("mode b must exceed minimum a, got a=$a b=$b"))
        b < c || throw(ArgumentError("maximum c must exceed mode b, got b=$b c=$c"))
        λ > zero(T) || throw(ArgumentError("shape λ must be positive, got λ=$λ"))
        r = c - a
        new{T}(a, b, c, λ, one(T) + λ * (b - a) / r, one(T) + λ * (c - b) / r)
    end
end

PERT(a::T, b::T, c::T, λ::T) where {T<:Real} = PERT{T}(a, b, c, λ)
PERT(a::Real, b::Real, c::Real, λ::Real) = PERT(promote(a, b, c, λ)...)
PERT(a::Integer, b::Integer, c::Integer, λ::Integer) = PERT(float(a), float(b), float(c), float(λ))
PERT(a::Real, b::Real, c::Real) = PERT(a, b, c, oftype(float(a), 4))

# ---- Accessors ---------------------------------------------------------------

Distributions.params(d::PERT) = (d.a, d.b, d.c, d.λ)
Distributions.partype(d::PERT{T}) where {T} = T

# ---- Support -----------------------------------------------------------------

Base.minimum(d::PERT) = d.a
Base.maximum(d::PERT) = d.c

Distributions.insupport(d::PERT, x::Real) = d.a <= x <= d.c

# ---- Log-density -------------------------------------------------------------

function Distributions.logpdf(d::PERT, x::Real)
    insupport(d, x) || return oftype(float(x), -Inf)
    u = (x - d.a) / (d.c - d.a)
    return logpdf(Beta(d.α₁, d.α₂), u) - log(d.c - d.a)
end

# ---- CDF ---------------------------------------------------------------------

function Distributions.cdf(d::PERT, x::Real)
    x <= d.a && return zero(float(x))
    x >= d.c && return one(float(x))
    return cdf(Beta(d.α₁, d.α₂), (x - d.a) / (d.c - d.a))
end

# ---- Quantile ----------------------------------------------------------------

function Distributions.quantile(d::PERT, p::Real)
    zero(p) <= p <= one(p) || throw(DomainError(p, "p must be in [0, 1]"))
    return d.a + (d.c - d.a) * quantile(Beta(d.α₁, d.α₂), p)
end

# ---- Moments -----------------------------------------------------------------

function Distributions.mean(d::PERT)
    return (d.a + d.λ * d.b + d.c) / (d.λ + 2)
end

function Distributions.var(d::PERT)
    μ = mean(d)
    return (μ - d.a) * (d.c - μ) / (d.λ + 3)
end

Distributions.std(d::PERT) = sqrt(var(d))

Distributions.mode(d::PERT) = d.b

Distributions.median(d::PERT) = quantile(d, oftype(float(d.a), 0.5))

# ---- Sampling ----------------------------------------------------------------

Base.rand(rng::Random.AbstractRNG, d::PERT) = quantile(d, rand(rng))
