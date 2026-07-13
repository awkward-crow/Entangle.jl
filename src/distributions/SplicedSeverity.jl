"""
    SplicedSeverity(body, threshold, tail; p_u)

Two-component severity distribution: a body distribution below `threshold`
and a GPD tail above it, joined at the splice point. It has density

    f(x) = (1 − p_u) · f_body(x) / F_body(threshold)   for x ≤ threshold
    f(x) =       p_u · f_GPD(x − threshold)              for x > threshold

where p_u = P(L > threshold) and F_body(threshold) = cdf(body, threshold)
is the normalisation factor for the (possibly truncated) body piece.

The default p_u = 1 − cdf(body, threshold) makes the body density appear
untruncated: when the body distribution assigns all its mass below the
threshold the factor cancels exactly, and when it does not the tail above
the threshold is simply replaced by the GPD without renormalising the body.
"""

struct SplicedSeverity{T<:AbstractFloat, B<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    body::B         # body distribution
    threshold::T    # splice point
    tail::GPD{T}    # GPD for exceedances; support [0, ∞)
    p_u::T          # P(L > threshold); weight on the tail component
    F_b::T          # cdf(body, threshold); normalisation for body piece
end

# ---- Constructors ------------------------------------------------------------

function SplicedSeverity(
    body::ContinuousUnivariateDistribution,
    threshold::Real,
    tail::GPD;
    p_u::Real = 1 - cdf(body, threshold),
)
    T  = promote_type(float(typeof(threshold)), partype(tail))
    F_b = T(cdf(body, threshold))

    F_b > zero(T) ||
        throw(ArgumentError("cdf(body, threshold) = 0; threshold lies below the body support"))
    zero(T) < T(p_u) < one(T) ||
        throw(ArgumentError("p_u must be in (0, 1), got $p_u"))

    return SplicedSeverity{T, typeof(body)}(
        body, T(threshold), GPD(T(tail.σ), T(tail.ξ)), T(p_u), F_b,
    )
end

# Convenience: SplicedSeverity(body=..., threshold=..., gpd=...)
SplicedSeverity(; body, threshold, gpd::GPD, p_u::Real = 1 - cdf(body, threshold)) =
    SplicedSeverity(body, threshold, gpd; p_u = p_u)

# ---- Accessors ---------------------------------------------------------------

Distributions.params(d::SplicedSeverity) = (d.body, d.threshold, d.tail, d.p_u)
Distributions.partype(d::SplicedSeverity{T}) where {T} = T

# ---- Support -----------------------------------------------------------------

Base.minimum(d::SplicedSeverity) = minimum(d.body)
Base.maximum(d::SplicedSeverity{T}) where {T} = T(Inf)

Distributions.insupport(d::SplicedSeverity, x::Real) = x >= minimum(d)

# ---- Log-density -------------------------------------------------------------

function Distributions.logpdf(d::SplicedSeverity{T}, x::Real) where {T}
    insupport(d, x) || return T(-Inf)
    if x <= d.threshold
        return log(1 - d.p_u) + logpdf(d.body, x) - log(d.F_b)
    else
        return log(d.p_u) + logpdf(d.tail, T(x) - d.threshold)
    end
end

# ---- CDF ---------------------------------------------------------------------

function Distributions.cdf(d::SplicedSeverity{T}, x::Real) where {T}
    T(x) < T(minimum(d)) && return zero(T)
    if T(x) <= d.threshold
        return (1 - d.p_u) * T(cdf(d.body, x)) / d.F_b
    else
        return (1 - d.p_u) + d.p_u * cdf(d.tail, T(x) - d.threshold)
    end
end

# ---- Quantile ----------------------------------------------------------------

function Distributions.quantile(d::SplicedSeverity{T}, q::Real) where {T}
    zero(q) <= q <= one(q) || throw(DomainError(q, "q must be in [0, 1]"))
    q == one(q) && return T(Inf)
    w = 1 - d.p_u   # body weight
    if T(q) <= w
        # invert (1 − p_u) · cdf(body, x) / F_b = q
        body_p = T(q) * d.F_b / w
        return T(quantile(d.body, body_p))
    else
        tail_p = (T(q) - w) / d.p_u
        return d.threshold + quantile(d.tail, tail_p)
    end
end

# ---- Moments -----------------------------------------------------------------

# Mean via quantile integration (no closed form in general).
function Distributions.mean(d::SplicedSeverity{T}) where {T}
    n = 2000
    s = zero(T)
    for i in 1:(n - 1)
        s += quantile(d, T(i) / T(n))
    end
    return s / T(n - 1)
end

Distributions.median(d::SplicedSeverity) = quantile(d, oftype(float(d.threshold), 0.5))

# ---- Sampling ----------------------------------------------------------------

Base.rand(rng::Random.AbstractRNG, d::SplicedSeverity) = quantile(d, rand(rng))
