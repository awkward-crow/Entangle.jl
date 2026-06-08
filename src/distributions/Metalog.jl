"""
    Metalog(ps, qs; lower=-Inf)

Metalog distribution (Keelin 2016) fitted exactly to n quantile–probability
pairs.

- `ps`: probabilities ∈ (0, 1), strictly increasing
- `qs`: quantile values, strictly increasing
- `lower`: finite value produces a semi-bounded lower distribution with
  support [lower, ∞); default -Inf gives an unbounded distribution over ℝ

The n-term quantile function Q(p) = Σₖ aₖ mₖ(p) is solved from the linear
system M·a = qs (or M·a = log.(qs .- lower) for semi-bounded). After
fitting, Q'(p) > 0 is verified on a dense grid; an ArgumentError is thrown
if the inputs produce an invalid distribution.

Sampling is O(1) via the inverse-CDF method. CDF evaluation uses bisection.
"""
struct Metalog{T<:AbstractFloat} <: ContinuousUnivariateDistribution
    a::Vector{T}  # coefficients of the underlying unbounded metalog
    bl::T         # lower bound; -Inf means unbounded
end

# ---- Basis functions ---------------------------------------------------------
#
# Term k at probability p ∈ (0,1):
#   k = 1         → 1
#   k = 2         → logit(p)
#   k odd  ≥ 3   → (p−0.5)^((k−1)÷2) · logit(p)
#   k even ≥ 4   → (p−0.5)^(k÷2−1)

function _mbasis(k::Int, z::T, lgt::T) where {T<:AbstractFloat}
    k == 1 && return one(T)
    k == 2 && return lgt
    isodd(k) ? z^((k - 1) >> 1) * lgt : z^((k >> 1) - 1)
end

function _mbasis_deriv(k::Int, z::T, lgt::T, inv_pp::T) where {T<:AbstractFloat}
    k == 1 && return zero(T)
    k == 2 && return inv_pp
    if isodd(k)
        α = (k - 1) >> 1
        return T(α) * z^(α - 1) * lgt + z^α * inv_pp
    else
        β = (k >> 1) - 1
        return T(β) * z^(β - 1)
    end
end

function _metalog_quantile(a::Vector{T}, p::T) where {T<:AbstractFloat}
    z   = p - T(0.5)
    lgt = log(p / (one(T) - p))
    return sum(_mbasis(k, z, lgt) * a[k] for k in eachindex(a))
end

function _metalog_deriv(a::Vector{T}, p::T) where {T<:AbstractFloat}
    z      = p - T(0.5)
    lgt    = log(p / (one(T) - p))
    inv_pp = one(T) / (p * (one(T) - p))
    return sum(_mbasis_deriv(k, z, lgt, inv_pp) * a[k] for k in eachindex(a))
end

# ---- Fitting -----------------------------------------------------------------

function _metalog_fit(ps::AbstractVector, qs_target::AbstractVector)
    n = length(ps)
    T = promote_type(eltype(ps), eltype(qs_target))
    T = T <: AbstractFloat ? T : Float64
    M = Matrix{T}(undef, n, n)
    for (i, p) in enumerate(ps)
        p_T = T(p)
        z   = p_T - T(0.5)
        lgt = log(p_T / (one(T) - p_T))
        for k in 1:n
            M[i, k] = _mbasis(k, z, lgt)
        end
    end
    return M \ T.(qs_target)
end

# Grid density limitation: a non-monotone wiggle narrower than ~1/n_grid on the
# probability axis can pass undetected. Three paths to a stronger check:
#   1. Denser grid (e.g. 10_000) — cheap, still not a guarantee.
#   2. Find minima of Q'(p) by solving Q''(p) = 0 numerically and evaluating
#      Q' there — exact, but requires a root-finder for each fit.
#   3. Check at the input ps and a few Chebyshev nodes between them — wiggles
#      in practice appear near/between fitted points, so this catches the
#      common failure modes cheaply.
function _metalog_check_valid(a::Vector{T}) where {T<:AbstractFloat}
    n_grid = 1000
    for i in 1:n_grid
        p = T(i) / T(n_grid + 1)
        _metalog_deriv(a, p) > zero(T) && continue
        throw(ArgumentError(
            "fitted metalog has Q'(p) ≤ 0 near p=$(round(p, digits=4)); " *
            "check that quantile inputs are strictly increasing and consistent"
        ))
    end
end

# ---- Constructors ------------------------------------------------------------

function Metalog(ps::AbstractVector, qs::AbstractVector; lower::Real = -Inf)
    n = length(ps)
    length(qs) == n   || throw(ArgumentError("ps and qs must have the same length"))
    n >= 2            || throw(ArgumentError("at least 2 quantile–probability pairs required"))
    issorted(ps; lt = <) || throw(ArgumentError("ps must be strictly increasing"))
    issorted(qs; lt = <) || throw(ArgumentError("qs must be strictly increasing"))
    all(0 < p < 1 for p in ps) || throw(ArgumentError("all probabilities must be in (0, 1)"))

    semi = isfinite(lower)
    if semi
        all(q > lower for q in qs) ||
            throw(ArgumentError("all quantile values must exceed the lower bound ($lower)"))
        qs_fit = log.(qs .- lower)
    else
        qs_fit = qs
    end

    a = _metalog_fit(ps, qs_fit)
    _metalog_check_valid(a)

    T  = eltype(a)
    bl = semi ? T(lower) : T(-Inf)
    return Metalog{T}(a, bl)
end

# Convenience: Metalog(quantiles = [0.10 => 1.0, 0.50 => 3.0, 0.90 => 6.0])
function Metalog(; quantiles::AbstractVector{<:Pair}, lower::Real = -Inf)
    Metalog(first.(quantiles), last.(quantiles); lower = lower)
end

# ---- Accessors ---------------------------------------------------------------

Distributions.params(d::Metalog) = (d.a, d.bl)
Distributions.partype(d::Metalog{T}) where {T} = T

_is_semi(d::Metalog) = isfinite(d.bl)

# ---- Support -----------------------------------------------------------------

Base.minimum(d::Metalog{T}) where {T} = _is_semi(d) ? d.bl : T(-Inf)
Base.maximum(d::Metalog{T}) where {T} = T(Inf)

Distributions.insupport(d::Metalog, x::Real) =
    _is_semi(d) ? x >= d.bl : isfinite(x)

# ---- Quantile (closed form) --------------------------------------------------

function Distributions.quantile(d::Metalog{T}, p::Real) where {T}
    zero(p) <= p <= one(p) || throw(DomainError(p, "p must be in [0, 1]"))
    p == zero(p) && return minimum(d)
    p == one(p)  && return T(Inf)
    q = _metalog_quantile(d.a, T(p))
    return _is_semi(d) ? d.bl + exp(q) : q
end

# ---- CDF (bisection on the underlying unbounded metalog) --------------------

function _metalog_cdf_bisect(a::Vector{T}, target::T) where {T<:AbstractFloat}
    lo = T(1e-14)
    hi = T(1) - T(1e-14)
    for _ in 1:60
        mid = (lo + hi) / 2
        _metalog_quantile(a, mid) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function Distributions.cdf(d::Metalog{T}, x::Real) where {T}
    _is_semi(d) && x <= d.bl && return zero(T)
    target = _is_semi(d) ? log(T(x) - d.bl) : T(x)
    return _metalog_cdf_bisect(d.a, target)
end

# ---- Log-density -------------------------------------------------------------

function Distributions.logpdf(d::Metalog{T}, x::Real) where {T}
    insupport(d, x) || return T(-Inf)
    p  = cdf(d, x)
    dq = _metalog_deriv(d.a, T(p))
    dq > zero(T) || return T(-Inf)
    _is_semi(d) ? -log(T(x) - d.bl) - log(dq) : -log(dq)
end

# ---- Moments -----------------------------------------------------------------

# Mean = ∫₀¹ Q(p) dp via Gauss–Legendre-style trapezoidal rule.
function Distributions.mean(d::Metalog{T}) where {T}
    n_grid = 2000
    s = zero(T)
    for i in 1:(n_grid - 1)
        p = T(i) / T(n_grid)
        q = _metalog_quantile(d.a, p)
        s += _is_semi(d) ? d.bl + exp(q) : q
    end
    return s / T(n_grid - 1)
end

Distributions.median(d::Metalog) = quantile(d, oftype(float(d.bl), 0.5))

# ---- Sampling ----------------------------------------------------------------

Base.rand(rng::Random.AbstractRNG, d::Metalog) = quantile(d, rand(rng))
