"""
    FactorLoadings(; factor = value, ...)

Named exposure loadings for a single policyholder, mapping user-defined risk
factor symbols to loadings in [0, 1].

The norm constraint `Σβ² ≤ 1` is validated at construction; it is required
for the factor copula simulation to be well-defined.

    FactorLoadings(aws = 0.4, crowdstrike = 0.3)
    FactorLoadings(ransomware = 0.6, financial_sector = 0.5)
"""

struct FactorLoadings
    loadings :: Dict{Symbol, Float64}
end

function FactorLoadings(; kwargs...)
    d = Dict{Symbol, Float64}(k => Float64(v) for (k, v) in kwargs)
    ns = sum(v^2 for v in values(d); init = 0.0)
    ns <= 1.0 || throw(ArgumentError(
        "factor loadings violate norm constraint: ‖β‖² = $ns > 1"
    ))
    return FactorLoadings(d)
end

Base.getindex(fl::FactorLoadings, k::Symbol) = get(fl.loadings, k, 0.0)
Base.keys(fl::FactorLoadings)                = keys(fl.loadings)
Base.isempty(fl::FactorLoadings)             = isempty(fl.loadings)

norm_sq(fl::FactorLoadings) = sum(v^2 for v in values(fl.loadings); init = 0.0)

idiosyncratic_weight(fl::FactorLoadings) = sqrt(1.0 - norm_sq(fl))

# ---- PortfolioLoadings -------------------------------------------------------

"""
    PortfolioLoadings()

Collection of `FactorLoadings` indexed by organisation name, parallel to a
`Portfolio`. Kept separate so loadings can be revised without rerunning
single-organisation simulations.

    add!(pl, :acme, FactorLoadings(aws = 0.4, ransomware = 0.3))
"""

struct PortfolioLoadings
    loadings :: Dict{Symbol, FactorLoadings}
end

PortfolioLoadings() = PortfolioLoadings(Dict{Symbol, FactorLoadings}())

function add!(pl::PortfolioLoadings, name::Symbol, fl::FactorLoadings)
    pl.loadings[name] = fl
    return pl
end

Base.getindex(pl::PortfolioLoadings, name::Symbol) = pl.loadings[name]
Base.haskey(pl::PortfolioLoadings, name::Symbol)   = haskey(pl.loadings, name)
Base.keys(pl::PortfolioLoadings)                   = keys(pl.loadings)
Base.length(pl::PortfolioLoadings)                 = length(pl.loadings)
