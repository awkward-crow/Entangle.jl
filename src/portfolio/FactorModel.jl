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
Base.isempty(fl::FactorLoadings)             = isempty(fl.loadings)
Base.names(fl::FactorLoadings)               = keys(fl.loadings)

norm_sq(fl::FactorLoadings) = sum(v^2 for v in values(fl.loadings); init = 0.0)

idiosyncratic_weight(fl::FactorLoadings) = sqrt(1.0 - norm_sq(fl))

# ---- PortfolioLoadings -------------------------------------------------------

"""
    PortfolioLoadings()

Collection of `FactorLoadings` indexed by organisation name, parallel to a
`Portfolio`. Kept separate so loadings can be revised without rerunning
single-organisation simulations.

    insert!(pl, :acme, FactorLoadings(aws = 0.4, ransomware = 0.3))
"""

struct PortfolioLoadings
    loadings :: Dict{Symbol, FactorLoadings}
end

PortfolioLoadings() = PortfolioLoadings(Dict{Symbol, FactorLoadings}())

"""
    insert!(pl, name, fl::FactorLoadings) -> pl

Add loadings for `name`. Throws if loadings for that name are already present;
use `update!` to replace existing loadings.
"""
function Base.insert!(pl::PortfolioLoadings, name::Symbol, fl::FactorLoadings)
    haskey(pl.loadings, name) && throw(ArgumentError(
        "loadings for :$name already present — use update! to replace them"
    ))
    pl.loadings[name] = fl
    return pl
end

"""
    update!(pl, name, fl::FactorLoadings) -> Tuple{Symbol, FactorLoadings}

Replace loadings for `name`. Throws if no loadings for that name are present;
use `insert!` to add a new entry. Returns `(name, old_fl)` so that the result
can be splatted back: `update!(pl, update!(pl, name, fl)...)`.
"""
function update!(pl::PortfolioLoadings, name::Symbol, fl::FactorLoadings)
    haskey(pl.loadings, name) || throw(ArgumentError(
        "no loadings for :$name — use insert! to add a new entry"
    ))
    old = pl.loadings[name]
    pl.loadings[name] = fl
    return (name, old)
end

Base.insert!(pl::PortfolioLoadings, name::Symbol; kwargs...) =
    insert!(pl, name, FactorLoadings(; kwargs...))

Base.getindex(pl::PortfolioLoadings, name::Symbol) = pl.loadings[name]
Base.haskey(pl::PortfolioLoadings, name::Symbol)   = haskey(pl.loadings, name)
Base.keys(pl::PortfolioLoadings)                   = keys(pl.loadings)
Base.length(pl::PortfolioLoadings)                 = length(pl.loadings)

# ---- loading_matrix ----------------------------------------------------------

"""
    loading_matrix(factor_loadings) -> Matrix{Float64}

Construct an `(n × K)` factor loading matrix from a vector of `FactorLoadings`,
where `n` is the number of organisations and `K` is the number of distinct
factors across all entries. Factors are sorted for a stable column ordering.
"""
function loading_matrix(factor_loadings::Vector{FactorLoadings})
    factor_names = Set{Symbol}()
    for loadings in factor_loadings
        union!(factor_names, names(loadings))
    end
    factors    = sort!(collect(factor_names))
    factor_idx = Dict(f => i for (i, f) in enumerate(factors))
    β          = zeros(length(factor_loadings), length(factors))
    for (i, loadings) in enumerate(factor_loadings)
        for f in names(loadings)
            β[i, factor_idx[f]] = loadings[f]
        end
    end
    return β
end
