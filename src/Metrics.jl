"""
    ael(losses) -> Float64

Annual Expected Loss: the mean of the simulated loss distribution.
"""

ael(losses::AbstractVector{<:Real}) = mean(losses)

"""
    pml(losses, return_period) -> Float64

Probable Maximum Loss at `return_period` years: the (1 − 1/T) quantile of
the simulated loss distribution.
"""

function pml(losses::AbstractVector{<:Real}, return_period::Real)
    return_period > 1 ||
        throw(ArgumentError("return_period must be > 1, got $return_period"))
    return quantile(losses, 1 - 1 / return_period)
end

"""
    EPCurve

Exceedance probability curve at a set of return periods.

- `return_periods`: return periods in years
- `thresholds`: loss at each return period (i.e. PML values)
"""

struct EPCurve
    return_periods::Vector{Float64}
    thresholds::Vector{Float64}
end

const RETURN_PERIODS = [5.0, 10.0, 20.0, 50.0, 100.0]

"""
    exceedance_probability(losses; return_periods=STANDARD_RETURN_PERIODS) -> EPCurve

Compute the exceedance probability curve from a vector of simulated losses.
"""

function exceedance_probability(
    losses::AbstractVector{<:Real};
    return_periods::AbstractVector{<:Real} = RETURN_PERIODS,
)
    rp = collect(Float64, return_periods)
    return EPCurve(rp, [pml(losses, T) for T in rp])
end
