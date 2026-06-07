const STANDARD_RETURN_PERIODS = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 250.0, 500.0, 1000.0]

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

"""
    exceedance_probability(losses; return_periods=STANDARD_RETURN_PERIODS) -> EPCurve

Compute the exceedance probability curve from a vector of simulated losses.
"""
function exceedance_probability(
    losses::AbstractVector{<:Real};
    return_periods::AbstractVector{<:Real} = STANDARD_RETURN_PERIODS,
)
    rp = collect(Float64, return_periods)
    return EPCurve(rp, [pml(losses, T) for T in rp])
end
