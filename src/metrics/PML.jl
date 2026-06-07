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
