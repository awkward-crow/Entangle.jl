"""
    ael(losses) -> Float64

Annual Expected Loss: the mean of the simulated loss distribution.
"""
ael(losses::AbstractVector{<:Real}) = mean(losses)
