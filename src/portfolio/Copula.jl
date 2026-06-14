abstract type FactorCopula end

"""
    rand_uniforms(rng, copula, β) -> Vector{Float64}

Sample `n` dependent U[0,1] random variables from the copula with
`(n × K)` factor loading matrix `β`.

Implement this method for each concrete `FactorCopula` subtype.
"""
function rand_uniforms end
