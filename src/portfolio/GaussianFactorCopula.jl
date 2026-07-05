"""
    GaussianFactorCopula()

Gaussian factor copula parameterised by the factor loading structure.
"""
struct GaussianFactorCopula <: FactorCopula end

function rand_uniforms(
    rng    :: AbstractRNG,
    copula :: GaussianFactorCopula,
    β      :: Matrix{Float64},
)
    n, k = size(β)
    F    = [randn(rng) for _ in 1:k]
    return [
        cdf(Normal(), dot(β[i, :], F) + sqrt(1.0 - sum(abs2, β[i, :])) * randn(rng))
        for i in 1:n
    ]
end
