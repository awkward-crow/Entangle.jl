"""
    StudentTFactorCopula(ν = 4)

Student-t factor copula with `ν` degrees of freedom.

Lower `ν` produces stronger tail dependence.
"""
struct StudentTFactorCopula <: FactorCopula
    ν :: Float64
end

StudentTFactorCopula(; ν::Real = 4.0) = StudentTFactorCopula(Float64(ν))

function rand_uniforms(
    rng    :: AbstractRNG,
    copula :: StudentTFactorCopula,
    β      :: Matrix{Float64},
)
    ν     = copula.ν
    n, k  = size(β)
    scale = sqrt(rand(rng, Chisq(ν)) / ν)
    F     = [randn(rng) for _ in 1:k]
    return [
        cdf(TDist(ν), (dot(β[i, :], F) + sqrt(1.0 - sum(abs2, β[i, :])) * randn(rng)) / scale)
        for i in 1:n
    ]
end
