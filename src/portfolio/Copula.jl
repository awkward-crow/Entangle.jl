abstract type FactorCopula end

"""
    StudentTFactorCopula(ν = 4)

Student-t factor copula with `ν` degrees of freedom. Carries only `ν`; factor
loadings are held in the `PortfolioLoadings` passed to `simulate_portfolio`.

Lower `ν` produces stronger tail dependence.
"""
struct StudentTFactorCopula <: FactorCopula
    ν :: Float64
end

StudentTFactorCopula(; ν::Real = 4.0) = StudentTFactorCopula(Float64(ν))

function StudentTFactorCopula(ν::Real)
    ν > 0 || throw(ArgumentError("degrees of freedom ν must be > 0, got $ν"))
    return StudentTFactorCopula(Float64(ν))
end
