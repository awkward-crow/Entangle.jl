module Entangle

using Distributions
using LinearAlgebra
using Random
using Statistics

include("distributions/GPD.jl")
include("distributions/PERT.jl")
include("distributions/Metalog.jl")
include("distributions/SplicedSeverity.jl")

include("fair/Frequency.jl")
include("fair/Magnitude.jl")
include("fair/FAIRNode.jl")
include("fair/FAIRModel.jl")

include("Metrics.jl")

include("portfolio/FactorModel.jl")

export GPD, PERT, Metalog, SplicedSeverity
export FrequencyModel, rand_rate, rand_count, mean_rate
export MagnitudeModel, rand_components, rand_loss, mean_loss
export FAIRNode, FAIRModel, rand_annual_loss, mean_annual_loss, simulate
export ael, pml, EPCurve, exceedance_probability, STANDARD_RETURN_PERIODS
export FactorLoadings, norm_sq, idiosyncratic_weight
export PortfolioLoadings, add!

end # module Entangle
