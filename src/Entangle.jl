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

include("metrics/AEL.jl")
include("metrics/PML.jl")
include("metrics/EPCurve.jl")

# Re-export common Distributions.jl types so users only need `using Entangle`
export Beta, Gamma, LogNormal, Exponential, Normal, Uniform, Poisson

export GPD, PERT, Metalog, SplicedSeverity
export FrequencyModel, rand_rate, rand_count, mean_rate
export MagnitudeModel, rand_components, rand_loss, mean_loss
export FAIRNode, FAIRModel, rand_annual_loss, mean_annual_loss, simulate
export ael, pml, EPCurve, exceedance_probability, STANDARD_RETURN_PERIODS

end # module Entangle
