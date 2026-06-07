module Entangle

using Distributions
using LinearAlgebra
using Random

include("distributions/GPD.jl")
include("distributions/PERT.jl")
include("distributions/Metalog.jl")
include("distributions/SplicedSeverity.jl")

include("fair/Frequency.jl")
include("fair/Magnitude.jl")

export GPD, PERT, Metalog, SplicedSeverity
export FrequencyModel, rand_rate, rand_count, mean_rate
export MagnitudeModel, rand_components, rand_loss, mean_loss

end # module Entangle
