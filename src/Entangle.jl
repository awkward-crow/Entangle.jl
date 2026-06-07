module Entangle

using Distributions
using LinearAlgebra
using Random

include("distributions/GPD.jl")
include("distributions/PERT.jl")
include("distributions/Metalog.jl")
include("distributions/SplicedSeverity.jl")

include("fair/Frequency.jl")

export GPD, PERT, Metalog, SplicedSeverity
export FrequencyModel, rand_rate, rand_count, mean_rate

end # module Entangle
