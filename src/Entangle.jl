module Entangle

using Distributions
using Random

include("distributions/GPD.jl")
include("distributions/PERT.jl")

export GPD, PERT

end # module Entangle
