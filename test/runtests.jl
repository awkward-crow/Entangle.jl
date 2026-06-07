using Test
using Entangle
using Distributions
using Random

function _parse_seed(args, default)
    idx = findfirst(==("--seed"), args)
    idx === nothing ? default : parse(Int, args[idx + 1])
end

const TEST_SEED = _parse_seed(ARGS, 1033142)

@testset "Entangle.jl" begin
    include("test_distributions.jl")
    include("test_fair.jl")
    include("test_metrics.jl")
end
