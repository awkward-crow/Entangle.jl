using Test
using Entangle
using Distributions
using Random

function _parse_seed(args, default)
    idx = findfirst(==("--seed"), args)
    idx === nothing ? default : parse(Int, args[idx + 1])
end

const TEST_SEED = _parse_seed(ARGS, 1033142)

@testset "Distributions" begin include("test_distributions.jl") end
@testset "FAIR"          begin include("test_fair.jl")          end
@testset "Metrics"       begin include("test_metrics.jl")       end
@testset "Portfolio"     begin include("test_portfolio.jl")     end
@testset "Scenarios"     begin include("test_scenarios.jl")     end
