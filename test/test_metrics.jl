@testset "simulate" begin
    freq  = FrequencyModel(Gamma(4.0, 0.5), Beta(25.0, 75.0))
    mag   = MagnitudeModel(Exponential(2.0), Exponential(1.0))
    model = FAIRModel("test", FAIRNode(freq, mag))

    @testset "returns Vector{Float64}" begin
        losses = simulate(model; n_scenarios = 500, seed = TEST_SEED)
        @test losses isa Vector{Float64}
        @test length(losses) == 500
        @test all(>=(0.0), losses)
    end

    @testset "reproducibility" begin
        l1 = simulate(model; n_scenarios = 200, seed = TEST_SEED)
        l2 = simulate(model; n_scenarios = 200, seed = TEST_SEED)
        @test l1 == l2
    end
end

@testset "ael" begin
    # mean_rate = E[λ]·E[V] = 2.0 × 0.25 = 0.5
    # mean_loss = 2.0 + 1.0 = 3.0  →  AEL = 1.5
    freq  = FrequencyModel(Gamma(4.0, 0.5), Beta(25.0, 75.0))
    mag   = MagnitudeModel(Exponential(2.0), Exponential(1.0))
    model = FAIRModel("test_ael", FAIRNode(freq, mag))

    losses = simulate(model; n_scenarios = 200_000, seed = TEST_SEED)
    @test ael(losses) ≈ 1.5  atol=0.05
end

@testset "pml" begin
    freq  = FrequencyModel(Gamma(4.0, 0.5), Beta(25.0, 75.0))
    mag   = MagnitudeModel(Exponential(2.0), Exponential(1.0))
    model = FAIRModel("test_pml", FAIRNode(freq, mag))
    losses = simulate(model; n_scenarios = 100_000, seed = TEST_SEED)

    @testset "monotone in return period" begin
        @test pml(losses, 10) <= pml(losses, 100)
        @test pml(losses, 100) <= pml(losses, 1000)
    end

    @testset "matches quantile directly" begin
        @test pml(losses, 2) ≈ quantile(losses, 0.5)
        @test pml(losses, 100) ≈ quantile(losses, 0.99)
    end

    @testset "invalid return period" begin
        @test_throws ArgumentError pml(losses, 0.5)
        @test_throws ArgumentError pml(losses, 1.0)
    end
end

@testset "exceedance_probability" begin
    freq  = FrequencyModel(Gamma(4.0, 0.5), Beta(25.0, 75.0))
    mag   = MagnitudeModel(Exponential(2.0), Exponential(1.0))
    model = FAIRModel("test_ep", FAIRNode(freq, mag))
    losses = simulate(model; n_scenarios = 50_000, seed = TEST_SEED)

    ep = exceedance_probability(losses)

    @test ep isa EPCurve
    @test ep.return_periods == STANDARD_RETURN_PERIODS
    @test length(ep.thresholds) == length(STANDARD_RETURN_PERIODS)
    @test issorted(ep.thresholds)
    @test all(>=(0.0), ep.thresholds)

    @testset "custom return periods" begin
        ep2 = exceedance_probability(losses; return_periods = [10.0, 100.0])
        @test ep2.return_periods == [10.0, 100.0]
        @test ep2.thresholds[1] ≈ pml(losses, 10)
        @test ep2.thresholds[2] ≈ pml(losses, 100)
    end
end
