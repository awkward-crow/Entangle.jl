@testset "GPD" begin

    @testset "constructor" begin
        @test GPD(1.0, 0.5) isa Distributions.ContinuousUnivariateDistribution
        @test GPD(1.0, 0.5).σ === 1.0
        @test GPD(1.0, 0.5).ξ === 0.5
        # integer args promote to float
        @test GPD(1, 0).σ isa Float64
        # mixed types promote
        @test GPD(1.0f0, 0.5f0) isa GPD{Float32}
        # invalid scale
        @test_throws ArgumentError GPD(0.0, 0.5)
        @test_throws ArgumentError GPD(-1.0, 0.5)
    end

    @testset "support" begin
        # ξ ≥ 0: unbounded above
        @test minimum(GPD(1.0, 0.5)) == 0.0
        @test maximum(GPD(1.0, 0.5)) == Inf
        @test maximum(GPD(1.0, 0.0)) == Inf

        # ξ < 0: bounded at -σ/ξ
        @test minimum(GPD(2.0, -0.5)) == 0.0
        @test maximum(GPD(2.0, -0.5)) == 4.0   # -2 / -0.5

        # insupport
        @test insupport(GPD(1.0, 0.5), 0.0)
        @test insupport(GPD(1.0, 0.5), 1e6)
        @test !insupport(GPD(1.0, 0.5), -0.1)
        @test insupport(GPD(2.0, -0.5), 3.9)
        @test !insupport(GPD(2.0, -0.5), 4.1)
    end

    @testset "logpdf / pdf" begin
        # GPD(1, 0.5) at x=2: pdf = (1 + 0.5·2)^{-3} = 2^{-3} = 0.125
        d = GPD(1.0, 0.5)
        @test logpdf(d, 2.0) ≈ -3log(2)
        @test pdf(d, 2.0) ≈ 0.125
        @test pdf(d, -1.0) == 0.0
        @test logpdf(d, -1.0) == -Inf

        # ξ = 0: Exponential(1) — pdf(1) = exp(-1)
        d0 = GPD(1.0, 0.0)
        @test logpdf(d0, 1.0) ≈ -1.0
        @test pdf(d0, 1.0) ≈ exp(-1)

        # ξ < 0, bounded: GPD(2, -0.5) at x=2 → pdf = (1/2)(1 − 0.5)^1 = 0.25
        dn = GPD(2.0, -0.5)
        @test pdf(dn, 2.0) ≈ 0.25
        @test pdf(dn, 4.1) == 0.0   # outside support
    end

    @testset "cdf / ccdf" begin
        # GPD(1, 0.5) at x=2: cdf = 1 − (1 + 0.5·2)^{-2} = 1 − 0.25 = 0.75
        d = GPD(1.0, 0.5)
        @test cdf(d, 2.0) ≈ 0.75
        @test ccdf(d, 2.0) ≈ 0.25
        @test cdf(d, 0.0) == 0.0
        @test ccdf(d, 0.0) == 1.0

        # ξ = 0: Exponential(1) — cdf(1) = 1 − exp(−1)
        d0 = GPD(1.0, 0.0)
        @test cdf(d0, 1.0) ≈ 1 - exp(-1)
        @test ccdf(d0, 1.0) ≈ exp(-1)

        # ξ < 0: cdf at upper bound = 1
        dn = GPD(2.0, -0.5)
        @test cdf(dn, 0.0) == 0.0
        @test cdf(dn, 4.0) == 1.0
        @test cdf(dn, 2.0) ≈ 0.75   # 1 − (1 − 0.5)^2 = 0.75

        # logccdf consistency
        @test logccdf(d, 2.0) ≈ log(ccdf(d, 2.0))
    end

    @testset "quantile" begin
        d = GPD(1.0, 0.5)
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == Inf

        # round-trip: cdf(quantile(p)) ≈ p
        for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
            @test cdf(d, quantile(d, p)) ≈ p
        end

        # ξ = 0
        d0 = GPD(1.0, 0.0)
        for p in [0.1, 0.5, 0.9]
            @test cdf(d0, quantile(d0, p)) ≈ p
        end

        # ξ < 0: quantile(1) = upper bound
        dn = GPD(2.0, -0.5)
        @test quantile(dn, 1.0) ≈ 4.0
        @test quantile(dn, 0.0) == 0.0

        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    @testset "moments" begin
        # mean = σ/(1 − ξ); var = σ²/((1−ξ)²(1−2ξ))
        d = GPD(2.0, -0.5)
        @test mean(d) ≈ 4/3
        @test var(d) ≈ 8/9

        # ξ ≥ 1: infinite mean
        @test mean(GPD(1.0, 1.0)) == Inf
        @test mean(GPD(1.0, 1.5)) == Inf

        # ξ ≥ 0.5: infinite variance
        @test var(GPD(1.0, 0.5)) == Inf
        @test var(GPD(1.0, 0.8)) == Inf

        # std consistency
        d2 = GPD(2.0, 0.3)
        @test std(d2) ≈ sqrt(var(d2))

        # mode is always 0
        @test mode(GPD(1.0, 0.5)) == 0.0
        @test mode(GPD(2.0, -0.5)) == 0.0

        # median round-trip
        d3 = GPD(1.0, 0.5)
        @test cdf(d3, median(d3)) ≈ 0.5
    end

    @testset "sampling" begin
        # GPD(1, 0.3): finite mean and variance → sample mean converges
        d = GPD(1.0, 0.3)
        rng = Xoshiro(TEST_SEED)
        samples = rand(rng, d, 200_000)

        @test all(s >= 0.0 for s in samples)
        @test abs(mean(samples) - mean(d)) < 0.05

        # ξ < 0: all samples within support
        dn = GPD(2.0, -0.5)
        samples_n = rand(Xoshiro(TEST_SEED), dn, 10_000)
        @test all(0.0 .<= samples_n .<= 4.0)
    end

end
