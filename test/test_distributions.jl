@testset "PERT" begin

    @testset "constructor" begin
        @test PERT(0.0, 5.0, 10.0) isa Distributions.ContinuousUnivariateDistribution
        @test PERT(0.0, 5.0, 10.0, 4.0).a === 0.0
        @test PERT(0.0, 5.0, 10.0, 4.0).b === 5.0
        @test PERT(0.0, 5.0, 10.0, 4.0).c === 10.0
        @test PERT(0.0, 5.0, 10.0, 4.0).λ === 4.0
        # default λ = 4
        @test PERT(0.0, 5.0, 10.0).λ === 4.0
        # integer args promote to float
        @test PERT(0, 5, 10).a isa Float64
        @test PERT(0, 5, 10, 4).λ isa Float64
        # Float32 preserved
        @test PERT(0.0f0, 5.0f0, 10.0f0) isa PERT{Float32}
        # invalid params
        @test_throws ArgumentError PERT(5.0, 5.0, 10.0)   # a == b
        @test_throws ArgumentError PERT(5.0, 6.0, 6.0)   # b == c
        @test_throws ArgumentError PERT(5.0, 3.0, 10.0)  # b < a
        @test_throws ArgumentError PERT(0.0, 8.0, 5.0)   # c < b
        @test_throws ArgumentError PERT(0.0, 5.0, 10.0, 0.0)  # λ = 0
        @test_throws ArgumentError PERT(0.0, 5.0, 10.0, -1.0) # λ < 0
    end

    @testset "Beta parameters" begin
        # PERT(0, 5, 10, 4): r=10, α₁ = 1 + 4*5/10 = 3, α₂ = 3 (symmetric)
        d = PERT(0.0, 5.0, 10.0, 4.0)
        @test d.α₁ ≈ 3.0
        @test d.α₂ ≈ 3.0
        # PERT(0, 2, 10, 4): r=10, α₁ = 1 + 4*2/10 = 1.8, α₂ = 1 + 4*8/10 = 4.2
        d2 = PERT(0.0, 2.0, 10.0, 4.0)
        @test d2.α₁ ≈ 1.8
        @test d2.α₂ ≈ 4.2
    end

    @testset "support" begin
        d = PERT(0.0, 5.0, 10.0)
        @test minimum(d) == 0.0
        @test maximum(d) == 10.0
        @test insupport(d, 0.0)
        @test insupport(d, 5.0)
        @test insupport(d, 10.0)
        @test !insupport(d, -0.1)
        @test !insupport(d, 10.1)
        # shifted support
        d2 = PERT(2.0, 4.0, 8.0)
        @test minimum(d2) == 2.0
        @test maximum(d2) == 8.0
        @test !insupport(d2, 1.9)
        @test insupport(d2, 2.0)
    end

    @testset "logpdf / pdf" begin
        # PERT(0, 5, 10, 4): Beta(3, 3) scaled to [0, 10]
        # pdf at x=5: Beta(3,3) pdf at u=0.5 divided by range (10)
        # B(3,3) = 2!*2!/5! = 1/30, so beta_pdf(0.5) = 0.5^2 * 0.5^2 * 30 = 1.875
        # pert_pdf(5) = 1.875 / 10 = 0.1875 = 3/16
        d = PERT(0.0, 5.0, 10.0, 4.0)
        @test logpdf(d, 5.0) ≈ log(3/16)
        @test pdf(d, 5.0) ≈ 3/16
        # outside support
        @test pdf(d, -0.1) == 0.0
        @test pdf(d, 10.1) == 0.0
        @test logpdf(d, -0.1) == -Inf
        # round-trip: exp(logpdf) == pdf
        for x in [1.0, 3.0, 5.0, 7.0, 9.0]
            @test exp(logpdf(d, x)) ≈ pdf(d, x)
        end
    end

    @testset "cdf" begin
        d = PERT(0.0, 5.0, 10.0, 4.0)
        @test cdf(d, 0.0) == 0.0
        @test cdf(d, 10.0) == 1.0
        # symmetric: cdf at mode = 0.5
        @test cdf(d, 5.0) ≈ 0.5
        # below minimum and above maximum clamp
        @test cdf(d, -1.0) == 0.0
        @test cdf(d, 11.0) == 1.0
        # cdf is monotone
        xs = [1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0]
        cdfs = cdf.(Ref(d), xs)
        @test all(diff(cdfs) .> 0)
    end

    @testset "quantile" begin
        d = PERT(0.0, 5.0, 10.0, 4.0)
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == 10.0
        @test quantile(d, 0.5) ≈ 5.0   # symmetric
        # round-trip: cdf(quantile(p)) ≈ p
        for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            @test cdf(d, quantile(d, p)) ≈ p
        end
        # asymmetric case round-trips too
        d2 = PERT(0.0, 2.0, 10.0, 4.0)
        for p in [0.1, 0.5, 0.9]
            @test cdf(d2, quantile(d2, p)) ≈ p
        end
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    @testset "moments" begin
        # PERT(0, 5, 10, 4): mean = (0 + 4*5 + 10)/6 = 5, var = 5*5/7 = 25/7
        d = PERT(0.0, 5.0, 10.0, 4.0)
        @test mean(d) ≈ 5.0
        @test var(d) ≈ 25/7
        @test std(d) ≈ sqrt(25/7)
        @test mode(d) == 5.0
        @test median(d) ≈ 5.0   # symmetric

        # PERT(0, 2, 10, 4): mean = (0 + 8 + 10)/6 = 3, var = (3-0)*(10-3)/7 = 3
        d2 = PERT(0.0, 2.0, 10.0, 4.0)
        @test mean(d2) ≈ 3.0
        @test var(d2) ≈ 3.0
        @test mode(d2) == 2.0

        # non-default λ: PERT(0, 5, 10, 2) → mean = (0+2*5+10)/4 = 20/4 = 5
        d3 = PERT(0.0, 5.0, 10.0, 2.0)
        @test mean(d3) ≈ 5.0
        @test var(d3) ≈ 25/5   # 5*5/(2+3) = 5.0

        # std consistency
        @test std(d2) ≈ sqrt(var(d2))
    end

    @testset "sampling" begin
        d = PERT(0.0, 5.0, 10.0, 4.0)
        rng = Xoshiro(TEST_SEED)
        samples = rand(rng, d, 100_000)
        # all samples within support
        @test all(0.0 .<= samples .<= 10.0)
        # sample mean converges to analytical mean
        @test abs(mean(samples) - mean(d)) < 0.05
        # asymmetric case: mean < mode
        d2 = PERT(0.0, 2.0, 10.0, 4.0)
        samples2 = rand(Xoshiro(TEST_SEED), d2, 100_000)
        @test all(0.0 .<= samples2 .<= 10.0)
        @test abs(mean(samples2) - mean(d2)) < 0.05
    end

end

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
