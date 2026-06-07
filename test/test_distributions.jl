@testset "SplicedSeverity" begin

    # Anchor case: PERT(0,5,10,4) body, threshold=10, GPD(2,0.5) tail, p_u=0.1.
    # F_b = cdf(PERT, 10) = 1.0, so the body normalisation factor is trivial.
    let body = PERT(0.0, 5.0, 10.0, 4.0), threshold = 10.0, tail = GPD(2.0, 0.5)
        d = SplicedSeverity(body, threshold, tail; p_u = 0.1)

        @testset "constructor" begin
            @test d isa Distributions.ContinuousUnivariateDistribution
            @test d.threshold == 10.0
            @test d.p_u == 0.1
            @test d.F_b ≈ 1.0
        end

        @testset "support" begin
            @test minimum(d) == 0.0
            @test maximum(d) == Inf
            @test insupport(d, 0.0)
            @test insupport(d, 5.0)
            @test insupport(d, 10.0)
            @test insupport(d, 15.0)
            @test !insupport(d, -0.1)
        end

        @testset "logpdf / pdf" begin
            # body side: f(5) = 0.9 × pdf(PERT, 5) / 1.0 = 0.9 × (3/16) = 27/160
            @test pdf(d, 5.0) ≈ 27/160
            @test logpdf(d, 5.0) ≈ log(27/160)
            # tail side: f(12) = 0.1 × pdf(GPD(2,0.5), 2)
            # pdf(GPD(2,0.5), 2) = (1/2)(1 + 0.5·2/2)^{−3} = (1/2)(1.5)^{−3} = 4/27
            @test pdf(d, 12.0) ≈ 0.1 * 4/27
            @test logpdf(d, 12.0) ≈ log(0.1 * 4/27)
            # outside support
            @test pdf(d, -1.0) == 0.0
            @test logpdf(d, -1.0) == -Inf
            # consistency
            for x in [1.0, 5.0, 9.0, 11.0, 15.0]
                @test exp(logpdf(d, x)) ≈ pdf(d, x)
            end
        end

        @testset "cdf" begin
            # body side: cdf(5) = 0.9 × cdf(PERT,5) / 1 = 0.9 × 0.5 = 0.45
            @test cdf(d, 5.0) ≈ 0.45
            # at threshold: 0.9 × F_b / F_b = 0.9
            @test cdf(d, 10.0) ≈ 0.9
            # tail side: 0.9 + 0.1 × cdf(GPD(2,0.5), 2)
            # cdf(GPD(2,0.5), 2) = 1 − (1.5)^{−2} = 5/9
            @test cdf(d, 12.0) ≈ 0.9 + 0.1 * 5/9
            # below support
            @test cdf(d, -1.0) == 0.0
        end

        @testset "quantile" begin
            # body region: q=0.45 → body_p = 0.45/0.9 = 0.5 → quantile(PERT, 0.5) = 5
            @test quantile(d, 0.45) ≈ 5.0
            # boundary: q=0.9 → body_p = 1.0 → quantile(PERT, 1.0) = 10
            @test quantile(d, 0.9) ≈ 10.0
            # tail: q=0.95 → tail_p = 0.5 → 10 + quantile(GPD(2,0.5), 0.5)
            # quantile(GPD(2,0.5), 0.5) = (2/0.5)((0.5)^{−0.5}−1) = 4(√2−1)
            @test quantile(d, 0.95) ≈ 10.0 + 4*(sqrt(2) - 1)
            # round-trip
            for q in [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
                @test cdf(d, quantile(d, q)) ≈ q  atol=1e-10
            end
            @test quantile(d, 0.0) == 0.0
            @test quantile(d, 1.0) == Inf
            @test_throws DomainError quantile(d, -0.1)
        end

        @testset "median" begin
            # cdf(d, 5) = 0.45 < 0.5, so median > 5 (in body region)
            m = median(d)
            @test cdf(d, m) ≈ 0.5  atol=1e-10
        end

        @testset "sampling" begin
            rng = Xoshiro(TEST_SEED)
            samples = rand(rng, d, 100_000)
            @test all(s >= 0.0 for s in samples)
            @test count(s > 10.0 for s in samples) / 100_000 ≈ 0.1  atol=0.01
        end
    end

    @testset "default p_u from body CDF" begin
        # Body extends beyond threshold; p_u = 1 − cdf(body, threshold)
        body = Metalog([0.25, 0.75], [1.0, 3.0])       # unbounded, median = 2
        d = SplicedSeverity(body, 4.0, GPD(1.0, 0.3))  # threshold above median
        @test d.p_u ≈ 1 - cdf(body, 4.0)
        # round-trip quantile / cdf
        for q in [0.1, 0.3, 0.5, 0.7, 0.9]
            @test cdf(d, quantile(d, q)) ≈ q  atol=1e-10
        end
    end

    @testset "constructor errors" begin
        body = PERT(0.0, 5.0, 10.0, 4.0)
        tail = GPD(1.0, 0.5)
        @test_throws ArgumentError SplicedSeverity(body, 10.0, tail; p_u = 0.0)
        @test_throws ArgumentError SplicedSeverity(body, 10.0, tail; p_u = 1.0)
        @test_throws ArgumentError SplicedSeverity(body, 10.0, tail; p_u = -0.1)
        # threshold below body support → F_b = 0
        @test_throws ArgumentError SplicedSeverity(body, -1.0, tail; p_u = 0.1)
    end

end

@testset "Metalog" begin

    @testset "constructor — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        @test d isa Distributions.ContinuousUnivariateDistribution
        @test length(d.a) == 2
        @test isinf(d.bl) && d.bl < 0

        # 2-term case: Q(p) = a₁ + a₂·logit(p)
        # System: a₁ − a₂·log(3) = 1, a₁ + a₂·log(3) = 3  →  a₁=2, a₂=1/log(3)
        @test d.a[1] ≈ 2.0
        @test d.a[2] ≈ 1 / log(3)

        # 3-term fit
        d3 = Metalog([0.1, 0.5, 0.9], [1.0, 3.0, 6.0])
        @test length(d3.a) == 3
        # fitted quantile must recover inputs exactly
        @test Entangle._metalog_quantile(d3.a, 0.1) ≈ 1.0
        @test Entangle._metalog_quantile(d3.a, 0.5) ≈ 3.0
        @test Entangle._metalog_quantile(d3.a, 0.9) ≈ 6.0
    end

    @testset "constructor — semi-bounded lower" begin
        # Fit to (ps=[0.25,0.75], qs=[1,3], lower=0)
        # Target log-qs: [0, log(3)]  →  a₁=log(3)/2, a₂=0.5
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test isfinite(d.bl)
        @test d.bl == 0.0
        @test d.a[1] ≈ log(3) / 2
        @test d.a[2] ≈ 0.5
    end

    @testset "constructor — validation errors" begin
        # wrong length
        @test_throws ArgumentError Metalog([0.25, 0.75], [1.0])
        # too few points
        @test_throws ArgumentError Metalog([0.5], [1.0])
        # p out of (0,1)
        @test_throws ArgumentError Metalog([0.0, 0.75], [1.0, 3.0])
        @test_throws ArgumentError Metalog([0.25, 1.0], [1.0, 3.0])
        # ps not strictly increasing
        @test_throws ArgumentError Metalog([0.75, 0.25], [1.0, 3.0])
        # qs not strictly increasing
        @test_throws ArgumentError Metalog([0.25, 0.75], [3.0, 1.0])
        # qs below lower bound
        @test_throws ArgumentError Metalog([0.25, 0.75], [-1.0, 3.0]; lower = 0.0)
        # invalid (Q' ≤ 0) — decreasing qs produce negative a₂
        # already caught by qs strictly increasing check above
    end

    @testset "support — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        @test minimum(d) == -Inf
        @test maximum(d) == Inf
        @test insupport(d, -1e10)
        @test insupport(d, 0.0)
        @test insupport(d, 1e10)
        @test !insupport(d, Inf)
        @test !insupport(d, NaN)
    end

    @testset "support — semi-bounded lower" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test minimum(d) == 0.0
        @test maximum(d) == Inf
        @test insupport(d, 0.0)
        @test insupport(d, 1.0)
        @test !insupport(d, -0.1)
    end

    @testset "quantile — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        # recovers input quantiles exactly
        @test quantile(d, 0.25) ≈ 1.0
        @test quantile(d, 0.75) ≈ 3.0
        # median = a₁ = 2.0 (logit term vanishes at p=0.5)
        @test quantile(d, 0.5) ≈ 2.0
        # boundary
        @test quantile(d, 0.0) == -Inf
        @test quantile(d, 1.0) == Inf
        @test_throws DomainError quantile(d, -0.1)
        @test_throws DomainError quantile(d, 1.1)
    end

    @testset "quantile — semi-bounded lower" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test quantile(d, 0.25) ≈ 1.0
        @test quantile(d, 0.75) ≈ 3.0
        # Q_sl(0.5) = 0 + exp(log(3)/2) = √3
        @test quantile(d, 0.5) ≈ sqrt(3)
        @test quantile(d, 0.0) == 0.0
        @test quantile(d, 1.0) == Inf
    end

    @testset "cdf — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        # round-trip: cdf(quantile(p)) ≈ p
        for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            @test cdf(d, quantile(d, p)) ≈ p  atol=1e-10
        end
        @test cdf(d, -Inf) == 0.0 || cdf(d, quantile(d, 1e-12)) < 1e-10
    end

    @testset "cdf — semi-bounded lower" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test cdf(d, 0.0) == 0.0
        @test cdf(d, -1.0) == 0.0
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]
            @test cdf(d, quantile(d, p)) ≈ p  atol=1e-10
        end
    end

    @testset "logpdf / pdf — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        # at median x=2: Q'(0.5) = a₂/(0.5·0.5) = 4/log(3)
        # f(2) = log(3)/4
        @test pdf(d, 2.0) ≈ log(3) / 4  atol=1e-8
        @test logpdf(d, 2.0) ≈ log(log(3) / 4)  atol=1e-8
        # consistency: exp(logpdf) == pdf
        for x in [1.0, 2.0, 3.0, 0.0, 4.0]
            @test exp(logpdf(d, x)) ≈ pdf(d, x)  atol=1e-10
        end
    end

    @testset "logpdf / pdf — semi-bounded lower" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test pdf(d, -0.1) == 0.0
        @test logpdf(d, -0.1) == -Inf
        for x in [0.5, 1.0, sqrt(3), 3.0, 5.0]
            @test exp(logpdf(d, x)) ≈ pdf(d, x)  atol=1e-10
        end
    end

    @testset "median" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        @test median(d) ≈ 2.0
        d_sl = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        @test median(d_sl) ≈ sqrt(3)
    end

    @testset "mean" begin
        # 2-term symmetric: Q(p) = 2 + (1/log3)·logit(p)
        # mean = ∫₀¹ Q(p)dp = 2 + 0 = 2 (logit integrates to 0)
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        @test mean(d) ≈ 2.0  atol=1e-4
    end

    @testset "sampling — unbounded" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0])
        rng = Xoshiro(TEST_SEED)
        samples = rand(rng, d, 50_000)
        @test abs(mean(samples) - mean(d)) < 0.05
        # empirical quantiles close to theoretical
        sort!(samples)
        @test abs(samples[Int(0.25 * 50_000)] - 1.0) < 0.05
        @test abs(samples[Int(0.75 * 50_000)] - 3.0) < 0.05
    end

    @testset "sampling — semi-bounded lower" begin
        d = Metalog([0.25, 0.75], [1.0, 3.0]; lower = 0.0)
        rng = Xoshiro(TEST_SEED)
        samples = rand(rng, d, 50_000)
        @test all(s >= 0.0 for s in samples)
        @test abs(samples[Int(0.5 * 50_000)] - sqrt(3)) < 0.1
    end

end

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
