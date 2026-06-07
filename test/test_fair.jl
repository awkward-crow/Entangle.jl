@testset "Frequency" begin

    # Standard case: Gamma TEF, Beta vulnerability
    # E[λ] = 2×1.5 = 3.0,  E[V] = 2/10 = 0.2  →  mean_rate = 0.6
    let tef = Gamma(2.0, 1.5), vuln = Beta(2.0, 8.0)
        m = FrequencyModel(tef, vuln)

        @testset "constructor" begin
            @test m.tef === tef
            @test m.vulnerability === vuln
        end

        @testset "mean_rate" begin
            @test mean_rate(m) ≈ 0.6
        end

        @testset "rand_rate" begin
            rng = Xoshiro(TEST_SEED)
            r = rand_rate(rng, m)
            @test r isa Float64
            @test r >= 0.0

            # E[rate] ≈ mean_rate;  SE ≈ √(Var[λV]/n) ≈ 0.002
            rng = Xoshiro(TEST_SEED)
            rates = [rand_rate(rng, m) for _ in 1:100_000]
            @test mean(rates) ≈ 0.6  atol=0.01
        end

        @testset "rand_count" begin
            rng = Xoshiro(TEST_SEED)
            n = rand_count(rng, m)
            @test n isa Integer
            @test n >= 0

            rng = Xoshiro(TEST_SEED)
            counts = [rand_count(rng, m) for _ in 1:100_000]

            # E[N] = mean_rate
            @test mean(counts) ≈ 0.6  atol=0.02

            # Var[N] = E[λV] + Var[λV]
            # E[λ²] = Var[λ] + E[λ]² = 4.5 + 9.0 = 13.5
            # E[V²] = a(a+1)/((a+b)(a+b+1)) = 6/110 = 3/55
            # Var[λV] = 13.5 × 3/55 − 0.36 ≈ 0.3764
            # Var[N]  = 0.6 + 0.3764 ≈ 0.9764;  SE of var estimate ≈ 0.004
            @test var(counts)  ≈ 0.9764  atol=0.05

            # overdispersion: parameter uncertainty inflates variance beyond Poisson
            @test var(counts) > mean(counts)
        end
    end

    @testset "degenerate: near-fixed λ and V → pure Poisson limit" begin
        # λ₀ = 4.0, v₀ = 0.25  →  effective rate ≈ 1.0
        # Tight Gamma and Beta approximate Dirac masses
        m = FrequencyModel(Gamma(40_000.0, 1e-4), Beta(25_000.0, 75_000.0))
        @test mean_rate(m) ≈ 1.0  atol=1e-5

        rng = Xoshiro(TEST_SEED)
        counts = [rand_count(rng, m) for _ in 1:50_000]

        # Should approximate Poisson(1.0): mean ≈ 1, var ≈ 1
        @test mean(counts) ≈ 1.0  atol=0.02
        @test var(counts)  ≈ 1.0  atol=0.05
    end

    @testset "Poisson-Gamma mixture → NegativeBinomial marginal" begin
        # λ ~ Gamma(3, 1),  V ≈ 0.5 (near-fixed via tight Beta)
        # Marginal: N ~ NegBin(3, 2/3)  [p = 1/(1 + v·θ) = 1/1.5 = 2/3]
        # P(N=0) = (2/3)³ = 8/27
        # P(N=1) = 3·(2/3)³·(1/3) = 8/27
        # P(N=2) = 6·(2/3)³·(1/3)² = 16/81
        # P(N=3) = 10·(2/3)³·(1/3)³ = 80/729
        m  = FrequencyModel(Gamma(3.0, 1.0), Beta(25_000.0, 25_000.0))
        nb = NegativeBinomial(3.0, 2/3)

        rng = Xoshiro(TEST_SEED)
        counts = [rand_count(rng, m) for _ in 1:200_000]

        for k in 0:3
            @test mean(counts .== k) ≈ pdf(nb, k)  atol=0.005
        end
    end

end

@testset "Magnitude" begin

    # Exponential components: exact means, easy to verify
    # primary ~ Exp(2.0) → mean = 2.0
    # secondary ~ Exp(1.0) → mean = 1.0
    # total mean = 3.0
    let pri = Exponential(2.0), sec = Exponential(1.0)
        m = MagnitudeModel(pri, sec)

        @testset "constructor" begin
            @test m.primary === pri
            @test m.secondary === sec
        end

        @testset "mean_loss" begin
            @test mean_loss(m) ≈ 3.0
        end

        @testset "rand_components" begin
            rng = Xoshiro(TEST_SEED)
            (p, s) = rand_components(rng, m)
            @test p isa Float64
            @test s isa Float64
            @test p >= 0.0
            @test s >= 0.0

            # simulate and check marginal means
            rng = Xoshiro(TEST_SEED)
            ps = [rand_components(rng, m) for _ in 1:100_000]
            @test mean(first.(ps)) ≈ 2.0  atol=0.02
            @test mean(last.(ps))  ≈ 1.0  atol=0.02
        end

        @testset "rand_loss" begin
            rng = Xoshiro(TEST_SEED)
            l = rand_loss(rng, m)
            @test l isa Float64
            @test l >= 0.0

            rng = Xoshiro(TEST_SEED)
            losses = [rand_loss(rng, m) for _ in 1:100_000]
            @test mean(losses) ≈ 3.0  atol=0.02
        end

        @testset "components sum to total" begin
            rng = Xoshiro(TEST_SEED)
            for _ in 1:1_000
                rng2 = copy(rng)
                (p, s) = rand_components(rng, m)
                l = rand_loss(rng2, m)
                @test p + s ≈ l
            end
        end
    end

    @testset "SplicedSeverity components" begin
        pri = SplicedSeverity(LogNormal(0.0, 1.0), 5.0, GPD(2.0, 0.4); p_u=0.1)
        sec = SplicedSeverity(Exponential(1.0), 3.0, GPD(1.0, 0.2); p_u=0.05)
        m = MagnitudeModel(pri, sec)

        @test mean_loss(m) ≈ mean(pri) + mean(sec)  atol=1e-10

        rng = Xoshiro(TEST_SEED)
        losses = [rand_loss(rng, m) for _ in 1:50_000]
        @test mean(losses) ≈ mean_loss(m)  atol=(mean_loss(m) * 0.03)
        @test all(>=(0.0), losses)
    end

end

@testset "FAIRNode" begin

    # freq: mean_rate = E[λ]·E[V] = 3.0 × 0.25 = 0.75
    # mag:  mean_loss = 2.0 + 1.0 = 3.0
    # E[annual loss] = 0.75 × 3.0 = 2.25  (Wald's identity)
    let freq = FrequencyModel(Gamma(3.0, 1.0), Beta(25.0, 75.0)),
        mag  = MagnitudeModel(Exponential(2.0), Exponential(1.0))

        node = FAIRNode(freq, mag)

        @testset "constructor" begin
            @test node.frequency === freq
            @test node.magnitude === mag
        end

        @testset "mean_annual_loss" begin
            @test mean_annual_loss(node) ≈ 2.25
        end

        @testset "rand_annual_loss type and sign" begin
            rng = Xoshiro(TEST_SEED)
            l = rand_annual_loss(rng, node)
            @test l isa Float64
            @test l >= 0.0
        end

        @testset "simulated mean ≈ Wald expectation" begin
            rng = Xoshiro(TEST_SEED)
            losses = [rand_annual_loss(rng, node) for _ in 1:200_000]
            @test mean(losses) ≈ 2.25  atol=0.05
            @test all(>=(0.0), losses)
        end

        @testset "zero-frequency node produces zero loss" begin
            # near-zero rate: mean_rate ≈ 1e-6
            node0 = FAIRNode(
                FrequencyModel(Gamma(1.0, 1e-6), Beta(1.0, 1.0)),
                mag,
            )
            rng = Xoshiro(TEST_SEED)
            losses = [rand_annual_loss(rng, node0) for _ in 1:10_000]
            @test mean(losses) ≈ 0.0  atol=0.001
        end
    end

end

@testset "FAIRModel" begin

    let freq = FrequencyModel(Gamma(2.0, 1.0), Beta(10.0, 10.0)),
        mag  = MagnitudeModel(Exponential(5.0), Exponential(2.0))

        node = FAIRNode(freq, mag)

        # single-node constructor
        model = FAIRModel("test_org", node)

        @testset "constructor (single node)" begin
            @test model.name == "test_org"
            @test length(model.nodes) == 1
            @test model.nodes[1] === node
        end

        @testset "mean_annual_loss single node" begin
            @test mean_annual_loss(model) ≈ mean_annual_loss(node)
        end

        @testset "rand_annual_loss type and sign" begin
            rng = Xoshiro(TEST_SEED)
            l = rand_annual_loss(rng, model)
            @test l isa Float64
            @test l >= 0.0
        end

        # multi-node model: total mean = sum of node means
        node2 = FAIRNode(
            FrequencyModel(Gamma(1.0, 0.5), Beta(5.0, 95.0)),
            MagnitudeModel(Exponential(10.0), Exponential(1.0)),
        )
        model2 = FAIRModel("test_org_multi", [node, node2])

        @testset "constructor (multi-node)" begin
            @test length(model2.nodes) == 2
        end

        @testset "mean_annual_loss multi-node = sum" begin
            @test mean_annual_loss(model2) ≈
                mean_annual_loss(node) + mean_annual_loss(node2)
        end

        @testset "simulated mean ≈ analytical mean" begin
            rng = Xoshiro(TEST_SEED)
            losses = [rand_annual_loss(rng, model2) for _ in 1:200_000]
            @test mean(losses) ≈ mean_annual_loss(model2)  atol=0.1
            @test all(>=(0.0), losses)
        end
    end

end
