@testset "FactorLoadings" begin

    @testset "construction" begin
        fl = FactorLoadings(aws = 0.4, crowdstrike = 0.3)
        @test fl[:aws]         ≈ 0.4
        @test fl[:crowdstrike] ≈ 0.3
        @test fl[:unknown]     == 0.0
    end

    @testset "empty loadings" begin
        fl = FactorLoadings()
        @test isempty(fl)
        @test norm_sq(fl)              == 0.0
        @test idiosyncratic_weight(fl) ≈ 1.0
    end

    @testset "norm_sq and idiosyncratic_weight" begin
        # 0.4² + 0.3² = 0.16 + 0.09 = 0.25
        fl = FactorLoadings(aws = 0.4, crowdstrike = 0.3)
        @test norm_sq(fl)              ≈ 0.25
        @test idiosyncratic_weight(fl) ≈ sqrt(0.75)
    end

    @testset "boundary: norm_sq exactly 1" begin
        fl = FactorLoadings(aws = 1.0)
        @test norm_sq(fl)              ≈ 1.0
        @test idiosyncratic_weight(fl) ≈ 0.0  atol=1e-15
    end

    @testset "norm constraint violation" begin
        @test_throws ArgumentError FactorLoadings(aws = 0.8, crowdstrike = 0.8)
    end

    @testset "names" begin
        fl = FactorLoadings(ransomware = 0.5, financial_sector = 0.3)
        @test :ransomware       in names(fl)
        @test :financial_sector in names(fl)
        @test length(names(fl)) == 2
    end

end

@testset "PortfolioLoadings" begin

    @testset "construction and insert!" begin
        pl = PortfolioLoadings()
        @test length(pl) == 0

        fl = FactorLoadings(aws = 0.4)
        insert!(pl, :acme, fl)
        @test length(pl)    == 1
        @test haskey(pl, :acme)
        @test pl[:acme][:aws] ≈ 0.4
    end

    @testset "insert! accepts kwargs" begin
        pl = PortfolioLoadings()
        insert!(pl, :acme; aws = 0.4, ransomware = 0.3)
        @test pl[:acme][:aws]       ≈ 0.4
        @test pl[:acme][:ransomware] ≈ 0.3
    end

    @testset "insert! throws on duplicate" begin
        pl = PortfolioLoadings()
        insert!(pl, :acme, FactorLoadings(aws = 0.4))
        @test_throws ArgumentError insert!(pl, :acme, FactorLoadings(ransomware = 0.6))
    end

    @testset "update! replaces and returns old value" begin
        pl = PortfolioLoadings()
        insert!(pl, :acme, FactorLoadings(aws = 0.4))
        (name, old) = update!(pl, :acme, FactorLoadings(ransomware = 0.6))
        @test length(pl)             == 1
        @test pl[:acme][:aws]        == 0.0
        @test pl[:acme][:ransomware] ≈  0.6
        @test name                   == :acme
        @test old[:aws]              ≈  0.4
    end

    @testset "update! throws when absent" begin
        pl = PortfolioLoadings()
        @test_throws ArgumentError update!(pl, :acme, FactorLoadings(aws = 0.4))
    end

    @testset "keys" begin
        pl = PortfolioLoadings()
        insert!(pl, :acme,   FactorLoadings(aws = 0.3))
        insert!(pl, :globex, FactorLoadings(ransomware = 0.5))
        @test :acme   in keys(pl)
        @test :globex in keys(pl)
    end

end

# ---------------------------------------------------------------------------

function _simple_model(name::Symbol)
    FAIRModel(
        name               = name,
        tef                = PERT(0.5, 1.0, 3.0),
        vulnerability      = Beta(2.0, 5.0),
        primary_severity   = PERT(100.0, 1_000.0, 5_000.0),
        secondary_severity = PERT(50.0, 300.0, 1_500.0),
    )
end

# ---------------------------------------------------------------------------

@testset "loading_matrix" begin

    @testset "empty" begin
        β = loading_matrix(FactorLoadings[])
        @test size(β) == (0, 0)
    end

    @testset "single org, single factor" begin
        β = loading_matrix([FactorLoadings(aws = 0.5)])
        @test size(β) == (1, 1)
        @test β[1, 1] ≈ 0.5
    end

    @testset "factor columns are sorted" begin
        β = loading_matrix([FactorLoadings(ransomware = 0.3, aws = 0.4)])
        @test size(β) == (1, 2)
        @test β[1, 1] ≈ 0.4   # aws sorts before ransomware
        @test β[1, 2] ≈ 0.3
    end

    @testset "absent factor gives zero entry" begin
        fl1 = FactorLoadings(aws = 0.4, ransomware = 0.3)
        fl2 = FactorLoadings(aws = 0.6)
        β   = loading_matrix([fl1, fl2])
        @test size(β) == (2, 2)
        @test β[2, 2] == 0.0   # fl2 has no ransomware loading
    end

end

@testset "empirical_quantile" begin

    v = [1.0, 2.0, 3.0, 4.0, 5.0]

    @test empirical_quantile(v, 0.0)   == 1.0
    @test empirical_quantile(v, 1.0)   == 5.0
    @test empirical_quantile(v, 0.5)   == 3.0
    @test empirical_quantile(v, 0.375) ≈  2.5
    @test empirical_quantile(v, -0.5)  == 1.0   # clamp below
    @test empirical_quantile(v, 1.5)   == 5.0   # clamp above

end

@testset "Portfolio" begin

    @testset "construction" begin
        p = Portfolio()
        @test length(p) == 0
    end

    @testset "insert! and accessors" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        @test length(p) == 1
        @test haskey(p, :acme)
        @test :acme in names(p)
        @test !has_loss_samples(p, :acme)
    end

    @testset "insert! throws on duplicate" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        @test_throws ArgumentError insert!(p, _simple_model(:acme))
    end

    @testset "update! invalidates samples" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        calculate_marginal_loss!(p, :acme; n_scenarios = 100, seed = TEST_SEED)
        @test has_loss_samples(p, :acme)
        update!(p, _simple_model(:acme))
        @test !has_loss_samples(p, :acme)
    end

    @testset "update! throws when absent" begin
        p = Portfolio()
        @test_throws ArgumentError update!(p, _simple_model(:acme))
    end

    @testset "update! returns old model" begin
        p = Portfolio()
        m1 = _simple_model(:acme)
        insert!(p, m1)
        old = update!(p, _simple_model(:acme))
        @test old === m1
    end

    @testset "calculate_marginal_loss! unknown name" begin
        p = Portfolio()
        @test_throws ArgumentError calculate_marginal_loss!(p, :unknown; n_scenarios = 100, seed = TEST_SEED)
    end

    @testset "calculate_marginal_loss! stores sorted samples" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        calculate_marginal_loss!(p, :acme; n_scenarios = 1_000, seed = TEST_SEED)
        s = p.sorted_loss_samples[:acme]
        @test length(s)    == 1_000
        @test issorted(s)
        @test all(>=(0.0), s)
    end

    @testset "seed stability" begin
        p1 = Portfolio()
        insert!(p1, _simple_model(:acme))
        calculate_marginal_loss!(p1, :acme; n_scenarios = 500, seed = TEST_SEED)

        p2 = Portfolio()
        insert!(p2, _simple_model(:acme))
        insert!(p2, _simple_model(:globex))
        calculate_marginal_loss!(p2, :acme; n_scenarios = 500, seed = TEST_SEED)

        @test p1.sorted_loss_samples[:acme] == p2.sorted_loss_samples[:acme]
    end

    @testset "calculate_marginal_losses!" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        insert!(p, _simple_model(:globex))
        calculate_marginal_losses!(p; n_scenarios = 500, seed = TEST_SEED)
        @test has_loss_samples(p, :acme)
        @test has_loss_samples(p, :globex)
    end

end

@testset "rand_uniforms" begin

    @testset "StudentTFactorCopula" begin
        copula = StudentTFactorCopula(ν = 4.0)
        β      = [0.4 0.3; 0.5 0.0; 0.0 0.6]

        @testset "returns uniforms in [0, 1]" begin
            U = rand_uniforms(Xoshiro(TEST_SEED), copula, β)
            @test length(U) == 3
            @test all(0.0 .<= U .<= 1.0)
        end

        @testset "reproducible" begin
            @test rand_uniforms(Xoshiro(TEST_SEED), copula, β) ==
                  rand_uniforms(Xoshiro(TEST_SEED), copula, β)
        end

        @testset "empty (n = 0)" begin
            U = rand_uniforms(Xoshiro(TEST_SEED), copula, zeros(0, 0))
            @test isempty(U)
        end
    end

    @testset "GaussianFactorCopula" begin
        copula = GaussianFactorCopula()
        β      = [0.4 0.3; 0.5 0.0; 0.0 0.6]

        @testset "returns uniforms in [0, 1]" begin
            U = rand_uniforms(Xoshiro(TEST_SEED), copula, β)
            @test length(U) == 3
            @test all(0.0 .<= U .<= 1.0)
        end

        @testset "reproducible" begin
            @test rand_uniforms(Xoshiro(TEST_SEED), copula, β) ==
                  rand_uniforms(Xoshiro(TEST_SEED), copula, β)
        end

        @testset "empty (n = 0)" begin
            U = rand_uniforms(Xoshiro(TEST_SEED), copula, zeros(0, 0))
            @test isempty(U)
        end
    end

end

@testset "rand_portfolio_loss" begin

    copula = StudentTFactorCopula()

    function _portfolio()
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        insert!(p, _simple_model(:globex))
        calculate_marginal_losses!(p; n_scenarios = 2_000, seed = TEST_SEED)
        pl = PortfolioLoadings()
        insert!(pl, :acme,   FactorLoadings(aws = 0.4, ransomware = 0.3))
        insert!(pl, :globex, FactorLoadings(aws = 0.5))
        return p, pl
    end

    @testset "error when samples missing" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        @test_throws ArgumentError rand_portfolio_loss(
            TEST_SEED, PortfolioLoadings(), copula, p, 100
        )
    end

    @testset "returns correct length" begin
        p, pl = _portfolio()
        losses = rand_portfolio_loss(TEST_SEED, pl, copula, p, 200)
        @test length(losses) == 200
    end

    @testset "single draw returns scalar" begin
        p, pl = _portfolio()
        loss = rand_portfolio_loss(TEST_SEED, pl, copula, p)
        @test loss isa Float64
        @test loss >= 0.0
    end

    @testset "non-negative" begin
        p, pl = _portfolio()
        losses = rand_portfolio_loss(TEST_SEED, pl, copula, p, 500)
        @test all(>=(0.0), losses)
    end

    @testset "reproducible" begin
        p, pl = _portfolio()
        l1 = rand_portfolio_loss(TEST_SEED, pl, copula, p, 200)
        l2 = rand_portfolio_loss(TEST_SEED, pl, copula, p, 200)
        @test l1 == l2
    end

    @testset "insertion order independence" begin
        p1 = Portfolio()
        insert!(p1, _simple_model(:acme))
        insert!(p1, _simple_model(:globex))
        calculate_marginal_losses!(p1; n_scenarios = 500, seed = TEST_SEED)

        p2 = Portfolio()
        insert!(p2, _simple_model(:globex))
        insert!(p2, _simple_model(:acme))
        calculate_marginal_losses!(p2; n_scenarios = 500, seed = TEST_SEED)

        pl = PortfolioLoadings()
        insert!(pl, :acme,   FactorLoadings(aws = 0.4))
        insert!(pl, :globex, FactorLoadings(aws = 0.3))

        l1 = rand_portfolio_loss(TEST_SEED, pl, copula, p1, 200)
        l2 = rand_portfolio_loss(TEST_SEED, pl, copula, p2, 200)
        @test l1 == l2
    end

    @testset "absent loadings treated as idiosyncratic" begin
        p = Portfolio()
        insert!(p, _simple_model(:acme))
        calculate_marginal_losses!(p; n_scenarios = 500, seed = TEST_SEED)
        losses = rand_portfolio_loss(TEST_SEED, PortfolioLoadings(), copula, p, 200)
        @test length(losses) == 200
        @test all(>=(0.0), losses)
    end

    @testset "no-seed form returns correct length" begin
        p, pl = _portfolio()
        losses = rand_portfolio_loss(pl, copula, p, 100)
        @test length(losses) == 100
        @test all(>=(0.0), losses)
    end

    @testset "independent form returns correct length" begin
        p, _ = _portfolio()
        losses = rand_portfolio_loss(TEST_SEED, p, 200)
        @test length(losses) == 200
        @test all(>=(0.0), losses)
    end

    @testset "independent form single draw returns scalar" begin
        p, _ = _portfolio()
        loss = rand_portfolio_loss(TEST_SEED, p)
        @test loss isa Float64
        @test loss >= 0.0
    end

    @testset "independent form reproducible" begin
        p, _ = _portfolio()
        l1 = rand_portfolio_loss(TEST_SEED, p, 200)
        l2 = rand_portfolio_loss(TEST_SEED, p, 200)
        @test l1 == l2
    end

    @testset "independent no-seed form returns correct length" begin
        p, _ = _portfolio()
        losses = rand_portfolio_loss(p, 100)
        @test length(losses) == 100
        @test all(>=(0.0), losses)
    end

end
