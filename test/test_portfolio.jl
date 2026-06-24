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

    @testset "construction and add!" begin
        pl = PortfolioLoadings()
        @test length(pl) == 0

        fl = FactorLoadings(aws = 0.4)
        add!(pl, :acme, fl)
        @test length(pl)    == 1
        @test haskey(pl, :acme)
        @test pl[:acme][:aws] ≈ 0.4
    end

    @testset "overwrite existing entry" begin
        pl = PortfolioLoadings()
        add!(pl, :acme, FactorLoadings(aws = 0.4))
        add!(pl, :acme, FactorLoadings(ransomware = 0.6))
        @test length(pl)        == 1
        @test pl[:acme][:aws]       == 0.0
        @test pl[:acme][:ransomware] ≈ 0.6
    end

    @testset "keys" begin
        pl = PortfolioLoadings()
        add!(pl, :acme,  FactorLoadings(aws = 0.3))
        add!(pl, :globex, FactorLoadings(ransomware = 0.5))
        @test :acme  in keys(pl)
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

    @testset "add! and accessors" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        @test length(p) == 1
        @test haskey(p, :acme)
        @test :acme in names(p)
        @test !has_loss_samples(p, :acme)
    end

    @testset "add! invalidates samples" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        calculate_loss!(p, :acme; n_scenarios = 100, seed = TEST_SEED)
        @test has_loss_samples(p, :acme)
        add!(p, _simple_model(:acme))
        @test !has_loss_samples(p, :acme)
    end

    @testset "calculate_loss! unknown name" begin
        p = Portfolio()
        @test_throws ArgumentError calculate_loss!(p, :unknown; n_scenarios = 100, seed = TEST_SEED)
    end

    @testset "calculate_loss! stores sorted samples" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        calculate_loss!(p, :acme; n_scenarios = 1_000, seed = TEST_SEED)
        s = p.sorted_loss_samples[:acme]
        @test length(s)    == 1_000
        @test issorted(s)
        @test all(>=(0.0), s)
    end

    @testset "seed stability" begin
        p1 = Portfolio()
        add!(p1, _simple_model(:acme))
        calculate_loss!(p1, :acme; n_scenarios = 500, seed = TEST_SEED)

        p2 = Portfolio()
        add!(p2, _simple_model(:acme))
        add!(p2, _simple_model(:globex))
        calculate_loss!(p2, :acme; n_scenarios = 500, seed = TEST_SEED)

        @test p1.sorted_loss_samples[:acme] == p2.sorted_loss_samples[:acme]
    end

    @testset "calculate_losses!" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        add!(p, _simple_model(:globex))
        calculate_losses!(p; n_scenarios = 500, seed = TEST_SEED)
        @test has_loss_samples(p, :acme)
        @test has_loss_samples(p, :globex)
    end

end

@testset "rand_uniforms" begin

    copula = StudentTFactorCopula(ν = 4.0)

    @testset "returns uniforms in [0, 1]" begin
        β = [0.4 0.3; 0.5 0.0; 0.0 0.6]
        U = rand_uniforms(Xoshiro(TEST_SEED), copula, β)
        @test length(U) == 3
        @test all(0.0 .<= U .<= 1.0)
    end

    @testset "reproducible" begin
        β = [0.4 0.3; 0.5 0.0; 0.0 0.6]
        @test rand_uniforms(Xoshiro(TEST_SEED), copula, β) ==
              rand_uniforms(Xoshiro(TEST_SEED), copula, β)
    end

    @testset "empty (n = 0)" begin
        U = rand_uniforms(Xoshiro(TEST_SEED), copula, zeros(0, 0))
        @test isempty(U)
    end

end

@testset "rand_portfolio_losses" begin

    copula = StudentTFactorCopula()

    function _portfolio()
        p = Portfolio()
        add!(p, _simple_model(:acme))
        add!(p, _simple_model(:globex))
        calculate_losses!(p; n_scenarios = 2_000, seed = TEST_SEED)
        pl = PortfolioLoadings()
        add!(pl, :acme,   FactorLoadings(aws = 0.4, ransomware = 0.3))
        add!(pl, :globex, FactorLoadings(aws = 0.5))
        return p, pl
    end

    @testset "error when samples missing" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        @test_throws ArgumentError rand_portfolio_losses(
            TEST_SEED, p, PortfolioLoadings(), copula; n_scenarios = 100
        )
    end

    @testset "returns correct length" begin
        p, pl = _portfolio()
        losses = rand_portfolio_losses(TEST_SEED, p, pl, copula; n_scenarios = 200)
        @test length(losses) == 200
    end

    @testset "non-negative" begin
        p, pl = _portfolio()
        losses = rand_portfolio_losses(TEST_SEED, p, pl, copula; n_scenarios = 500)
        @test all(>=(0.0), losses)
    end

    @testset "reproducible" begin
        p, pl = _portfolio()
        l1 = rand_portfolio_losses(TEST_SEED, p, pl, copula; n_scenarios = 200)
        l2 = rand_portfolio_losses(TEST_SEED, p, pl, copula; n_scenarios = 200)
        @test l1 == l2
    end

    @testset "insertion order independence" begin
        p1 = Portfolio()
        add!(p1, _simple_model(:acme))
        add!(p1, _simple_model(:globex))
        calculate_losses!(p1; n_scenarios = 500, seed = TEST_SEED)

        p2 = Portfolio()
        add!(p2, _simple_model(:globex))
        add!(p2, _simple_model(:acme))
        calculate_losses!(p2; n_scenarios = 500, seed = TEST_SEED)

        pl = PortfolioLoadings()
        add!(pl, :acme,   FactorLoadings(aws = 0.4))
        add!(pl, :globex, FactorLoadings(aws = 0.3))

        l1 = rand_portfolio_losses(TEST_SEED, p1, pl, copula; n_scenarios = 200)
        l2 = rand_portfolio_losses(TEST_SEED, p2, pl, copula; n_scenarios = 200)
        @test l1 == l2
    end

    @testset "absent loadings treated as idiosyncratic" begin
        p = Portfolio()
        add!(p, _simple_model(:acme))
        calculate_losses!(p; n_scenarios = 500, seed = TEST_SEED)
        losses = rand_portfolio_losses(
            TEST_SEED, p, PortfolioLoadings(), copula; n_scenarios = 200
        )
        @test length(losses) == 200
        @test all(>=(0.0), losses)
    end

end
