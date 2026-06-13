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

    @testset "keys" begin
        fl = FactorLoadings(ransomware = 0.5, financial_sector = 0.3)
        @test :ransomware      in keys(fl)
        @test :financial_sector in keys(fl)
        @test length(keys(fl)) == 2
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
